# python -m eval_sim.eval_rdt_maniskill --pretrained_path mp_rank_00_model_states.pt --env-id StackCube-v1 --lang_embeddings_path text_embed_StackCube-v1.pt
from typing import Callable, List, Type
import sys
sys.path.append('/')
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
from scripts.maniskill_model import create_model, RoboticDiffusionTransformerModel
import torch
from collections import deque
from PIL import Image
import cv2
import os
import random

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1",
                        help="ManiSkill environment ID.")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb",
                        help="Observation mode (e.g., 'rgb').")
    parser.add_argument("-n", "--num-traj", type=int, default=25,
                        help="Number of trajectories to test.")
    parser.add_argument("--only-count-success", action="store_true",
                        help="If set, collect until num_traj successes and only save those.")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto",
                        help="Simulation backend: 'auto' | 'cpu' | 'gpu'.")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        help="Render mode: 'sensors' or 'rgb_array'.")
    parser.add_argument("--shader", default="default", type=str,
                        help="Shader: 'default' | 'rt' | 'rt-fast'.")
    parser.add_argument("--num-procs", type=int, default=1,
                        help="Unused here; CPU multiprocessing for replay.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to the pretrained RDT model checkpoint.")
    parser.add_argument("--random_seed", type=int, default=0,
                        help="Random seed.")
    # NEW: force using a precomputed language embedding; this bypasses T5
    parser.add_argument("--lang_embeddings_path", type=str, required=True,
                        help="Path to precomputed language embedding .pt (e.g., text_embed_StackCube-v1.pt).")
    return parser.parse_args(args)

# Map of default instructions (kept for reference; we won't encode on the fly)
task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}

def set_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_text_embed(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Load a precomputed language embedding from a .pt file.
    Handles multiple formats:
      - dict with key 'embeddings' or 'embedding'
      - dict with any tensor value
      - raw torch.Tensor
    Normalizes shape to [B, T, D]:
      - [D]       -> [1, 1, D]
      - [T, D]    -> [1, T, D]
      - [B, T, D] -> as-is
    """
    obj = torch.load(path, map_location=device)

    # Extract tensor
    if isinstance(obj, dict):
        emb = obj.get("embeddings") or obj.get("embedding")
        if emb is None:
            emb = next((v for v in obj.values() if isinstance(v, torch.Tensor)), None)
        if emb is None:
            raise ValueError(f"Could not find a tensor in {path}. Keys: {list(obj.keys())}")
    elif isinstance(obj, torch.Tensor):
        emb = obj
    else:
        raise TypeError(f"Unsupported embedding type in {path}: {type(obj)}")

    # Normalize shapes
    if emb.ndim == 1:        # [D]
        emb = emb.unsqueeze(0).unsqueeze(0)
    elif emb.ndim == 2:      # [T, D]
        emb = emb.unsqueeze(0)
    elif emb.ndim == 3:      # [B, T, D]
        pass
    else:
        raise ValueError(
            f"Expected shape [D], [T, D], or [B, T, D]; got {tuple(emb.shape)} from {path}"
        )

    return emb.to(device=device, dtype=dtype)

def main():
    args = parse_args()
    set_seeds(args.random_seed)

    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        max_episode_steps=400,
    )

    # Load config
    config_path = 'configs/base.yaml'
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # IMPORTANT: Do NOT load T5. Tell the model we're using precomputed embeddings.
    pretrained_text_encoder_name_or_path = "precomputed"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    pretrained_path = args.pretrained_path

    policy = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=pretrained_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,  # <- bypass text encoder
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path
    )

    # Device/dtype for embeddings (match model)
    device = next(policy.parameters()).device
    dtype = torch.bfloat16

    # Always use precomputed language embedding; never call encode_instruction (which would need T5)
    if not os.path.exists(args.lang_embeddings_path):
        raise FileNotFoundError(
            f"--lang_embeddings_path not found: {args.lang_embeddings_path}\n"
            f"Expected something like 'lang_embeds/text_embed_{env_id}.pt'"
        )
    text_embed = load_text_embed(args.lang_embeddings_path, device=device, dtype=dtype)

    MAX_EPISODE_STEPS = 400
    total_episodes = args.num_traj
    success_count = 0
    base_seed = 20241201

    import tqdm
    for episode in tqdm.trange(total_episodes):
        obs_window = deque(maxlen=2)
        obs, _ = env.reset(seed=episode + base_seed)
        policy.reset()

        img = env.render().squeeze(0).detach().cpu().numpy()
        obs_window.append(None)
        obs_window.append(np.array(img))
        proprio = obs['agent']['qpos'][:, :-1]

        global_steps = 0
        video_frames = []
        done = False

        while global_steps < MAX_EPISODE_STEPS and not done:
            image_arrs = []
            for window_img in obs_window:
                image_arrs.append(window_img)
                image_arrs.append(None)
                image_arrs.append(None)
            images = [Image.fromarray(arr) if arr is not None else None for arr in image_arrs]

            # Note: RDT policy.step expects text_embeds already computed.
            actions = policy.step(proprio, images, text_embed).squeeze(0).cpu().numpy()

            # Take 8 steps since RDT predicts 64-step chunk; many setups downsample
            actions = actions[::4, :]
            for idx in range(actions.shape[0]):
                action = actions[idx]
                obs, reward, terminated, truncated, info = env.step(action)
                img = env.render().squeeze(0).detach().cpu().numpy()
                obs_window.append(img)
                proprio = obs['agent']['qpos'][:, :-1]
                video_frames.append(img)
                global_steps += 1
                if terminated or truncated:
                    assert "success" in info, sorted(info.keys())
                    if info['success']:
                        success_count += 1
                    done = True
                    break
        print(f"Trial {episode+1} finished, success: {info.get('success', False)}, steps: {global_steps}")
        # After each trial ends:
        if len(video_frames) > 0:
            out_dir = "PickCube_Videos"
            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, f"{env_id}_trial{episode+1}.mp4")

            # video_frames are numpy arrays (H, W, 3)
            h, w, _ = video_frames[0].shape
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # codec
                20.0,  # FPS, adjust if too fast/slow
                (w, h)
            )

            for frame in video_frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR
                writer.write(bgr)
            writer.release()
            print(f"Saved video: {video_path}")

    success_rate = success_count / total_episodes * 100
    print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    main()
