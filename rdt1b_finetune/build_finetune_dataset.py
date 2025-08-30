# Code to convert data from Huggingface to the format required for finetuning (.h5 + .json files)
import h5py, json, os, textwrap

# H5_PATH = "Demos_PickSingleYCB-v0/002_master_chef_can.h5"
# JSON_PATH = "Demos_PickSingleYCB-v0/002_master_chef_can.json"

H5_PATH = "Demos_PickSingleYCB-v0/065-j_cups.h5"
JSON_PATH = "Demos_PickSingleYCB-v0/065-j_cups.json"

def walk(group, indent=0, max_items=4):
    for k in group.keys():
        obj = group[k]
        if isinstance(obj, h5py.Dataset):
            print("  " * indent + f"- {k}: dataset shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print("  " * indent + f"[{k}]/")
            walk(obj, indent+1, max_items=max_items)

with h5py.File(H5_PATH, "r") as f:
    # Top-level is a lot of trajectories
    print("Top-level groups:", list(f.keys())[:10], "...")
    # Peek the first trajectory
    g = f["traj_0"]
    print("\ntraj_0 contents:")
    walk(g)

with open(JSON_PATH, "r") as jf:
    meta = json.load(jf)
    print("\nJSON top-level keys:", meta.keys())
    print("JSON.episodes length:", len(meta.get("episodes", [])))
    # If present, these are very useful:
    env_info = meta.get("env_info", {})
    print("env_info keys:", env_info.keys())
    print("env_info sample:", {k: env_info[k] for k in list(env_info)[:6]})
