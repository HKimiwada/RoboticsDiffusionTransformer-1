# Code to convert data from Huggingface to the format required for finetuning (.h5 + .json files)
import h5py
import json

h5_file = h5py.File("Demos_PickSingleYCB-v0/002_master_chef_can.h5", "r")
print("Structure of the h5 file:")
print(list(h5_file.keys()))

with open("Demos_PickSingleYCB-v0/002_master_chef_can.json", "r") as json_file:
    data = json.load(json_file)
print("Keys in the JSON file:")
print(data.keys())