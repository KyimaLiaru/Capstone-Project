import os
import numpy as np

directory = "../../../dataset/Preprocessed/Lakh/MultiTrack"
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        path = os.path.join(directory, filename)
        try:
            np.load(path, allow_pickle=True).item()
        except Exception as e:
            print(f"Corrupted file: {filename} -- {e}")