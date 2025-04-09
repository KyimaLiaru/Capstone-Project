from glob import glob
import numpy as np

paths = glob("../../../dataset/Preprocessed/Lakh/MultiTrack/*.npy")
total = {"drum": 0, "bass": 0, "pad": 0, "lead": 0}

for path in paths[:500]:  # Sample 500 files
    data = np.load(path, allow_pickle=True).item()
    for key in total:
        total[key] += np.sum(data[key] > 0)

print(total)