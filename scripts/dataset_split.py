import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Split datasets')
parser.add_argument("-i", '--infile', required=True, type=str, help='path of dataset')
parser.add_argument("-o", '--outpath', required=True, type=str, help='path of dataset')
args = parser.parse_args()

seed = 1018
infile = args.infile
outpath = args.outpath
os.makedirs(outpath, exist_ok=True)

data = np.load(infile)
direction = data["direction"]
time = data["time"]
size = data["size"]
label = data["label"]

X = np.stack((direction, time, size), axis=-1)
y = label

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=seed)

print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test: X = {X_test.shape}, y = {y_test.shape}")

np.savez(os.path.join(outpath, "train.npz"), X=X_train, y=y_train)
np.savez(os.path.join(outpath, "valid.npz"), X=X_valid, y=y_valid)
np.savez(os.path.join(outpath, "test.npz"), X=X_test, y=y_test)
print(f"split {infile} done")
