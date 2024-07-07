import os
import torch
import argparse
import numpy as np
import ARES

def load_array(features, labels, batch_size, is_train=True, num_workers=8):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def process_data(data, max_len, is_direction=False):
    X = data["X"]
    y = data["y"]

    if max_len < X.shape[-1]:
        X = X[..., 0:max_len]
    
    if max_len > X.shape[-1]:
        last_dim_padding = max_len - X.shape[1]
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, last_dim_padding)]
        X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
    
    if is_direction:
        X[X > 0] = 1
        X[X < 0] = -1
     
    X = torch.tensor(X[:, np.newaxis], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

parser = argparse.ArgumentParser(description='Test of ARES')
parser.add_argument("-g", '--gpu', default=0, type=int, help='Device id')
parser.add_argument("-d", '--dataset', required=True, type=str, help='Dataset name')
parser.add_argument("-m", '--model', required=True, type=str, help='Model file path')

batch_size = 64
feat_length = 10000

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
in_path = args.dataset

test_path = os.path.join(in_path, "test.npz")
test_data = np.load(test_path)
test_X, test_y = process_data(test_data, feat_length, is_direction=True)

num_classes = test_y.shape[1]
print(f"test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

test_iter = load_array(test_X, test_y, batch_size, is_train=False)

model = ARES.Trans_WF(num_classes)
model_file = args.model
print("loading model:", model_file)
model.load_state_dict(torch.load(model_file, map_location="cpu"))
model = model.cuda()

y_pred_score = np.zeros((0, num_classes))
with torch.no_grad():
    model.eval()
    for index, cur_data in enumerate(test_iter):
        cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
        outs = model(cur_X)
        y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
y_true = test_y.numpy()

# Calculate accuracy
correct = 0
total = y_true.shape[0]

for idx in range(y_pred_score.shape[0]):
    cur_pred = y_pred_score[idx]
    top_pred = np.argmax(cur_pred)  # Get the index of the highest score
    if y_true[idx, top_pred] > 0:   # Check if the top prediction is correct
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.3f}")
