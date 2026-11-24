import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from timeit import default_timer as timer
from mednet import ResNet10
from pathlib import Path
import matplotlib.pyplot as plt
import copy

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
print(device)

random_state = 12
np.random.seed(random_state)
torch.manual_seed(random_state)
ROOT_PATH = Path("/scratch/leuven/393/vsc39388/DL")
os.chdir(ROOT_PATH)

model = ResNet10()
print(model)

# Pretrained weights from MedicalNet (Google Drive: MedicalNet_pytorch_files2/pretrain/)
# resnet_10_23dataset.pth = ResNet-10 pretrained on 23 medical datasets
PRETRAINED_PATH = ROOT_PATH / "pretrain_mednet" / "resnet_10_23dataset.pth"

# weights_only=True: only tensors are loaded, no code from the pickle file is executed
checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=True)
pretrained_dict = checkpoint["state_dict"]

# MedicalNet was trained with nn.DataParallel, so every key starts with "module." (e.g. "module.conv1.weight")
pretrained_dict = {k.replace("module.", "", 1): v for k, v in pretrained_dict.items()}

# Only keep the weights that exist in our model with the same shape
# (our classification head fc is new, so it keeps its random initialization)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} tensors from {PRETRAINED_PATH}")
print("Not pretrained (random init):", [k for k in model_dict if k not in pretrained_dict])

df = pd.read_csv("metadata.csv")

labels = df["label"].values
print(labels)

df = pd.read_csv("metadata.csv")

# remove nodules with mean malignancy exactly 3 ("indeterminate"): these are ambiguous labels
# no reset_index: df.index must keep pointing to the right row in X_all.npy
df = df[~df["mean_malignancy"].between(2.5, 3.5)]

labels = df["label"].values
print(len(df), "nodules left, share 'bad':", labels.mean())

patients = df["patient_id"].unique().to_numpy()

temp_patients, test_patients = train_test_split(patients, test_size=0.15, random_state=random_state)
train_patients, val_patients = train_test_split(temp_patients, test_size=0.15, random_state=random_state)


# get the df with the right patient id
train_df = df[df["patient_id"].isin(train_patients)]
val_df = df[df["patient_id"].isin(val_patients)]
test_df = df[df["patient_id"].isin(test_patients)]

# one patient can have multiple nodules, so multiple samples
train_files, y_train = train_df["patch_path"].values, train_df["label"].values
val_files, y_val = val_df["patch_path"].values, val_df["label"].values
test_files, y_test = test_df["patch_path"].values, test_df["label"].values

print("Patiënts:", len(train_patients), len(val_patients), len(test_patients))
print("Samples:", len(train_files), len(val_files), len(test_files))
print("Amount positief:", y_train.mean(), y_val.mean(), y_test.mean())

X_train = np.array([np.load(f) for f in tqdm(train_files)])
X_val   = np.array([np.load(f) for f in tqdm(val_files)])
X_test  = np.array([np.load(f) for f in tqdm(test_files)])

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)


# one global mean and std, computed only on the train set (no data leakage)
train_mean = X_train.mean()
train_std = X_train.std()

def zscore(x):
    # every patch is normalized with the same train statistics, so differences in density between patches are kept
    return (x - train_mean) / (train_std + 1e-6) # + 1e-6 to never divide by 0

X_train = zscore(X_train)
X_val   = zscore(X_val)
X_test  = zscore(X_test)
print(f"Train mean/std used: {train_mean.item():.4f} / {train_std.item():.4f}")
print("Train after:", X_train.mean().item(), X_train.std().item())
print("Val after:  ", X_val.mean().item(), X_val.std().item())

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Training loop

# this function will mirror each sample around the 3 axis
# so every trainings epoch the model will see slightly different data
# this can be done because nodules are rotationsymmertical enough
# we don't have very much data so could make out model more resistent
# made with help of ClaudeAI
def augment(x): # random flips + random 90° turns
    for d in (2, 3, 4): # depth, height, width
        mask = torch.rand(x.size(0), device=x.device) < 0.5 # here you will get for every sample a true or false, true means that that samples will be augmented, false means that that sample will stay the same
        x = torch.where(mask.view(-1, 1, 1, 1, 1), x.flip(d), x) # mask.view makes the shape the same as x, torch.where chooses x.flip(d) if the mask is true else x
    k = int(torch.randint(0, 4, (1,))) # random number, this number is the amount of time we turn 90°
    return torch.rot90(x, k, dims=(3, 4)) # turn 90° k times, same rotation per batch


EPOCHS = 100
USE_SCHEDULER = True

loss_fn = nn.BCEWithLogitsLoss()
# with weighing the classes:
# pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()], device=device)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Pretrained backbone gets a 10x lower learning rate than the new classification head (fc),
# so the pretrained features are fine-tuned instead of overwritten
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
head_params = list(model.fc.parameters())
backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=1e-2) # AdamW improves upon Adam by decoupling weight decay from the gradient update process, enhancing regularisation and model generalisation (https://blog.prodia.com/post/adam-w-vs-adam-key-differences-and-best-use-cases-for-developers)

model.to(device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau( # Used to lower the learning rate when a metric has stopped improving
    optimizer,
    mode='max',
    factor=0.5,     # Halve the LR (lr = lr * 0.5)
    patience=3     # Wait 3 epochs before any improvement before lowering the LR
)

history = {"train_loss": [], "val_loss": []} # after every epoch data is set in it here

train_time_start = timer() # Timer starten
for epoch in tqdm(range(EPOCHS)):
    # Training:
    model.train()
    train_loss = 0.0
    for x,y in train_dataloader:
        x = x.to(device)
        x = augment(x)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits.squeeze(1), y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # To prevent exploding gradients: if the overall gradient norm exceeds 1, all gradients are scaled down proportionally
        optimizer.step()

        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_dataset)

    # Validation:
    model.eval()
    val_loss = 0.0
    val_probs = []
    with torch.inference_mode():
        for x,y in val_dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits.squeeze(1), y)

            val_loss += loss.item() * x.size(0)
            val_probs.append(torch.sigmoid(logits).cpu().numpy()) # to CPU because Sklearn works on RAM, to numpy because Sklearn works with numpy arrays

    val_loss /= len(val_dataset)
    val_auprc = average_precision_score(y_val.numpy(), np.concatenate(val_probs).ravel()) # concatenate groups all the batchs array to one array, .ravel() goes from form (x,1) to (x,) beacause Sklearn expect 1D scores here for binarry classification

    if epoch == 0 or val_auprc > best_auprc:
        best_auprc = val_auprc
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        best_val_probs = np.concatenate(val_probs).ravel()


    if USE_SCHEDULER:
        scheduler.step(val_auprc)

    for k, v in zip(history, [train_loss, val_loss]): # zip() koppels every key with every value on that position
        history[k].append(v)

    print(f"Epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
          f"val AUPRC {val_auprc:.4f} | lr backbone {optimizer.param_groups[0]['lr']:.6f} / head {optimizer.param_groups[1]['lr']:.6f}")
train_time_end = timer()
total_time = train_time_end - train_time_start
print(f"Train time on {str(device)}: {total_time:.3f} seconds")



plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train loss")
plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig("mednet_loss_curves.png", dpi=300)

### Testing

from sklearn.metrics import precision_recall_curve
model.load_state_dict(best_state)
print(f"Best epoch {best_epoch}, val AUPRC {best_auprc:.4f}")

prec, rec, thr = precision_recall_curve(y_val.numpy(), best_val_probs)
f1 = 2 * prec * rec / (prec + rec + 1e-8)
threshold = thr[f1[:-1].argmax()]
print("Threshold chosen on val:", threshold)

model.eval()

test_probs = []
with torch.inference_mode():
    for x, y in tqdm(test_dataloader):
        x = x.to(device)
        logits = model(x).squeeze(1)
        test_probs.append(torch.sigmoid(logits).cpu().numpy())

y_prob = np.concatenate(test_probs) # (N,)
y_hat  = (y_prob >= threshold).astype(int)
y_true = y_test.numpy().astype(int)

print("Predicted 'Bad':", y_hat.sum())
print("Real 'Bad':", y_true.sum())
print("Shape:", y_true.shape, y_hat.shape)

print("AUPRC: ", average_precision_score(y_true, y_prob))
baseline = y_true.mean()
print(f"Random Baseline ('Bad' %): {baseline:.4f}")
print("Accuracy:", accuracy_score(y_true, y_hat))
print(confusion_matrix(y_true, y_hat))
print(classification_report(y_true, y_hat, target_names=["good", "bad"]))

### Bootstrap resampling
from bootstrap import bootstrap_auprc

patient_ids = test_df["patient_id"].values.astype(str) # same order as y_true / y_prob
os.makedirs("results", exist_ok=True)
np.savez("results/mednet.npz", y_true=y_true, y_prob=y_prob, patient_ids=patient_ids)

lo, hi, std = bootstrap_auprc(y_true, y_prob, patient_ids)
print(f"Test AUPRC: {average_precision_score(y_true, y_prob):.4f} ± {std:.4f}  95% CI [{lo:.4f} - {hi:.4f}]")


