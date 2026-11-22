# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report
from unetModel import UNet
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from timeit import default_timer as timer 

if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
print(device)

# folder on the cluster where the script, metadata.csv and the data_new folder (.npy files) are stored
project_dir = "/data/leuven/394/vsc39405/deep_learning/DL_project"

random_state = 12
np.random.seed(random_state)
torch.manual_seed(random_state)

# %% [markdown]
# Now we will train a model that will predict a 'good' VS a 'bad' nodule.

# %% [markdown]
# ## Initialize model

# %%
n_class = 1
model = UNet(n_class)
print(model)

# %% [markdown]
# ## Make data ready for training

# %% [markdown]
# ### Getting the labels

# %%
df = pd.read_csv(os.path.join(project_dir, "metadata.csv"))

labels = df["label"].values
print(labels)

# %% [markdown]
# ### Splitting the data
# 
# We need to split on patients because otherwise we could have data leakage:
# <br>
# First we take 15% for testing, then we take 15% of the train data for validation
# <br>
# We can see that the dat is not really skewed

# %%
patients = df["patient_id"].unique()

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
print("Aandeel positief:", y_train.mean(), y_val.mean(), y_test.mean())

# %% [markdown]
# Then we load the data:

# %%
directory = os.path.join(project_dir, "data_new")

# the replace turns Windows backslashes in the csv paths into forward slashes so they also work on Linux
X_train = np.array([
    np.load(os.path.join(directory, f.replace("\\", "/")))
    for f in tqdm(train_files)
])

X_val = np.array([
    np.load(os.path.join(directory, f.replace("\\", "/")))
    for f in tqdm(val_files)
])

X_test = np.array([
    np.load(os.path.join(directory, f.replace("\\", "/")))
    for f in tqdm(test_files)
])

# %% [markdown]
# To tensors:
# <br>
# The unsqueeze is because conv3 want: (batch, channels, depth, height, width)

# %%
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val   = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# %% [markdown]
# Making the datasets and dataloaders:

# %%
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# ## Training loop

# %% [markdown]
# ### Posssible options to test further:
# Todo:
# - Scheduler on 
# 
# Tried:
# - **Data Augmentation** (gave improvement)
# - **Dropout higher** (both from 0.2 -> 0.4)
#     

# %%
# this function will mirror each sample around the 3 axis
# so every trainings epoch the model will see slightly different data
# this can be done because nodules are rotationsymmertical enough
# we don't have very much data so could make out model more resistent 
# made with help of ClaudeAI
def augment(x):
    for d in (2, 3, 4): # depth, height, width
        mask = torch.rand(x.size(0), device=x.device) < 0.5 # here you will get for every sample a true or false, true means that that samples will be augmented, false means that that sample will stay the same
        x = torch.where(mask.view(-1, 1, 1, 1, 1), x.flip(d), x) # mask.view makes the shape the same as x, torch.where chooses x.flip(d) if the mask is true else x
    return x


EPOCHS = 10
USE_SCHEDULER = False 

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2) # AdamW improves upon Adam by decoupling weight decay from the gradient update process, enhancing regularisation and model generalisation (https://blog.prodia.com/post/adam-w-vs-adam-key-differences-and-best-use-cases-for-developers)

model.to(device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau( # Used to lower the learning rate when a metric has stopped improving (in this case the validation verb_top1 and noun_top1 accuracy).
    optimizer,
    mode='min',
    factor=0.5,     # Halve the LR (lr = lr * 0.5)
    patience=2     # Wait 2 epochs before any improvement before lowering the LR
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
    

    if USE_SCHEDULER:
        scheduler.step(val_loss)

    for k, v in zip(history, [train_loss, val_loss]): # zip() koppels every key with every value on that position
        history[k].append(v)

    print(f"Epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
          f"val AUPRC {val_auprc:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")
train_time_end = timer()
total_time = train_time_end - train_time_start
print(f"Train time on {str(device)}: {total_time:.3f} seconds")

# save the trained model so it is not lost when the job ends
torch.save(model.state_dict(), os.path.join(project_dir, "unet_model.pt"))

# %%
import matplotlib
matplotlib.use("Agg") # no screen on the cluster, so draw to a file instead of a window
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train loss")
plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(project_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")


# %% [markdown]
# ## Testing

# %%
model.eval()

test_probs = []
with torch.inference_mode():
    for x, y in tqdm(test_dataloader):
        x = x.to(device)
        logits = model(x).squeeze(1)
        test_probs.append(torch.sigmoid(logits).cpu().numpy())

y_prob = np.concatenate(test_probs) # (N,)
y_hat  = (y_prob >= 0.5).astype(int) 
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

# save the test predictions so you can make plots (ROC, PR curve) later without retraining
np.savez(os.path.join(project_dir, "test_predictions.npz"), y_prob=y_prob, y_true=y_true)