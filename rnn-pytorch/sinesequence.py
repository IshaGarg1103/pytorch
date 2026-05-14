import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


def show_plot(filename):
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.show(block=True)
    plt.close()


# -----------------------------
# 1. Hyperparameters
# -----------------------------

batch_size = 16
T = 1000
num_train = 600
tau = 4
lr = 0.01
epochs = 5

torch.manual_seed(42)

# -----------------------------
# 2. Create synthetic sequence data
# -----------------------------

time = torch.arange(1, T + 1, dtype=torch.float32)

# x = sine wave + noise
x = torch.sin(0.01 * time) + torch.randn(T) * 0.2

plt.figure(figsize=(6, 3))
plt.plot(time.numpy(), x.numpy())
plt.xlabel("time")
plt.ylabel("x")
plt.title("Noisy Sine Wave")
plt.grid(True)
show_plot("noisy_sine_wave.png")

# -----------------------------
# 3. Convert sequence into features and labels
# -----------------------------
# Goal:
# [x1, x2, x3, x4] -> x5
# [x2, x3, x4, x5] -> x6
# [x3, x4, x5, x6] -> x7

features = []

for i in range(tau):
    features.append(x[i : T - tau + i])

features = torch.stack(features, dim=1)

labels = x[tau:].reshape(-1, 1)

print("features shape:", features.shape)  # (996, 4)
print("labels shape:", labels.shape)      # (996, 1)

# -----------------------------
# 4. Train-validation split
# -----------------------------

train_features = features[:num_train]
train_labels = labels[:num_train]

val_features = features[num_train:]
val_labels = labels[num_train:]

train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 5. Define model
# -----------------------------
# Since tau = 4, input size is 4.
# Output size is 1 because we predict one next value.

model = nn.Linear(tau, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# -----------------------------
# 6. Train model
# -----------------------------

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_loader:
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * X_batch.shape[0]

    avg_train_loss = total_train_loss / len(train_dataset)

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            total_val_loss += loss.item() * X_batch.shape[0]

    avg_val_loss = total_val_loss / len(val_dataset)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )

# Plot training and validation loss

plt.figure(figsize=(6, 3))
plt.plot(range(1, epochs + 1), train_losses, label="train_loss")
plt.plot(range(1, epochs + 1), val_losses, label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
show_plot("training_loss.png")

# -----------------------------
# 7. One-step-ahead prediction
# -----------------------------
# Here model always uses real previous values.

model.eval()

with torch.no_grad():
    onestep_preds = model(features).reshape(-1)

plt.figure(figsize=(6, 3))
plt.plot(time[tau:].numpy(), labels.reshape(-1).numpy(), label="labels")
plt.plot(time[tau:].numpy(), onestep_preds.numpy(), label="1-step preds")
plt.xlabel("time")
plt.ylabel("x")
plt.title("One-step-ahead Prediction")
plt.legend()
plt.grid(True)
show_plot("one_step_prediction.png")

# -----------------------------
# 8. Multi-step prediction
# -----------------------------
# Use real values until time 604.
# Then predict future values using previous predictions.

multistep_preds = torch.zeros(T)
multistep_preds[:] = x

with torch.no_grad():
    for i in range(num_train + tau, T):
        input_window = multistep_preds[i - tau : i].reshape(1, -1)
        multistep_preds[i] = model(input_window).reshape(())

plt.figure(figsize=(6, 3))
plt.plot(
    time[tau:].numpy(),
    onestep_preds.numpy(),
    label="1-step preds"
)
plt.plot(
    time[num_train + tau:].numpy(),
    multistep_preds[num_train + tau:].numpy(),
    label="multistep preds"
)
plt.xlabel("time")
plt.ylabel("x")
plt.title("Multi-step Prediction")
plt.legend()
plt.grid(True)
show_plot("multi_step_prediction.png")


def k_step_pred(k):
    feature_columns = []
    for i in range(tau):
        feature_columns.append(x[i : i + T - tau - k + 1])

    with torch.no_grad():
        for i in range(k):
            input_features = torch.stack(
                feature_columns[i : i + tau],
                dim=1
            )

            preds = model(input_features).reshape(-1)
            feature_columns.append(preds)
            
    return feature_columns[tau:]

steps = (1, 4, 16, 64)

preds = k_step_pred(steps[-1])

plt.figure(figsize=(6, 3))

for k in steps:
    plt.plot(
        time[tau + steps[-1] - 1:].numpy(),
        preds[k - 1].numpy(),
        label=f"{k}-step preds"
    )

plt.xlabel("time")
plt.ylabel("x")
plt.title("k-step-ahead Predictions")
plt.legend()
plt.grid(True)
show_plot("k_step_predictions.png")