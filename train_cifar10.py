import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from QPCnet import QHybridClassifier  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)


model = QHybridClassifier(num_classes=10, input_size=(32, 32), batch_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train_losses, test_accuracies = [], []
best_metrics = {'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
best_cm = None

with open("epoch_prf_log_c10.txt", "w") as f:
    f.write("Epoch\tPrecision\tRecall\tF1-score\tAccuracy\n")


import time

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    start_epoch_time = time.time()
    total_batches = len(train_loader)

    for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        elapsed = time.time() - start_epoch_time
        batches_done = batch_idx
        batches_left = total_batches - batches_done
        time_per_batch = elapsed / batches_done
        eta_seconds = time_per_batch * batches_left

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Batch [{batch_idx}/{total_batches}], "
            f"Loss: {loss.item():.4f}, "
            f"ETA: {int(eta_seconds // 60)}m {int(eta_seconds % 60)}s",
            end='\r', 
        )
    print() 

    train_losses.append(running_loss / total_batches)

    model.eval()
    all_preds, all_labels = [], []
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / len(test_dataset)
    test_accuracies.append(acc)

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    macro = report['macro avg']
    p, r, f1 = macro['precision'], macro['recall'], macro['f1-score']

    print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={acc:.2f}%, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

    with open("epoch_prf_log_c10.txt", "a") as f:
        f.write(f"{epoch+1}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{acc:.2f}\n")
        
    if acc > best_metrics['acc']:
        best_metrics.update({'acc': acc, 'precision': p, 'recall': r, 'f1': f1})
        best_cm = confusion_matrix(all_labels, all_preds)


print("\nBest Accuracy Model:")
for k, v in best_metrics.items():
    print(f"{k.upper()}: {v:.4f}")


# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, marker='o')
# plt.title("Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.subplot(1, 2, 2)
# plt.plot(test_accuracies, marker='s')
# plt.title("Test Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy (%)")

# plt.tight_layout()
# plt.savefig("training_results_c10.png")


# plt.figure(figsize=(8, 6))
# sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Best Confusion Matrix")
# plt.savefig("confusion_matrix_c10.png")

# np.savetxt("train_losses_c10.txt", np.array(train_losses), fmt="%.6f")
# np.savetxt("test_accuracies_c10.txt", np.array(test_accuracies), fmt="%.2f")
