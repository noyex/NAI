import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = BaseModel()
model.load_state_dict(torch.load("base_model.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_loader:
        output = model(X)
        predictions = output.argmax(1)
        y_true.extend(y.numpy())
        y_pred.extend(predictions.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
train_counts = np.bincount(train_data.targets.numpy())
test_counts = np.bincount(test_data.targets.numpy())

plt.figure(figsize=(12, 6))
x = np.arange(len(classes))
width = 0.4
plt.bar(x - width / 2, train_counts, width, label="Train")
plt.bar(x + width / 2, test_counts, width, label="Test")
plt.xticks(x, classes, rotation=45)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Train and Test Sets")
plt.legend()
plt.tight_layout()
plt.show()

#zadanie 2 w konsoli, macierz bledu + analiza wyswietla sie