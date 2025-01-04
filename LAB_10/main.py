import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class ConvModel1(nn.Module):
    def __init__(self):
        super(ConvModel1, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14 * 14 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class ConvModel2(nn.Module):
    def __init__(self):
        super(ConvModel2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7 * 7 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.linear_relu_stack(x)

def train_model(model, train_loader, test_loader, epochs, optimizer_fn):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizer_fn(model.parameters())

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                correct += (pred.argmax(1) == y).sum().item()
                total += y.size(0)
            print(f"Epoch {epoch + 1}, Accuracy: {100 * correct / total:.2f}%")
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    models = {
        "base_model.pth": BaseModel(),
        "conv_model_1.pth": ConvModel1(),
        "conv_model_2.pth": ConvModel2()
    }

    for model_name, model in models.items():
        print(f"Training {model_name} with SGD")
        train_model(model, train_loader, test_loader, epochs=5, optimizer_fn=lambda params: optim.SGD(params, lr=0.01))
        torch.save(model.state_dict(), model_name)

    for model_name, model in models.items():
        print(f"Training {model_name} with Adam")
        model.load_state_dict(torch.load(model_name))
        train_model(model, train_loader, test_loader, epochs=5, optimizer_fn=lambda params: optim.Adam(params, lr=0.001))
        torch.save(model.state_dict(), f"adam_{model_name}")

if __name__ == "__main__":
    main()