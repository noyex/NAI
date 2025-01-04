import torch
from torch import nn
from torchvision import transforms
from PIL import Image

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

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_path = "sandal.jpg"
image = Image.open(image_path).convert("L")
image_tensor = transform(image).unsqueeze(0)

models = {
    "Base Model": ("base_model.pth", BaseModel()),
    "Conv Model 1": ("conv_model_1.pth", ConvModel1()),
    "Conv Model 2": ("conv_model_2.pth", ConvModel2())
}

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

for model_name, (model_path, model) in models.items():
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = output.argmax(1).item()
        print(f"{model_name}: Predicted class is {classes[predicted_class]}")