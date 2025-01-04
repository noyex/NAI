import torch
from torchvision import transforms
from PIL import Image

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


image_path = "bag.jpg"

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    image = Image.open(image_path).convert("L")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

image_tensor = transform(image).unsqueeze(0)


with torch.no_grad():
    output = model(image_tensor)
    predicted_class = output.argmax(1).item()


classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"Predicted class: {classes[predicted_class]}")