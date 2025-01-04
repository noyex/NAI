import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

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

image_path = "high_res_image.jpg"

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    high_res_image = Image.open(image_path).convert("L")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

step_size = 50
window_size = 100
draw = ImageDraw.Draw(high_res_image)
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

with torch.no_grad():
    for top in range(0, high_res_image.height - window_size, step_size):
        for left in range(0, high_res_image.width - window_size, step_size):
            cropped_image = high_res_image.crop((left, top, left + window_size, top + window_size))
            image_tensor = transform(cropped_image).unsqueeze(0)
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = probabilities.argmax().item()

            if probabilities[predicted_class] > 0.8:
                label = classes[predicted_class]
                draw.rectangle([left, top, left + window_size, top + window_size], outline="red", width=2)
                draw.text((left, top), label, fill="red")


plt.figure(figsize=(10, 10))
plt.imshow(high_res_image, cmap="gray")
plt.title("Detected Elements")
plt.axis("off")
plt.show()