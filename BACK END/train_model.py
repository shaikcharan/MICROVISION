import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 1. Define transforms (resize images + convert to tensor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 2. Load dataset (assume DATASET folder has 'train' and 'test' subfolders with class images)
train_dataset = datasets.ImageFolder(root="../DATASET/train", transform=transform)
test_dataset = datasets.ImageFolder(root="../DATASET/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)
        x = self.fc(x)
        return x

num_classes = len(train_dataset.classes)
model = SimpleCNN(num_classes)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(5):  # train for 5 epochs
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5], Loss: {running_loss/len(train_loader):.4f}")

# 6. Save model
torch.save(model.state_dict(), "model.pth")
print("✅ Model training completed and saved as model.pth")
