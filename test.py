import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class HAL9000(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 10)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(self.maxpool1(x))
        x = F.relu(self.conv2(x))
        x = self.bn2(self.maxpool2(x))
        x = F.relu(self.dropout(self.fc1(x.view(-1, 300))))
        x = F.softmax(self.fc2(x), dim=1)
        return x


DEVICE = torch.device("cuda")

test_set = torchvision.datasets.MNIST(
    root="./data", download=True, transform=T.ToTensor(),
)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

network = HAL9000().to(DEVICE)
network.load_state_dict(torch.load("HAL9000.pt")["network"])
network.eval()
num_correct = 0
total_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        predictions = network(images.to(DEVICE))
        loss = F.cross_entropy(predictions, labels.to(DEVICE))
        num_correct += predictions.argmax(dim=1).eq(labels.to(DEVICE)).sum().item()
        total_loss += loss.item()

print(
    "Accuracy:",
    str(round(num_correct / len(test_set) * 100, 2)) + "%,",
    "Average loss:",
    round(total_loss / len(test_set), 4),
)
