from itertools import count
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T


class HAL9000(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(12)
        self.fc1 = nn.Linear(4800, 2400)
        self.fc2 = nn.Linear(2400, 10)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.dropout(self.fc1(x.view(-1, 4800))))
        x = F.softmax(self.fc2(x), dim=1)
        return x


DEVICE = torch.device("cpu")
SAVE_FILE = "HAL9000.pt"

train_set = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=T.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
network = HAL9000().to(DEVICE)
optimizer = optim.Adam(network.parameters())
epoch = 0

if os.path.isfile(SAVE_FILE):
    checkpoint = torch.load(SAVE_FILE)
    epoch = checkpoint["epoch"]
    network.load_state_dict(checkpoint["network"])
    optimizer.load_state_dict(checkpoint["optimizer"])

start_time = time.thread_time()

for i in count(epoch):
    total_correct = 0
    total_loss = 0

    for images, labels in train_loader:
        # Get the predictions
        predictions = network(images.to(DEVICE))
        # Calculate the loss
        loss = F.cross_entropy(predictions, labels.to(DEVICE))

        # Reset the gradients
        optimizer.zero_grad()
        # Calculate the gradients
        loss.backward()
        # Update the network
        optimizer.step()

        num_correct = predictions.argmax(dim=1).eq(labels.to(DEVICE)).sum().item()
        total_correct += num_correct
        total_loss += loss.item()

    print(
        f"Epoch: {i + 1}",
        "Average accuracy: "
        + str(round(total_correct / len(train_set) * 100, 2))
        + "%",
        f"Average loss: {str(round(total_loss / len(train_set), 4))}",
        f"Time: {round(time.thread_time() - start_time, 2)}",
        sep=", "
    )
    checkpoint = {
        "epoch": i + 1,
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, SAVE_FILE)
    start_time = time.thread_time()