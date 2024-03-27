from model import Net
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics import Accuracy, Precision

#initalise the model
net = Net()

#prepare the train and test datasets
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAutocontrast(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#image paths
image_path_train = "../data/training_set"
image_path_test = "../data/test_set"

train_dataset = ImageFolder(image_path_train, transform=train_transform)
test_dataset = ImageFolder(image_path_test, transform=test_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)


def train_and_test():
    #training
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in train_dataloader:
            outputs = net(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        print(f"Epoch: {epoch}, loss: {epoch_loss}")

    torch.save(net.state_dict(), "./model/model_state_dict.pth")

    #testing
    metric_accuracy = Accuracy(task='multiclass', num_classes=3, average='macro')
    metric_precision = Precision(task='multiclass', num_classes=3, average='macro') 

    net.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = net(images)
            _, preds = torch.max(outputs, 1)
            metric_accuracy(preds, labels)
            metric_precision(preds, labels)

    acc = metric_accuracy.compute()
    prec = metric_precision.compute()
    print(f'Accuracy: {acc}\nPrecision: {prec}')

if __name__ == "__main__":
    train_and_test()