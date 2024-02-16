import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)  
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier:
    def __init__(self, learning_rate=0.001, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mnist_train = MNIST(root=os.getcwd(), train=True, download=True, transform=ToTensor())
        mnist_test = MNIST(root=os.getcwd(), train=False, download=True, transform=ToTensor())
        emnist_train = EMNIST(root=os.getcwd(), split='digits', train=True, download=True, transform=ToTensor())
        emnist_test = EMNIST(root=os.getcwd(), split='digits', train=False, download=True, transform=ToTensor())

        self.train_loader = DataLoader(ConcatDataset([mnist_train, emnist_train]), batch_size=64, shuffle=True)
        self.test_loader = DataLoader(ConcatDataset([mnist_test, emnist_test]), batch_size=64, shuffle=True)

        self.model = CNN(num_classes=10).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
       
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
               
        self.best_val_loss = np.inf
        self.best_model_path = "best_model.pth"

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            val_loss = self.validate()
            
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(self.train_loader)}, Train Accuracy: {train_accuracy}%')
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)


    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        validation_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {validation_accuracy}%, Validation Loss: {total_loss / len(self.test_loader)}')
        return total_loss / len(self.test_loader)


model_trainer = Classifier(learning_rate=0.001, epochs=25)
model_trainer.train()



# import onnxruntime as ort
# import numpy as np

# input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
# sess = ort.InferenceSession("model.onnx")
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name
# output_data = sess.run([output_name], {input_name: input_data})
# print(output_data)
