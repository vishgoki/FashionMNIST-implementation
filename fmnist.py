#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:17:38 2024

@author: vishwa
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
    
# Loading and preprocessing the FashionMNIST dataset
trans = transforms.ToTensor()

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=trans)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=trans)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

print("Loading done")

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        trans = self.linear_relu_stack(x)
        return trans

model = NeuralNetwork()

# Training
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 40 
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        fit = loss(outputs, labels)
        fit.backward()
        optimizer.step()
        train_loss += fit.item()
    train_losses.append(train_loss / len(trainloader))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            fit = loss(outputs, labels)
            test_loss += fit.item()
    test_losses.append(test_loss / len(testloader))

    print('Epoch %s, Train loss %s, Test loss %s'%(epoch, train_loss, test_loss))
    
# Evaluating
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        pred_outputs = model(images)
        _, predicted = torch.max(pred_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total

# Visualizing
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Test Loss')
plt.legend()
plt.show()


for idx in range(3):
    # Apply the softmax function to the model's predictions to get probabilities
    softmax_probs = torch.nn.functional.softmax(outputs[idx].cpu(), dim=0).numpy()
    
    # Plotting the probabilities for each class
    plt.scatter(range(len(softmax_probs)), softmax_probs)
    plt.title(f'Probability Distribution for Image {idx}')
    plt.xlabel('Class Index')
    plt.ylabel('Probability')
    plt.show()
    plt.clf()
    image_to_show = images[idx].cpu().squeeze().detach().numpy()
    plt.imshow(image_to_show, cmap='gray')
    plt.show()
