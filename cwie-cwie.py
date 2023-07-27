import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          transform=transforms.Compose([
                                              transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1307,), std=(0.3081,))
                                          ]),
                                          download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform=transforms.Compose([
                                             transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.1325,), std=(0.3105,))
                                         ]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        shape_before_conv1 = x.size()
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        shape_before_conv2 = x.size()
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        shape_before_conv3 = x.size()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1), shape_before_conv1, shape_before_conv2, shape_before_conv3

# Create the model and move it to the device
model = ConvNeuralNet().to(device)

# Loss function and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, _, _, _ = model(images)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 400 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Evaluate the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs, _, _, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# Function to calculate L1-norm of each filter
def calculate_l1_norm(weights):
    return torch.sum(torch.abs(weights), dim=(1, 2, 3))

# Get the L1-norm for each layer's filters
conv1_l1_norms = calculate_l1_norm(model.conv1.weight)
conv2_l1_norms = calculate_l1_norm(model.conv2.weight)
conv3_l1_norms = calculate_l1_norm(model.conv3.weight)

# Define the number of filters to prune in each layer
k = 10

# List of convolutional layers and their corresponding L1-norms
conv_layers = [model.conv1, model.conv2, model.conv3]
conv_l1_norms = [conv1_l1_norms, conv2_l1_norms, conv3_l1_norms]

# Prune and print pruned filters from each layer in a loop
for layer, l1_norms in zip(conv_layers, conv_l1_norms):
    # Get the indices of the filters to prune (the ones with the least L1-norm)
    pruned_filters_indices = torch.argsort(l1_norms)[:k]

    # Print the indices of filters to prune
    print("Indices of filters to prune in this layer:", pruned_filters_indices)

    # Prune filters from the current layer
    layer.weight.data[pruned_filters_indices, :, :, :].zero_()
    layer.bias.data[pruned_filters_indices].zero_()

    # Print the filters that were pruned from the current layer
    print("Pruned filters in this layer:")
    for idx in pruned_filters_indices:
        print("Pruning filter at index:", idx.item())
        print(layer.weight.data[idx])

# Print the feature map shape before and after pruning in a loop
def print_feature_map_shapes(model, images):
    _, *shapes_before_conv = model(images)
    print("Shapes before pruning:")
    for i, shape in enumerate(shapes_before_conv, 1):
        print(f"Shape before Conv{i}:", shape)

    # Forward pass through the pruned model
    model(images)

    # Get the shape of weight data for each convolutional layer after pruning
    shapes_after_conv = [layer.weight.data.size() for layer in conv_layers]
    print("\nShapes after pruning:")
    for i, shape in enumerate(shapes_after_conv, 1):
        print(f"Shape after Conv{i}:", shape)

# Test feature map shapes before and after pruning
sample_images, _ = next(iter(test_loader))
sample_images = sample_images.to(device)
print_feature_map_shapes(model, sample_images)
