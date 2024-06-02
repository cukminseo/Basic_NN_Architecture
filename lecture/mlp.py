import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


"""
https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
if train true, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
transform is used to define preprocessing and scaling operations to apply to input data
transform=transforms.ToTensor()? Converts a PIL Image or NumPy ndarray to a PyTorch Tensor
"""
train_dataset = datasets.MNIST(root='MNIST_data/', train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# load test dataset
test_dataset = datasets.MNIST(root='MNIST_data/', train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# EDA work
#check data
print(len(train_dataset))

# slice train-validation set(0.85:0.15 ratio)
train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

# data split use random skill
# if dataset is unbalance, we can use stratified split
train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

# Check data
print(len(train_dataset), len(validation_dataset), len(test_dataset))

# hyperparameter setting - batch size
BATCH_SIZE = 32

# get train data(size : BATCH_SIZE)
train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# get validation data(size : BATCH_SIZE)
validation_dataset_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
# get test data(size : BATCH_SIZE)
test_dataset_loader = DataLoader(dataset=test_dataset , batch_size=BATCH_SIZE, shuffle=True)

# Checking a single piece of data
train_features, train_labels = next(iter(train_dataset_loader))
# As the batch size is 32, 32 pieces of data make up one tensor.
print (f"Feature batch shape: {train_features.size()}")
# Each of the 32 pieces of data has a label, making up a tensor.
print ( f"Label is batch shape : {train_labels.size()} ")
"""
Here, train_features[0] refers to the first image in the batch fetched from the data loader.
This image is a tensor with a shape of (1, 28, 28), where the first dimension represents a grayscale image.
However, we don't actually need this dimension, so we remove it with the squeeze() function.
The result is a tensor with a shape of (28, 28), which represents a grayscale image of 28x28 pixels.
"""
img = train_features[0].squeeze()
# get label
label = train_labels[0]
# Since it's a colorless bitmap, we retrieve it in gray.
plt.imshow(img, cmap= "gray")
# show plt
plt.show()
# Print the label of the above data
print (f"Label: {label}")

# Model definition
class MyDeepLearningModel(nn.Module):
    # init method
    def __init__(self):
        super().__init__()
        # Function to flatten the input before feeding it into the MLP
        self.flatten = nn.Flatten()
        # MLP with 784 inputs and 256 outputs
        self.fc1 = nn.Linear(784, 256)
        # Using ReLU as the activation function
        self.relu = nn.ReLU()
        # Using dropout to prevent overfitting (with a ratio of 0.3)
        self.dropout = nn.Dropout(p=0.3)
        # Function with 256 inputs and 10 outputs
        # Since there are 10 classes for the final classification, the output is set to 10
        self.fc2 = nn.Linear(256, 10)

    # Using the functions declared above
    def forward(self, x):
        # flatten layer
        x = self.flatten(x)
        # fully connected layer1
        x = self.fc1(x)
        # activation function
        x = self.relu(x)
        # set dropout
        x = self.dropout(x)
        # fully connected layer2
        logits = self.fc2(x)
        return logits

# Model declaration
model = MyDeepLearningModel()

# Using CrossEntropyLoss. It also includes softmax function
loss_function = nn.CrossEntropyLoss()
# Using SGD as the optimizer, and setting the learning rate as a hyperparameter
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# Define the training function
def model_train(dataloader, model, loss_function, optimizer):
    # Call the train method of the nn.Module superclass
    model.train()
    # Variable to accumulate the total loss
    train_loss_sum = 0
    # Variable to accumulate the count of correct predictions
    train_correct = 0
    # Variable to accumulate the total count
    train_total = 0
    # Save the total number of batches
    total_train_batch = len(dataloader)

    # Iterate over each batch
    for images, labels in dataloader:
        # Reshape images from (batch_size, 1, 28, 28) to (batch_size, 784)
        x_train = images.view(-1, 28 * 28)
        # Assign labels to y_train
        y_train = labels

        # Input the data into the model and save the output
        outputs = model(x_train)
        # Compute the loss between actual and predicted values
        loss = loss_function(outputs, y_train)

        # Zero the gradients
        optimizer.zero_grad()
        # Compute the gradients
        loss.backward()
        # Update the parameters
        optimizer.step()

        # Accumulate the loss
        train_loss_sum += loss.item()
        # Accumulate the total count
        train_total += y_train.size(0)
        # Accumulate the count of correct predictions
        train_correct += (torch.argmax(outputs, 1) == y_train).sum().item()

    # Compute the average loss
    train_avg_loss = train_loss_sum / total_train_batch
    # Compute the average accuracy
    train_avg_accuracy = 100 * train_correct / train_total

    # Return the average loss and accuracy
    return train_avg_loss, train_avg_accuracy

# Define the evaluation function
def model_evaluate(dataloader, model, loss_function):
    # Set the model to evaluation mode (disable dropout, batch normalization, etc.)
    model.eval()
    # Do not compute gradients during evaluation
    with torch.no_grad():
        # Variable to accumulate the total loss
        val_loss_sum = 0
        # Variable to accumulate the count of correct predictions
        val_correct = 0
        # Variable to accumulate the total count
        val_total = 0
        # Save the total number of batches
        total_val_batch = len(dataloader)

        # Iterate over each batch
        for images, labels in dataloader:
            # Reshape images from (batch_size, 1, 28, 28) to (batch_size, 784)
            x_val = images.view(-1, 28 * 28)
            # Assign labels to y_val
            y_val = labels
            # Input the data into the model and save the output
            outputs = model(x_val)
            # Compute the loss between actual and predicted values
            loss = loss_function(outputs, y_val)
            # Accumulate the loss
            val_loss_sum += loss.item()
            # Accumulate the total count
            val_total += y_val.size(0)
            # Accumulate the count of correct predictions
            val_correct += (torch.argmax(outputs, 1) == y_val).sum().item()

        # Compute the average loss
        val_avg_loss = val_loss_sum / total_val_batch
        # Compute the average accuracy
        val_avg_accuracy = 100 * val_correct / val_total

        # Return the average loss and accuracy
        return val_avg_loss, val_avg_accuracy

# Define the test function
def model_test(dataloader, model, loss_function):
    # Set the model to evaluation mode (disable dropout, batch normalization, etc.)
    model.eval()
    # Do not compute gradients during evaluation
    with torch.no_grad():
        # Variable to accumulate the total loss
        test_loss_sum = 0
        # Variable to accumulate the count of correct predictions
        test_correct = 0
        # Variable to accumulate the total count
        test_total = 0
        # Save the total number of batches
        total_test_batch = len(dataloader)

        # Iterate over each batch
        for images, labels in dataloader:
            # Reshape images from (batch_size, 1, 28, 28) to (batch_size, 784)
            x_test = images.view(-1, 28 * 28)
            # Assign labels to y_test
            y_test = labels
            # Input the data into the model and save the output
            outputs = model(x_test)
            # Compute the loss between actual and predicted values
            loss = loss_function(outputs, y_test)
            # Accumulate the loss
            test_loss_sum += loss.item()
            # Accumulate the total count
            test_total += y_test.size(0)
            # Accumulate the count of correct predictions
            test_correct += (torch.argmax(outputs, 1) == y_test).sum().item()

        # Compute the average loss
        test_avg_loss = test_loss_sum / total_test_batch
        # Compute the average accuracy
        test_avg_accuracy = 100 * test_correct / test_total

        # Print the accuracy
        print('Accuracy:', test_avg_accuracy)
        # Print the loss
        print('Loss:', test_avg_loss)

# Lists to save training loss and accuracy
train_loss_list = []
train_accuracy_list = []
# Lists to save validation loss and accuracy
val_loss_list = []
val_accuracy_list = []

# Record the start time of training
start_time = datetime.now()

# Total number of epochs
EPOCHS = 20

# Iterate over each epoch
for epoch in range(EPOCHS):
    # ------------ model train --------------
    # Train the model and return average loss and accuracy
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    # Append the training loss to the list
    train_loss_list.append(train_avg_loss)
    # Append the training accuracy to the list
    train_accuracy_list.append(train_avg_accuracy)

    # =========== model evaluation --------------
    # Validate the model and return average loss and accuracy
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function)
    # Append the validation loss to the list
    val_loss_list.append(val_avg_loss)
    # Append the validation accuracy to the list
    val_accuracy_list.append(val_avg_accuracy)

    # Print the results for each epoch
    print('epoch:', '%02d' % (epoch + 1),
          'train loss=', '{:.4f}'.format(train_avg_loss), 'train accuracy=', '{:.4f}'.format(train_avg_accuracy),
          'validation loss=', '{:.4f}'.format(val_avg_loss), 'validation accuracy=', '{:.4f}'.format(val_avg_accuracy))

# Record the end time of training
end_time = datetime.now()

# Print the elapsed time
print('elapsed time=', end_time - start_time)
