from data import data_utils as data
from torchvision.models.mobilenet import mobilenet_v2
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_confusion_matrix(name, truth_labels, prediction_labels):
    # Confusion Matrix From:
    # https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
    
    # constant for classes
    classes = ('0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9')

    # Build confusion matrix
    cf_matrix = confusion_matrix(truth_labels, prediction_labels)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
        
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(name +'.png')

def main():
    
    # Get Transformed Data
    svhn_train, svhn_test = data.get_svhn()
    _, mnist_test = data.get_mnist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32

    svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size = batch_size, shuffle = True, num_workers = 4)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size = batch_size, shuffle = False, num_workers = 4)
    mnist_loader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = 4)

    # Setup Model
    model = mobilenet_v2(weights = None)
    # Out Features of 10 to represent the 10 possible labels
    model.classifier[1] = torch.nn.Linear(in_features = model.last_channel, out_features = 10)
    model = model.to(device)

    # Setup Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr = .01)
    loss_func = nn.CrossEntropyLoss()

    num_epochs = 10

    # Train the Model
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(svhn_train_loader):
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate Accuracy
            predictions = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            accuracy = np.sum(np.array(labels) == np.array(predictions)) / batch_size

            # Compute loss
            loss = loss_func(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print statistics
            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i}/{len(svhn_train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy}')

    prediction_labels = []
    truth_labels = []

    # Test on SVHN
    for inputs, labels in svhn_test_loader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        prediction_labels.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        truth_labels.extend(labels) # Save Truth
    
    accuracy = np.sum(np.array(prediction_labels) == np.array(truth_labels)) / len(svhn_test)
    print(f'SVHN Test Accuracy: {accuracy}')
    
    generate_confusion_matrix("svhn_confusion", truth_labels, prediction_labels)

    prediction_labels = []
    truth_labels = []
    
    # Test on MNIST
    for inputs, labels in mnist_loader:
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        prediction_labels.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        truth_labels.extend(labels) # Save Truth
    
    accuracy = np.sum(np.array(prediction_labels) == np.array(truth_labels)) / mnist_test.data.size(dim=0)
    print(f'MNIST Test Accuracy: {accuracy}')

    generate_confusion_matrix("mnist_confusion", truth_labels, prediction_labels)
        

if __name__ == '__main__':
    main()