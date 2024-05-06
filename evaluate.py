from train import *
from data import *

_, test_loader = process_dataset(dataset_location=dataset_location)

model.eval()

# Initialize variables to monitor test loss and accuracy
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
        # Calculate loss
        test_loss += loss.item()
        
        # Calculate accuracy
        predicted = outputs.round()
        correct += (predicted.squeeze() == labels).sum().item()
        total += labels.size(0)

# Calculate average loss and accuracy
average_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

# Print evaluation results
print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%')