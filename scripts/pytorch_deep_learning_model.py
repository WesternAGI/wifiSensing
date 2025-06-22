# PyTorch Deep Learning Model for WiFi Sensing
# Add these code blocks to your visualizations.ipynb notebook

# =============================================================================
# CELL 1: Additional Imports for PyTorch (add to your imports section)
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

# =============================================================================
# CELL 2: PyTorch Neural Network Model Definition
# =============================================================================

class WiFiSensingNet(nn.Module):
    """
    Deep Neural Network for WiFi Sensing Classification
    Input: CSI data with shape (batch_size, 10800)
    Output: Binary classification (empty vs working)
    """
    def __init__(self, input_size=10800, hidden_sizes=[2048, 1024, 512, 256], num_classes=2, dropout_rate=0.3):
        super(WiFiSensingNet, self).__init__()
        
        # Create layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# =============================================================================
# CELL 3: Data Preparation for PyTorch
# =============================================================================

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training data shape: {X_train_tensor.shape}")
print(f"Test data shape: {X_test_tensor.shape}")
print(f"Number of classes: {len(torch.unique(y_train_tensor))}")

# =============================================================================
# CELL 4: Model Training
# =============================================================================

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = WiFiSensingNet(input_size=10800, hidden_sizes=[2048, 1024, 512, 256], num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training parameters
num_epochs = 100
best_accuracy = 0.0
patience = 15
patience_counter = 0

# Training loop
train_losses = []
train_accuracies = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
    
    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total_val += target.size(0)
            correct_val += (predicted == target).sum().item()
    
    # Calculate accuracies
    train_accuracy = 100 * correct_train / total_train
    val_accuracy = 100 * correct_val / total_val
    avg_loss = running_loss / len(train_loader)
    
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
    
    # Early stopping
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_wifi_sensing_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    scheduler.step()

print(f'Training completed. Best validation accuracy: {best_accuracy:.2f}%')

# =============================================================================
# CELL 5: Model Evaluation
# =============================================================================

# Load best model
model.load_state_dict(torch.load('best_wifi_sensing_model.pth'))
model.eval()

# Get predictions for test set
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

# Print classification report
print("\nDeep Learning Model Performance:")
print("="*50)
print(classification_report(all_targets, all_predictions, target_names=['empty', 'working']))
print("\nConfusion Matrix:")
print(confusion_matrix(all_targets, all_predictions))

# Calculate final accuracy
final_accuracy = 100 * sum([1 for i, j in zip(all_targets, all_predictions) if i == j]) / len(all_targets)
print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")

# =============================================================================
# CELL 6: Training Visualization
# =============================================================================

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training')
plt.plot(val_accuracies, label='Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Plot confusion matrix
cm = confusion_matrix(all_targets, all_predictions)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['empty', 'working'])
plt.yticks(tick_marks, ['empty', 'working'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# =============================================================================
# CELL 7: Save the PyTorch Model (add this to your final save section)
# =============================================================================

# Save the complete model (architecture + weights)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': model,
    'input_size': 10800,
    'num_classes': 2,
    'class_names': ['empty', 'working']
}, 'wifi_sensing_deep_model.pth')

# Also save just the state dict for easier loading
torch.save(model.state_dict(), 'wifi_sensing_model_weights.pth')

print("PyTorch models saved successfully!")
print("Files saved:")
print("- wifi_sensing_deep_model.pth (complete model)")
print("- wifi_sensing_model_weights.pth (weights only)")
print("- best_wifi_sensing_model.pth (best model during training)")
