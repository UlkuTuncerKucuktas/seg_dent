import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss
from tqdm import tqdm
from model import UNetSmall
from dataset import create_train_val_datasets
from loss import BoundaryLoss, dice_coefficient
import optuna

# Parameters
DEVICE = 'cuda'
num_epochs = 25
evaluation_interval = 5
num_channels = 1 

# Function to visualize mask and prediction
def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(10, 5))
    image = image.squeeze(0).permute(1, 2, 0)
    if mask.dim() > 2:
        mask = mask.squeeze(0).squeeze(0)
    if prediction.dim() > 2:
        prediction = prediction.squeeze(0).squeeze(0)
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().numpy())
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.title('True Mask')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction.cpu().numpy(), cmap='gray')
    plt.title('Predicted Mask')
    plt.show()

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size =  trial.suggest_int('batch_size', 4 , 32 , 4)
    min_channel =  trial.suggest_int('min_channel', 4 , 32 , 8)
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_train_val_datasets('/content/merged_annotations.json', '/content/dataset/dataset', preprocessing_fn=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Move model to device
    model = UNetSmall(num_channels=num_channels, num_classes=1,min_channel=min_channel).to(DEVICE)

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dice_loss_fn = DiceLoss(mode='binary')
    boundary_loss_fn = BoundaryLoss()

    best_val_dice = 0
    best_model_path = 'best_model.pth'

    # Training and Evaluation Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for org, images, masks in tqdm(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            dice_loss = dice_loss_fn(outputs, masks)
            boundary_loss = boundary_loss_fn(outputs, masks)
            loss = dice_loss + boundary_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}')

        # Evaluate every 'evaluation_interval' epochs
        if (epoch + 1) % evaluation_interval == 0:
            model.eval()
            running_loss = 0.0
            dice_scores = []
            with torch.no_grad():
                for org, images, masks in tqdm(val_loader):
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    outputs = model(images)
                    dice_loss = dice_loss_fn(outputs, masks)
                    boundary_loss = boundary_loss_fn(outputs, masks)
                    loss = dice_loss + boundary_loss
                    running_loss += loss.item()
                    dice_score = dice_coefficient(outputs, masks)
                    dice_scores.append(dice_score.item())
                visualize_prediction(images.cpu(), masks.cpu(), outputs.cpu())
                val_loss = running_loss / len(val_loader)
                avg_dice_score = sum(dice_scores) / len(dice_scores)
                print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Average Dice Score: {avg_dice_score}')
                if avg_dice_score > best_val_dice:
                    best_val_dice = avg_dice_score
                    torch.save(model.state_dict(), best_model_path)
                    print(f'Model saved as validation dice improved to {best_val_dice}')

    return best_val_dice

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print(f'Best hyperparameters: {study.best_params}')
