import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss

from tqdm import tqdm
from model import UNet
from dataset import create_train_val_datasets
from loss import BoundaryLoss , dice_coefficient

# Parameters
DEVICE = 'cuda'
num_epochs = 300
batch_size = 1
learning_rate = 0.001
evaluation_interval = 5


train_dataset, val_dataset = create_train_val_datasets('/content/merged_annotations.json', '/content/dataset/dataset', preprocessing_fn=None)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Move model to device
model = UNet(num_channels=1, num_classes=1).to(DEVICE)

# Optimizer and Loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dice_loss_fn = DiceLoss(mode='binary')
boundary_loss_fn = BoundaryLoss()

# Function to visualize mask and prediction
def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(10, 5))

    # Remove the batch dimension and permute to [H, W, C]
    image = image.squeeze(0).permute(1, 2, 0)

    # Adjust mask and prediction shape if they are not 2D
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


best_val_dice = 0
best_model_path = 'best_model.pth'
# Training and Evaluation Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for org , images, masks in tqdm(train_loader):
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
        dice_scores = []  # List to store dice scores for each batch

        with torch.no_grad():
            for org , images, masks in tqdm(val_loader):
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


print("Training complete")
