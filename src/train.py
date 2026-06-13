from calendar import EPOCH
from genericpath import exists
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DreamDriveDataset
from models import BallCoordinatePredictor

def train_model():
    # Hyperparamerers
    EPOCHS =20
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO]: Training starting.")
    print(f"[INFO]: Using device {DEVICE}.")

    # Data Initialisation
    csv_path = os.path.join("data", "dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[INFO]: {csv_path} not found. Run 'label_tool.py' first!")

    dataset = DreamDriveDataset(csv_file=csv_path, image_size=224)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Mean Squared Error (MSE) ,Optimiser
    model = BallCoordinatePredictor(backbone="resnet50", pretrained=True).to(DEVICE)   
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Core Training Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE)

            # Forward pass the images through the neural network
            predictions = model(images)

            # Compute distance/ error between predictions and actual coordinates
            loss = criterion(predictions, targets)

            optimizer.zero_grad()   # Reset old gradients
            loss.backward()         # Compute new gradients
            optimizer.step()        # Update network weights

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[INFO]:{epoch+1}/{EPOCHS} -> Loss: {epoch_loss:.6f}")
    # Save model artifacts
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "spot_the_ball_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print("[INFO]: Training Complete!")
    print(f"[INFO]: Model weights saved to {model_save_path}")


# Main
if __name__ == "__main__":
    train_model()

