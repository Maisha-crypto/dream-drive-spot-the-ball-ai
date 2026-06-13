import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models import BallCoordinatePredictor

def predict_ball_center(image_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the raw image using CV (for drawing later)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[INFO]:Could not load from{image_path}")
        return  
    h_original, w_original, _ = original_image.shape

    # Pre-process the image for the model
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # Load Model and Weights
    model = BallCoordinatePredictor(backbone="resnet18", pretrained=False).to(DEVICE)
    model_weights = os.path.join("models", "spot_the_ball_model.pth")

    if not os.path.exists(model_weights):
        print(f"[INFO]: Saved weight not found at {model_weights}. Run 'train.py' for first!")
        return

    model.eval()

    # Mode Prediction (Inference)
    with torch.no_grad():
        prediction = model(image_tensor)
        norm_x, norm_y = prediction[0][0].item(), prediction[0][1].item()

    # Reverse Normalisation
    pred_x = int(norm_x * w_original)
    pred_y = int(norm_y * h_original)

    print(f"[RESUTLS]: Predicted coordinates relative to scale: ({norm_x:.4f},{norm_y:.4f})")
    print(f"[RESULTS]: Mapped to original image pixels: ({pred_x:.4f},{pred_y:4f})")

    # Draw  a bright green crosshaire and circle where the predicted ball is
    cv2.circle(original_image, (pred_x, pred_y), 15, (0, 255, 0), 2)
    cv2.circle(original_image, (pred_x, pred_y), 2, (0, 255, 0), -1)
    cv2.line(original_image, (pred_x - 25, pred_y), (pred_x + 25, pred_y), (0, 255, 0), 2)
    cv2.line(original_image, (pred_x, pred_y - 25), (pred_x, pred_y + 25), (0, 255, 0, 2))

    # Save the displayed results
    output_path = "Prediction_output.jpg"
    cv2.imwrite(output_path, original_image)
    print(f"[INFO]: Visualised predidtion saved to {output_path}!")

    cv2.imshow("AI Spot The Ball Prediction", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image = os.path.join("data",  "raw", "IMG_2617.jpeg")
    predict_ball_center(test_image)




