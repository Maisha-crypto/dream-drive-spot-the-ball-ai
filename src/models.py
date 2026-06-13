import torch
import torch.nn as nn
import torchvision.models as models


class BallCoordinatePredictor(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super(BallCoordinatePredictor, self).__init__()

        # Load a standard pre-trained CNN backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
            num_features = self.backbone.fc.in_features
            # Remove original classification head
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights="DEFAULT" if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
    
        # Custom Regression Head: Map imagefeatures to 2 continuous numbers (x, y)
        self.regression_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
            nn.Sigmoid()
                )
    def forward(self, x):
        features = self.backbone(x)
        coordinates = self.regression_head(features)
        return coordinates

# #Validation check
# if __name__ == "__main__":
#     model = BallCoordinatePredictor(backbone="resnet18")
#     dummy_batch = torch.randn(4,3,224,224)
#     output = model(dummy_batch)
#     print("Output Shape:", output.shape)
#     print("Example Prediction:", output[0])   