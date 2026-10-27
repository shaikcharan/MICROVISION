import os
import joblib
import torch
import torchvision.models as models

# Random Forest
try:
    rf_model = joblib.load("random_forest_model.pkl")
    print("Random Forest loaded ✅")
except Exception as e:
    print("Random Forest error:", e)

# ResNet18
try:
    resnet_model = models.resnet18()
    resnet_model.load_state_dict(torch.load("resnet18_trained.pth", map_location='cpu'))
    resnet_model.eval()
    print("ResNet18 loaded ✅")
except Exception as e:
    print("ResNet18 error:", e)
