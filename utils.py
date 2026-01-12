import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import streamlit as st
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Load Pretrained EfficientNetV2-S
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
            
        # The classifier of EfficientNetV2 is a Sequential block. 
        # We need the input features of the last linear layer.
        # usually self.backbone.classifier[1] is the Linear layer.
        in_features = self.backbone.classifier[1].in_features
        
        # Remove the existing classifier to get raw features
        self.backbone.classifier = nn.Identity()
        
        # --- Define Heads ---
        
        # 1. Age Head (Regression: 1 output)
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) 
        )
        
        # 2. Gender Head (Classification: 2 classes)
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # 3. Race Head (Classification: 5 classes)
        self.race_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )
        
        # 4. Age Category Head (Classification: 5 classes)
        self.age_cat_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        # Get shared features from backbone
        features = self.backbone(x)
        
        # Pass features to each head
        age_pred = self.age_head(features)
        gender_pred = self.gender_head(features)
        race_pred = self.race_head(features)
        age_cat_pred = self.age_cat_head(features)
        
        return {
            'age': age_pred,
            'gender': gender_pred,
            'race': race_pred,
            'age_category': age_cat_pred
        }
        



@st.cache_resource
def load_model(backbone_path):
    model = MultiTaskEfficientNet()
    model.load_state_dict(torch.load(backbone_path, map_location=torch.device('cpu')))

    model.eval()

    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

