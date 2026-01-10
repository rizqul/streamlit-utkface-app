import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import streamlit as st

class EfficientNetV2MultiOutput(nn.Module):
    """
    This class MUST be identical to the one used in your training script.
    """
    def __init__(self, num_age_classes, num_race_classes):
        super(EfficientNetV2MultiOutput, self).__init__()
        # Load a pre-trained EfficientNetV2 model from torchvision
        self.base_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Get the number of input features from the pre-trained model's classifier
        num_features = self.base_model.classifier[1].in_features

        # Remove the original classifier and replace it with custom heads
        self.base_model.classifier = nn.Identity()
        self.age_head = nn.Linear(num_features, num_age_classes)
        self.gender_head = nn.Linear(num_features, 1)
        self.race_head = nn.Linear(num_features, num_race_classes)

    def forward(self, x):
        features = self.base_model(x)
        age_output = self.age_head(features)
        gender_output = self.gender_head(features).squeeze(1)
        race_output = self.race_head(features)
        return age_output, gender_output, race_output


@st.cache_resource
def load_model(backbone_path):
    NUM_AGE_CLASS = 5
    NUM_RACE_CLASS = 5
    model = EfficientNetV2MultiOutput(NUM_AGE_CLASS, NUM_RACE_CLASS)
    model.load_state_dict(torch.load(backbone_path, map_location=torch.device('cpu')))

    model.eval()

    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

