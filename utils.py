import torch
import torch.nn as nn
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import streamlit as st
from retinaface import RetinaFace
import numpy as np


MARGIN = 0.2
TARGET_SIZE = (224, 224)
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

def crop_align(image_pil):
    """
    Fungsi untuk melakukan alignment dan cropping wajah menggunakan RetinaFace,
    sesuai dengan logika preprocessing script Anda.
    """
    # 1. Convert PIL (RGB) ke NumPy Array (BGR untuk OpenCV)
    image_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 2. Detect Faces
    # RetinaFace bisa memakan waktu sedikit lama pada CPU
    resp = RetinaFace.detect_faces(img_bgr)

    # Jika tidak ada wajah terdeteksi atau return kosong, kembalikan gambar asli
    if not resp or isinstance(resp, tuple):
        return image_pil

    # Ambil wajah pertama (face_1) sesuai logika script Anda
    face_data = resp['face_1']
    landmarks = face_data['landmarks']
    bbox = face_data['facial_area']

    # 3. Get Eye Coordinates & Face Center
    left_eye_img = landmarks["left_eye"]
    right_eye_img = landmarks["right_eye"]

    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1
    face_cx = x1 + (face_w // 2)
    face_cy = y1 + (face_h // 2)

    # 4. Calculate Rotation Angle
    dy = left_eye_img[1] - right_eye_img[1]
    dx = left_eye_img[0] - right_eye_img[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # 5. Calculate Scale
    target_w_box = face_w * (1 + MARGIN * 2)
    target_h_box = face_h * (1 + MARGIN)
    
    desired_size = max(target_w_box, target_h_box)
    scale = TARGET_SIZE[0] / desired_size

    # 6. Build Affine Matrix
    center = (float(face_cx), float(face_cy))
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Centering translation
    t_x = (TARGET_SIZE[0] / 2) - face_cx
    t_y = (TARGET_SIZE[1] / 2) - face_cy

    # Shift Correction (logic from your script)
    shift_y = (face_h * MARGIN * 0.5) * scale

    M[0, 2] += t_x
    M[1, 2] += t_y + shift_y

    # 7. Apply Warp
    aligned_face = cv2.warpAffine(
        img_bgr, M, TARGET_SIZE, 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,0,0)
    )

    # Convert kembali ke PIL (RGB)
    aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(aligned_rgb)