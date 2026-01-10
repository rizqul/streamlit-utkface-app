import streamlit as st
import torch
from PIL import Image, ImageOps
from utils import load_model, get_transform

st.title('Facial Classification App')

st.write("""
### Upload facial image to get the classification of age, gender, and race
""")

model = load_model('./utkface_efficientnetv2_s_categorical_age.pth')

uploaded_file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.exif_transpose(image)
    
    col1, col2 = st.columns(2)

    age_classes = ['0-4', '5-12', '13-19', '20-29', '30-39', '40-59', '60-79', '80+'] 
    gender_classes = ['Male', 'Female'] # Assuming 0 is Male, 1 is Female after sigmoid
    race_classes = ['White', 'Black', 'Asian', 'Indian', 'Others']

    with col1:
        st.image(image, caption='Uploaded image', use_container_width=True)

    with col2:
        st.write('### Model Predictions')

        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            age_logits, gender_logit, race_logits = model(image_tensor)

            age_probs = torch.softmax(age_logits, dim=1)
            age_confidence, age_pred_idx = torch.max(age_probs, 1)
            predicted_age = age_classes[age_pred_idx.item()]
            
            gender_prob = torch.sigmoid(gender_logit)
            gender_confidence = gender_prob.item() if gender_prob.item() > 0.5 else 1 - gender_prob.item()
            predicted_gender = gender_classes[1] if gender_prob.item() > 0.5 else gender_classes[0]

            race_probs = torch.softmax(race_logits, dim=1)
            race_confidence, race_pred_idx = torch.max(race_probs, 1)
            predicted_race = race_classes[race_pred_idx.item()]

        st.metric(label='Predicted Age Bracket', value=f"{predicted_age}", help=f"Confidence: {age_confidence.item()*100:.2f}%")
        st.metric(label='Predicted Gender', value=f"{predicted_gender}", help=f"Confidence: {gender_confidence*100:.2f}%")
        st.metric(label='Predicted Race', value=f"{predicted_race}", help=f"Confidence: {race_confidence.item()*100:.2f}%")
            