import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import SimpleCNN
import pandas as pd
import altair as alt

# Charger le modèle
model = SimpleCNN()
model.load_state_dict(torch.load("simple_cnn.pth", map_location='cpu'))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["Benign", "Malignant"]

# Page Streamlit
st.set_page_config(page_title="Classification Peau", layout="centered")
st.title("🩺 Classification de peau - Benign vs Malignant")
st.markdown("Téléversez une image et le modèle prédit si elle est bénigne ou maligne.")

# Upload d'image
uploaded_file = st.file_uploader("📂 Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Prétraitement
    img_tensor = transform(image).unsqueeze(0)

    # Prédiction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        _, pred = torch.max(outputs, 1)

    st.markdown(f"### 🔹 Prediction : **{classes[pred.item()]}**")

    # Affichage des probabilités sous forme de graphique
    prob_df = pd.DataFrame({
        "Classe": classes,
        "Probabilité": probs.numpy()
    })

    chart = alt.Chart(prob_df).mark_bar().encode(
        x=alt.X("Classe", sort=None),
        y="Probabilité",
        color=alt.Color("Classe", scale=alt.Scale(scheme="set2")),
        tooltip=["Classe", alt.Tooltip("Probabilité", format=".2f")]
    ).properties(width=400, height=300, title="Probabilités par classe")

    st.altair_chart(chart)

    st.success("✅ Prédiction réalisée avec succès !")
