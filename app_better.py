import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# --- 1. DEFINISANJE NAPREDNE CNN ARHITEKTURE (Mora biti ista kao u Colabu) ---
class NapredanCNN(nn.Module):
    def __init__(self):
        super(NapredanCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. UCITAVANJE MODELA ---
@st.cache_resource
def ucitaj_napredni_model():
    model = NapredanCNN()
    # Učitavamo nove težine koje si skinula iz Colaba
    model.load_state_dict(torch.load('model_obuce_napredni.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = ucitaj_napredni_model()

# Nazivi klasa na srpskom/bosanskom radi boljeg korisničkog iskustva!
classes = ['Majica', 'Pantalone', 'Džemper', 'Haljina', 'Kaput', 'Sandale', 'Košulja', 'Patike', 'Torba', 'Čizme']

# --- 3. STREAMLIT KORISNIČKI INTERFEJS ---
st.title("🚀 Napredni AI Klasifikator Odeće i Obuće")
st.write("Ova aplikacija koristi duboku konvolucionu mrežu (CNN) sa regularizacijom za stabilnije prepoznavanje.")

uploaded_file = st.file_uploader("Izaberi sliku komada odeće ili obuće...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    slika_izvorna = Image.open(uploaded_file).convert('L')
    
    # Invertujemo pozadinu za model jer Fashion-MNIST zahtijeva crnu pozadinu
    slika_za_model = ImageOps.invert(slika_izvorna)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(slika_izvorna, caption='Originalna slika', width=150)
    with col2:
        st.image(slika_za_model, caption='Kako je AI analizira', width=150)
    
    # Priprema za model
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    slika_tensor = transform(slika_za_model).unsqueeze(0)
    
    # Predviđanje
    with torch.no_grad():
        izlaz = model(slika_tensor)
        _, predvidjeno = torch.max(izlaz, 1)
        klasa = classes[predvidjeno.item()]
    
    st.info(f"🤖 **Rezultat napredne analize: {klasa}**")