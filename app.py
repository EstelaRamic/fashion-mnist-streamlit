import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps

# --- 1. DEFINISANJE ARHITEKTURE (Mora biti ista kao u Colabu) ---
class JednostavanCNN(nn.Module):
    def __init__(self):
        super(JednostavanCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- 2. UČITAVANJE ISTRENIRANOG MODELA ---
@st.cache_resource # Ovo sprečava da se model učitava ispočetka svaki put kad klikneš dugme
def ucitaj_model():
    model = JednostavanCNN()
    # Učitavamo sačuvane težine (obavezno na CPU jer aplikacija radi lokalno)
    model.load_state_dict(torch.load('model_obuce.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = ucitaj_model()
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- 3. STREAMLIT KORISNIČKI INTERFEJS (UI) ---
st.title("👟 AI Klasifikator Obuće i Odjeće")
st.write("Učitaj sliku iz Fashion-MNIST stila (28x28 crno-bela) i tvoj AI model će pogoditi šta je na njoj!")

# Dugme za učitavanje slike
uploaded_file = st.file_uploader("Izaberi sliku...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Otvaranje slike i pretvaranje u crno-bijelu (Grayscale)
    slika_izvorna = Image.open(uploaded_file).convert('L')
    
    # 2. KLJUČNI KORAK: Invertujemo boje (pretvaramo bijelu pozadinu u crnu, a crnu patiku u bijelu)
    # Ovo radimo jer je model treniran na crnoj pozadini!
    from PIL import ImageOps
    slika_za_model = ImageOps.invert(slika_izvorna)
    
    # Prikazujemo korisniku originalnu sliku koju je učitao
    st.image(slika_izvorna, caption='Učitana slika', width=150)
    
    # --- 4. PRIPREMA SLIKE ZA MODEL ---
    # Smanjujemo sliku na 28x28 i pretvaramo je u matricu/tenzor
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    # Koristimo invertovanu sliku za model
    slika_tensor = transform(slika_za_model).unsqueeze(0) 
    
    # --- 5. AI PREDVIĐANJE ---
    with torch.no_grad():
        izlaz = model(slika_tensor)
        _, predvidjeno = torch.max(izlaz, 1)
        klasa = classes[predvidjeno.item()]
    
    # Prikaz rezultata na ekranu sa lijepim zelenim okvirom
    st.success(f"**AI Predviđanje: {klasa}**")