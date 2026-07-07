# 👟 Fashion-MNIST Image Classifier with PyTorch & Streamlit

A production-ready End-to-End Computer Vision application that classifies clothing and footwear items into 10 distinct categories. The project features a Convolutional Neural Network (CNN) trained using PyTorch and deployed as an interactive web application via Streamlit.

## 🚀 Live Demo Features
- **Instant Inference:** Upload any image of footwear or clothing, and the AI will classify it in real-time.
- **Grayscale Conversion & Auto-Resizing:** Built-in preprocessing pipeline that handles standard RGB images and formats them to the required 28x28 grayscale dimension.
- **Optimized Performance:** Cached model loading for sub-second response times.

---

## 🧠 Model Architecture & Training
The core of the project is a Convolutional Neural Network (CNN) optimized for feature extraction from localized image patterns (edges, shapes, texture).

- **Framework:** PyTorch
- **Dataset:** Fashion-MNIST (60,000 training images, 10,000 test images)
- **Architecture:** - Convolutional Layer (16 filters, 3x3 kernel, ReLU activation)
  - Max Pooling Layer (2x2 kernel, stride 2)
  - Fully Connected Layers (Dense layers mapping to 10 classes)
- **Optimization:** Adam Optimizer with Cross-Entropy Loss
- **Results:** Achieved **92.23% accuracy** on the validation set in just 5 epochs.

---

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Deployment & UI:** Streamlit
- **Data & Image Processing:** NumPy, Pillow (PIL), Matplotlib
- **Environment:** Anaconda, VS Code

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/EstelaRamic/fashion-mnist-streamlit.git](https://github.com/EstelaRamic/fashion-mnist-streamlit.git)
cd fashion-mnist-streamlit

Install the required packages:
Bash
pip install -r requirements.txt

Run the Streamlit app:
Bash
streamlit run app.py

🤝 Contact & Freelance Inquiries
Looking for a specialized Data Scientist / Machine Learning Engineer to build custom computer vision pipelines or deploy AI models into production? Let's connect!

Available for Freelance Projects on Upwork
