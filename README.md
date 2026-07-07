# 🚀 Fashion-MNIST Classification: From Baseline to Advanced CNN with Real-World Evaluation

This repository demonstrates the development, optimization, and evaluation of Deep Learning models for apparel and footwear classification using **PyTorch** and **Streamlit**. The project showcases a clear progression from a simple baseline model to an advanced Convolutional Neural Network (CNN) architecture with regularization, evaluated against real-world images.

*Note: The core variable names, code comments, and UI labels within the Python scripts are written in Bosnian/Serbian to demonstrate localized application development, while the architectural documentation below is fully provided in English.*

---

## 📂 Project Structure

* `app.py` - The baseline Streamlit web application utilizing a simple 1-layer CNN model.
* `app_better.py` - The advanced Streamlit web application featuring a deeper CNN with optimization.
* `model_obuce_napredni.pth` - Saved PyTorch model weights after 15 epochs of training in Google Colab.
* `requirements.txt` - Configuration file listing all necessary Python dependencies.

---

## 🧠 Base Model Architecture & Training
The core of the project is a Convolutional Neural Network (CNN) optimized for feature extraction from localized image patterns (edges, shapes, texture).

- **Framework:** PyTorch
- **Dataset:** Fashion-MNIST (60,000 training images, 10,000 test images)
- **Architecture:** - Convolutional Layer (16 filters, 3x3 kernel, ReLU activation)
  - Max Pooling Layer (2x2 kernel, stride 2)
  - Fully Connected Layers (Dense layers mapping to 10 classes)
- **Optimization:** Adam Optimizer with Cross-Entropy Loss
- **Results:** Achieved **92.23% accuracy** on the validation set in just 5 epochs.

---

## 🧠 Advanced Model Architecture (`NapredanCNN`)

To improve robustness and mitigate overfitting, the advanced model introduces several standard deep learning optimization techniques:
1.  **Deeper Network Structure:** Two sequential convolutional layers (`Conv2d`) scaling from 32 to 64 filters to extract more abstract spatial features.
2.  **Batch Normalization:** `BatchNorm2d` layers added after each convolution to stabilize training and accelerate convergence.
3.  **Dropout Regularization:** A 25% Dropout rate applied to both convolutional and fully connected layers to prevent the network from memorizing training data.
4.  **Extended Training:** Trained for 15 epochs using the Adam optimizer, achieving a final validation accuracy of **92.51%** on the Fashion-MNIST test split.

---

## 🔍 Engineering Analysis & Evaluation (Lessons Learned)

While the advanced network performed exceptionally well on standard test data, deploying it to a live Streamlit interface and testing it with arbitrary web images revealed critical, real-world Machine Learning challenges:

### 1. Data Distribution Mismatch (Dataset Bias)
* **The Issue:** Fashion-MNIST consists exclusively of low-resolution (28x28), centered grayscale photographs of actual clothing items. When the model is fed *vector illustrations, clip art, or sketches* with sharp, high-contrast outlines, it struggles to generalize (e.g., misclassifying a clear sketch of a sneaker as a "bag" or "sandal").
* **The Lesson:** A model is only as robust as the distribution of its training data. To effectively recognize stylized artwork, the training set must be augmented with sketches, or a Transfer Learning approach using massive, pre-trained vision models (like ResNet or CLIP) should be deployed.

### 2. Transparent Background Artefacts (PNG Grayscale Preprocessing)
* **The Issue:** Uploading transparent PNGs disrupts standard grayscale inversion. The alpha channel transparency, when compressed and inverted, generates random noise patterns and textured grids around the subject. This alters the input tensor drastically, confusing the model's spatial awareness.
* **The Lesson:** Data preprocessing pipelines must be rigidly standardized to clean, mask, or pad inbound user web uploads before converting them into tensors, ensuring the data format strictly matches what the model expects.

---

## 🛠️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/EstelaRamic/fashion-mnist-streamlit.git](https://github.com/EstelaRamic/fashion-mnist-streamlit.git)
   cd fashion-mnist-streamlit

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision
- **Deployment & UI:** Streamlit
- **Data & Image Processing:** NumPy, Pillow (PIL), Matplotlib
- **Environment:** Anaconda, VS Code

---
## ⚙️ Installation & Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/EstelaRamic/fashion-mnist-streamlit.git
cd fashion-mnist-streamlit
````
Install the required packages:
```bash
pip install -r requirements.txt
````
Run the Streamlit app:
 ```bash
streamlit run app_better.py
````
🤝 Contact & Freelance Inquiries
Looking for a specialized Data Scientist / Machine Learning Engineer to build custom computer vision pipelines or deploy AI models into production? Let's connect!

Available for Freelance Projects on Upwork

GitHub: EstelaRamic
