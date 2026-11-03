
# 🧠 VisionAI — AI Image Recognition

An intelligent image classification web application
powered by Deep Learning and Computer Vision.

Built with Python, Streamlit, PyTorch, and Hugging Face
Transformers.

## 🚀 Live Demo

The deployed application link will be added here.

## 📌 Overview

VisionAI is an AI-powered web application that allows
users to upload images and identify their content
using a pretrained Vision Transformer (ViT).

The application displays the top five predictions,
confidence scores, and interactive visualizations.

## ✨ Features

- Upload JPG, JPEG, and PNG images
- AI-powered image classification
- Top 5 predictions with confidence scores
- Interactive prediction comparison chart
- Modern Streamlit dashboard
- Image resolution information
- Download prediction results as JSON
- Cached model loading for improved efficiency

## 🛠️ Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- Pandas
- Pillow

## 🧠 AI Model

Model: `google/vit-base-patch16-224`

Architecture: Vision Transformer (ViT)

The model was pretrained on ImageNet-21k and
fine-tuned on ImageNet-1k.

It classifies images into 1,000 ImageNet categories.

This project uses transfer learning through a
pretrained model rather than training a new
model from scratch.

## 🏗️ Project Structure

```text
ai-image-recognition/
├── app.py
├── predictor.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Ionut255/ai-image-recognition
cd ai-image-recognition
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Run the application:

```bash
streamlit run app.py
```

## 📊 How It Works

1. The user uploads an image.
2. Pillow reads and converts the image to RGB.
3. The Vision Transformer processes the image.
4. The model generates classification scores.
5. The top five predictions are displayed.
6. The user can export the results as JSON.

## ⚠️ Limitations

- The model recognizes ImageNet categories only.
- It does not detect or localize multiple objects.
- Confidence scores are not guarantees of accuracy.
- First-time model loading requires internet access.
- Performance depends on available computing resources.

## 🔮 Future Improvements

- Support multiple AI models
- Batch image classification
- Image classification history
- Model performance comparison
- Additional Computer Vision features

## 👨‍💻 Project Purpose

This project was developed as part of a
Junior AI Developer portfolio to demonstrate:

- Python development
- Computer Vision fundamentals
- Pretrained model integration
- Deep Learning inference
- Web application development
- Data visualization
- AI application deployment
