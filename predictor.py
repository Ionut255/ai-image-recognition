
import streamlit as st
from transformers import pipeline


MODEL_NAME = "google/vit-base-patch16-224"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the image classification model."""
    return pipeline(
        task="image-classification",
        model=MODEL_NAME
    )


def predict_image(image):
    """Return the top 5 predictions for an image."""
    classifier = load_model()
    predictions = classifier(image, top_k=5)

    return predictions
