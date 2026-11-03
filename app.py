
import hashlib
import io
import json

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predictor import MODEL_NAME, predict_image


# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="VisionAI | Image Recognition",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------
# CUSTOM STYLING
# -----------------------------

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(
            120deg, #172554, #4338ca
        );
        padding: 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 17px;
        margin-bottom: 0;
    }
    </style>

    <div class="hero">
        <h1>🧠 VisionAI</h1>
        <p>
            Intelligent Image Recognition
            Powered by Deep Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:
    st.title("🧠 VisionAI")
    st.caption("AI Image Classification Platform")

    st.divider()

    st.subheader("About the Model")
    st.write("**Architecture:** Vision Transformer")
    st.write("**Model:** Google ViT Base")
    st.write("**Framework:** Hugging Face")
    st.write("**Task:** Image Classification")

    st.divider()

    st.subheader("How It Works")
    st.write("1. Upload an image.")
    st.write("2. Click Analyze Image.")
    st.write("3. Explore AI predictions.")
    st.write("4. Download your results.")

    st.divider()

    st.caption("Built with Python, PyTorch & Streamlit")


# -----------------------------
# MAIN INTERFACE
# -----------------------------

st.subheader("Image Analysis Dashboard")

st.write(
    "Upload a photo and discover what "
    "artificial intelligence recognizes."
)

left_col, right_col = st.columns(
    [1, 1],
    gap="large"
)

with left_col:
    st.markdown("### 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    image = None
    image_hash = None

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()

            image_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            image = Image.open(
                io.BytesIO(file_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            st.caption(
                f"Resolution: {image.width} x "
                f"{image.height} pixels"
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):
            st.error(
                "Invalid image. Please upload "
                "a valid JPG or PNG file."
            )

with right_col:
    st.markdown("### 🔍 AI Recognition")

    if image is None:
        st.info(
            "Upload an image to start "
            "the recognition process."
        )

    else:
        st.write(
            "Your image is ready for analysis."
        )

        if st.button(
            "🚀 Analyze Image",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "AI is analyzing your image..."
                ):
                    predictions = predict_image(image)

                st.session_state["analysis"] = {
                    "image_hash": image_hash,
                    "filename": uploaded_file.name,
                    "predictions": predictions
                }

            except Exception as error:
                st.error(
                    f"Analysis failed: {error}"
                )


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

analysis = st.session_state.get("analysis")

if (
    image is not None
    and analysis is not None
    and analysis["image_hash"] == image_hash
):

    predictions = analysis["predictions"]

    st.divider()
    st.header("📊 Analysis Results")

    top_prediction = predictions[0]

    top_label = top_prediction["label"]
    top_confidence = top_prediction["score"] * 100

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Top Prediction",
            top_label.title()
        )

    with metric2:
        st.metric(
            "Confidence Score",
            f"{top_confidence:.2f}%"
        )

    with metric3:
        st.metric(
            "Predictions Generated",
            len(predictions)
        )

    st.divider()

    chart_col, details_col = st.columns(
        [1.3, 1],
        gap="large"
    )

    with chart_col:
        st.subheader("Prediction Comparison")

        chart_data = pd.DataFrame([
            {
                "Category": p["label"].title(),
                "Confidence (%)": round(
                    p["score"] * 100, 2
                )
            }
            for p in predictions
        ])

        st.bar_chart(
            chart_data,
            x="Category",
            y="Confidence (%)",
            horizontal=True,
            sort=False
        )

    with details_col:
        st.subheader("Top 5 Predictions")

        for index, prediction in enumerate(
            predictions, start=1
        ):
            label = prediction["label"].title()
            score = prediction["score"]

            st.write(
                f"**{index}. {label}** "
                f"— {score * 100:.2f}%"
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

    # -----------------------------
    # EXPORT RESULTS
    # -----------------------------

    st.divider()
    st.subheader("📥 Export Analysis")

    export_data = {
        "application": "VisionAI",
        "model": MODEL_NAME,
        "image": analysis["filename"],
        "predictions": predictions
    }

    st.download_button(
        label="Download Results (JSON)",
        data=json.dumps(
            export_data,
            indent=4
        ),
        file_name="visionai_results.json",
        mime="application/json"
    )

    st.caption(
        "AI confidence scores are model estimates, "
        "not guarantees of correctness."
    )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "VisionAI | Deep Learning Image Recognition "
    "| Built with Streamlit & Hugging Face"
)
