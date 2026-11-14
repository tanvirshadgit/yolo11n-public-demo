from pathlib import Path
from io import BytesIO
import csv

import streamlit as st
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO11n Research Demo",
    page_icon="🔬",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SEED = 123
NOISE_LEVELS = [0, 5, 10, 20]

st.title("🔬 YOLO11n Object Detection Research Demo")
st.caption(
    "Bounding Box Annotation Noise | "
    "COCO2017 | Four Models | Seed 123"
)

st.info(
    "This is a visual demonstration using trained research "
    "checkpoints, not a substitute for independent test-set evaluation."
)

@st.cache_resource(show_spinner="Loading model...")
def load_model(noise):
    path = MODEL_DIR / f"noise_{noise}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return YOLO(str(path))

device = "cpu"

with st.sidebar:
    st.header("Demo Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )
    st.write("Device:", device)
    st.write("Model: YOLO11n")
    st.write("Training seed:", SEED)

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.write("Upload an image to start the comparison.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Cannot open image: {exc}")
    st.stop()

st.subheader("Original Image")
st.image(image, width=420)

if st.button("Run Four-Model Comparison", type="primary"):
    panels = []
    records = []
    summary = []

    progress = st.progress(0)

    for index, noise in enumerate(NOISE_LEVELS):
        try:
            model = load_model(noise)

            result = model.predict(
                source=image,
                imgsz=640,
                conf=confidence,
                device=device,
                verbose=False
            )[0]

            annotated = Image.fromarray(
                result.plot()[:, :, ::-1]
            ).convert("RGB")

            panels.append((noise, annotated))

            count = len(result.boxes)
            summary.append({
                "Noise (%)": noise,
                "Detections": count
            })

            for box in result.boxes:
                class_id = int(box.cls.item())
                xyxy = box.xyxy[0].tolist()

                records.append({
                    "noise_percent": noise,
                    "seed": SEED,
                    "class": model.names[class_id],
                    "confidence": round(float(box.conf.item()), 6),
                    "x1": round(xyxy[0], 2),
                    "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2),
                    "y2": round(xyxy[3], 2)
                })

        except Exception as exc:
            st.error(f"Noise {noise}% failed: {exc}")
            st.stop()

        progress.progress((index + 1) / len(NOISE_LEVELS))

    st.subheader("Side-by-Side Detection Results")

    cols = st.columns(2)
    for index, (noise, annotated) in enumerate(panels):
        with cols[index % 2]:
            st.markdown(f"**Noise {noise}% — Seed {SEED}**")

            preview = annotated.copy()
            preview.thumbnail((420, 320))

            st.image(preview, width="content")

    st.subheader("Detection Summary")
    st.dataframe(pd.DataFrame(summary), hide_index=True)

    st.subheader("Detailed Detection Report")
    df = pd.DataFrame(records)
    st.dataframe(df, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Detection CSV",
        data=csv_bytes,
        file_name="yolo11n_detection_comparison.csv",
        mime="text/csv",
        on_click="ignore"
    )

    width, height = panels[0][1].size
    header = 50

    canvas = Image.new(
        "RGB",
        (width * 2, (height + header) * 2),
        "white"
    )

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)

    for index, (noise, annotated) in enumerate(panels):
        x = (index % 2) * width
        y = (index // 2) * (height + header)

        draw.text(
            (x + 15, y + 15),
            f"YOLO11n | Noise {noise}% | Seed {SEED}",
            fill="black"
        )

        canvas.paste(annotated, (x, y + header))

    buffer = BytesIO()
    canvas.save(buffer, format="PNG")

    st.download_button(
        "Download Comparison Image",
        data=buffer.getvalue(),
        file_name="yolo11n_four_model_comparison.png",
        mime="image/png",
        on_click="ignore"
    )

    st.success("Four-model comparison completed successfully!")
