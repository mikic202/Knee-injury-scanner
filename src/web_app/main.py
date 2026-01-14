import streamlit as st
import pickle
import numpy as np
import torch
from pathlib import Path
from skimage import measure
import plotly.graph_objects as go
import sys
import logging

from src.model_architecture.resnet3d.resnet import get_resnet3d
from git_submodules.PyAiWrap.pyaiwrap.xai import LIMEExplainer
from src.explainibility.basic_gradient_based_methods import (
    explain_prediction_with_integrated_gradients,
    explain_prediction_with_saliency,
)

from src.web_app.config import AppConfig

if str(AppConfig.PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(AppConfig.PROJECT_ROOT))


logging.basicConfig(
    level=getattr(logging, AppConfig.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_lime_explainer():
    return LIMEExplainer(segmentation_mode=False)

def explain_lime(model, volume_3d, device, target_class=None):
    logger.info("Starting LIME explanation")
    explainer = get_lime_explainer()
    model.eval()
    if isinstance(volume_3d, np.ndarray):
        tensor = torch.tensor(volume_3d).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        if volume_3d.ndim == 3:
             tensor = volume_3d.unsqueeze(0).unsqueeze(0).float().to(device)
        elif volume_3d.ndim == 4:
             tensor = volume_3d.unsqueeze(0).float().to(device)
        else:
             tensor = volume_3d.float().to(device)
    if target_class is None:
        with torch.no_grad():
            pred = model(tensor)
            target_class = int(pred.argmax(dim=1).item())

    lime_explanation = explainer.explain(model, tensor, target=target_class)
    lime_map = lime_explanation.cpu().squeeze(0).squeeze(0).numpy()
    d_idx = lime_map.shape[0] // 2
    logger.info(f"LIME complete. Slice index: {d_idx}")
    slice_lime = lime_map[d_idx]
    slice_lime = (slice_lime - slice_lime.min()) / (slice_lime.max() - slice_lime.min() + 1e-8)
    return slice_lime, {"target_class": target_class}

def _preprocess_for_model(volume_3d, target_shape=(32, 128, 128)):
    vol = volume_3d.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    vol_resized = torch.nn.functional.interpolate(
        vol_tensor, size=target_shape, mode='trilinear', align_corners=False
    )
    return vol_resized

def explain_integrated_gradients_stream(model, volume_3d, device, target_class=None):
    vol_resized = _preprocess_for_model(volume_3d)
    input_numpy = vol_resized.squeeze(0).cpu().numpy() 
    
    if target_class is None:
        with torch.no_grad():
            logits = model(vol_resized.to(device))
            target_class = int(logits.argmax(dim=1).item())

    attributions = explain_prediction_with_integrated_gradients(
        model, input_numpy, target_class, device
    )
    attr_map = attributions[0] # (32, 128, 128)
    d_idx = attr_map.shape[0] // 2
    slice_map = attr_map[d_idx]
    
    slice_map = np.abs(slice_map)
    slice_map = (slice_map - slice_map.min()) / (slice_map.max() - slice_map.min() + 1e-8)
    
    return slice_map

def explain_saliency_stream(model, volume_3d, device):
    vol_resized = _preprocess_for_model(volume_3d)
    input_numpy = vol_resized.squeeze(0).cpu().numpy()
    
    with torch.no_grad():
        logits = model(vol_resized.to(device))
        target_class = int(logits.argmax(dim=1).item())

    attributions = explain_prediction_with_saliency(
        model, input_numpy, target_class, device
    )
    
    attr_map = attributions[0]
    d_idx = attr_map.shape[0] // 2
    slice_map = attr_map[d_idx]
    slice_map = (slice_map - slice_map.min()) / (slice_map.max() - slice_map.min() + 1e-8)
    
    return slice_map

@st.cache_resource
def load_model(checkpoint_path: str):
    logger.info(f"Loading model from: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet3d(num_classes=3, in_channels=1, device=device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        st.success(f"✓ Model załadowany z: {checkpoint_path}")
        logger.info("Model loaded successfully")
        return model, device
    except Exception as e:
        msg = f"✗ Błąd przy załadowaniu modelu: {e}"
        st.error(msg)
        logger.error(msg)
        return None, None


def _normalize_for_display(arr):
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    a_min, a_max = a.min(), a.max()
    if a_max == a_min:
        return (np.zeros_like(a) + 128).astype(np.uint8)
    norm = (a - a_min) / (a_max - a_min)
    img = (norm * 255).astype(np.uint8)
    return img


def _plot_volume_3d_plotly(volume, level):
    if measure is None or go is None:
        logger.warning("Missing dependencies for 3D plot (skimage, plotly)")
        return None

    vol = np.asarray(volume).astype(np.float32)
    try:
        verts, faces, normals, values = measure.marching_cubes(vol, level=level)
    except Exception as e:
        try:
            verts, faces, normals, values = measure.marching_cubes_lewiner(vol, level)
        except Exception:
            raise e

    x, y, z = verts.T
    i, j, k = faces.T

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=values,
        colorscale='Gray',
        showscale=True,
        opacity=0.6,
    )
    layout = go.Layout(
        scene=dict(aspectmode='auto'),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Izosurface (level={level:.2f})"
    )
    fig = go.Figure(data=[mesh], layout=layout)
    return fig


def predict_knee_diagnosis(model, device, volume_3d: np.ndarray, target_shape=(32, 128, 128)):
    if model is None:
        return None, {}
    
    vol = volume_3d.astype(np.float32)
    vol_min, vol_max = vol.min(), vol.max()
    if vol_max > vol_min:
        vol = (vol - vol_min) / (vol_max - vol_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    
    vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    vol_resized = torch.nn.functional.interpolate(
        vol_tensor, 
        size=target_shape, 
        mode='trilinear', 
        align_corners=False
    )  # (1,1,32,128,128)
    
    vol_resized = vol_resized.to(device)
    with torch.no_grad():
        logits = model(vol_resized)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    predicted_class = int(np.argmax(probs))
    
    return predicted_class, {
        "Zdrowe kolano (0)": float(probs[0]),
        "ACL częściowo zerwane (1)": float(probs[1]),
        "ACL całkowicie zerwane (2)": float(probs[2]),
    }


def get_diagnosis_text(class_idx: int) -> str:
    """Zwróć opis diagnozy."""
    diagnoses = {
        0: "✅ **Kolano jest zdrowe** — nie stwierdzono uszkodzenia więzadła krzyżowego (ACL).",
        1: "⚠️ **ACL częściowo uszkodzone** — więzadło krzyżowe ma częściowe zerwanie. Zalecana konsultacja lekarza.",
        2: "🔴 **ACL całkowicie zerwane** — więzadło krzyżowe jest w pełni zerwane. Wskazana operacja.",
    }
    return diagnoses.get(class_idx, "Nieznany wynik")


def main():
    st.set_page_config(page_title="Knee Injury Scanner", layout="wide")
    st.title("🏥 Knee Injury Scanner — AI Diagnosis")
    st.write("Analiza zdjęć MRI kolana przy użyciu sieci neuronowej ResNet3D")
    
    model_path = AppConfig.MODEL_PATH
    target_shape = (32, 128, 128)
    
    model, device = load_model(model_path)
    
    st.markdown("---")
    
    st.header("📁 Wgraj plik MRI (.pck)")
    uploaded = st.file_uploader("Wgraj plik .pck", type=["pck"])
    
    if uploaded is not None:
        try:
            b = uploaded.read()
            data = pickle.loads(b)
            st.success("✓ Plik .pck wczytany pomyślnie")
        except Exception as e:
            st.error(f"✗ Błąd przy wczytywaniu pickle: {e}")
            return

        arr = None
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, dict):
            logger.debug(f"Pck file dictionary keys: {list(data.keys())}")
            for key in ("image", "volume", "img", "data"):
                if key in data:
                    arr = data[key]
                    st.write(f"Wyświetlam zawartość klucza '{key}' — typ: {type(arr)}")
                    break

        if arr is None:
            st.write("Nie znaleziono tablicy 2D/3D w pliku .pck — pokażę reprezentację:")
            logger.warning("Could not find array in pck file")
            st.write(repr(data)[:1000])
        else:
            arr = np.asarray(arr).astype(np.float32)
            st.write(f"📊 Kształt danych: {arr.shape}")
            st.write(f"📈 Zakres intensywności: [{arr.min():.2f}, {arr.max():.2f}]")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2D Preview")
                if arr.ndim == 2:
                    st.image(_normalize_for_display(arr), caption="Obraz 2D", use_column_width=True)
                elif arr.ndim == 3:
                    mid = arr.shape[0] // 2
                    st.image(_normalize_for_display(arr[mid]), caption=f"Środkowy przekrój (slice {mid})", use_column_width=True)
            
            with col2:
                st.subheader("3D Visualization")
                if arr.ndim == 3:
                    if measure is None or go is None:
                        st.warning("Brak wymaganych pakietów (skimage, plotly)")
                    else:
                        vmin, vmax = float(arr.min()), float(arr.max())
                        level = st.slider("Poziom izosurface", min_value=vmin, max_value=vmax, value=(vmin + vmax) / 2.0)
                        if st.button("Pokaż 3D"):
                            try:
                                fig = _plot_volume_3d_plotly(arr, level=level)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Błąd przy tworzeniu 3D: {e}")
                                logger.error(f"Error creating 3D visualization: {e}")
            
            st.markdown("---")
            
            st.header("🔬 Diagnoza AI")

            if "ai_result" not in st.session_state:
                st.session_state.ai_result = None

            if model is not None and device is not None:
                if st.button("🚀 Uruchom analizę", key="diagnose_btn"):
                    with st.spinner("Analizuję obraz..."):
                        logger.info("Starting diagnosis prediction")
                        predicted_class, probs = predict_knee_diagnosis(
                            model, device, arr, target_shape=target_shape
                        )
                        logger.info(f"Diagnosis prediction done. Class: {predicted_class}")
                    st.session_state.ai_result = (predicted_class, probs)

            if st.session_state.ai_result:
                predicted_class, probs = st.session_state.ai_result
                st.success("✓ Analiza zakończona")
                st.markdown(f"### {get_diagnosis_text(predicted_class)}")
                col_prob1, col_prob2, col_prob3 = st.columns(3)
                with col_prob1:
                    st.metric(
                        "Zdrowe (0)",
                        f"{probs['Zdrowe kolano (0)']:.2%}",
                        delta=None
                    )
                with col_prob2:
                    st.metric(
                        "ACL Częściowe (1)",
                        f"{probs['ACL częściowo zerwane (1)']:.2%}",
                        delta=None
                    )
                with col_prob3:
                    st.metric(
                        "ACL Całkowite (2)",
                        f"{probs['ACL całkowicie zerwane (2)']:.2%}",
                        delta=None
                    )
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                classes = ["Zdrowe", "ACL Częściowe", "ACL Całkowite"]
                values = list(probs.values())
                colors = ['#2ecc71', '#f39c12', '#e74c3c']
                bars = ax.bar(classes, values, color=colors)
                ax.set_ylabel("Prawdopodobieństwo")
                ax.set_ylim([0, 1])
                ax.set_title("Rozkład predykcji modelu")
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{val:.2%}", ha='center', va='bottom', fontweight='bold')
                st.pyplot(fig)

                st.header("🧠 Wyjaśnialność (XAI)")
                if "xai_result" not in st.session_state:
                    st.session_state.xai_result = None
                if "xai_method" not in st.session_state:
                    st.session_state.xai_method = None

                xai_col1, xai_col2, xai_col3 = st.columns(3)

                with xai_col1:
                    if st.button("LIME (slice)"):
                        st.session_state.xai_method = "lime"
                        st.session_state.xai_result = None

                    if st.session_state.xai_method == "lime" and st.session_state.xai_result is None:
                        with st.spinner("Obliczanie LIME..."):
                            logger.info("Computing LIME on demand")
                            lime_img, pred_dict = explain_lime(model, arr, device)
                            st.session_state.xai_result = ("LIME", lime_img, pred_dict)

                    if st.session_state.xai_result and st.session_state.xai_result[0] == "LIME":
                        st.image(st.session_state.xai_result[1], caption="LIME (middle slice)", use_column_width=True)
                        st.write(st.session_state.xai_result[2])

                with xai_col2:
                    if st.button("Integrated Gradients (slice)"):
                        st.session_state.xai_method = "ig"
                        st.session_state.xai_result = None

                with xai_col3:
                    if st.button("Saliency (slice)"):
                        st.session_state.xai_method = "saliency"
                        st.session_state.xai_result = None

                if st.session_state.xai_method and st.session_state.xai_result is None:
                    with st.spinner("Obliczanie wyjaśnienia..."):
                        logger.info(f"Computing XAI: {st.session_state.xai_method}")
                        if st.session_state.xai_method == "lime":
                            img, pred_dict = explain_lime(model, arr, device)
                            st.session_state.xai_result = ("LIME", img, pred_dict)
                        elif st.session_state.xai_method == "ig":
                            img = explain_integrated_gradients_stream(model, arr, device)
                            st.session_state.xai_result = ("Integrated Gradients", img, None)
                        elif st.session_state.xai_method == "saliency":
                            img = explain_saliency_stream(model, arr, device)
                            st.session_state.xai_result = ("Saliency", img, None)

                if st.session_state.xai_result:
                    method, img, pred_dict = st.session_state.xai_result
                    st.subheader(f"Wynik XAI: {method}")
                    st.image(img, caption=f"{method} (middle slice)", use_column_width=True)
                    if pred_dict:
                        st.write(pred_dict)


if __name__ == "__main__":
    main()