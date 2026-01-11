import streamlit as st
import pickle
import numpy as np
import torch
from pathlib import Path
from skimage import measure
import plotly.graph_objects as go
import sys

from src.model_architecture.resnet3d.resnet import get_resnet3d
from lime import lime_image
from skimage.segmentation import mark_boundaries

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def explain_lime(model, volume_3d, device, num_samples=500):
    """
    Wyjaśnienie LIME dla środkowego przekroju 3D volumenu.
    Zwraca obrazek z boundaries i dict z predykcją.
    """
    model.eval()
    d_idx = volume_3d.shape[0] // 2 if volume_3d.ndim == 3 else 0
    slice_2d = volume_3d[d_idx, :, :] if volume_3d.ndim == 3 else volume_3d
    # Normalizacja do 0-255
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    if slice_max > slice_min:
        slice_2d_norm = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        slice_2d_norm = np.zeros_like(slice_2d, dtype=np.uint8)
    slice_rgb = np.stack([slice_2d_norm] * 3, axis=-1).astype(np.float32)
    def predict_fn(images):
        batch_size = images.shape[0]
        volumes = []
        for i in range(batch_size):
            img_single = images[i, :, :, 0]
            vol_single = np.tile(img_single[np.newaxis, :, :], (volume_3d.shape[0], 1, 1))
            vol_single = torch.tensor(vol_single, dtype=torch.float32).unsqueeze(0)
            volumes.append(vol_single)
        volumes_batch = torch.cat(volumes, dim=0).to(device)
        with torch.no_grad():
            logits = model(volumes_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        slice_rgb.astype(np.uint8),
        predict_fn,
        top_labels=1,
        num_samples=num_samples,
        hide_color=0
    )
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    img_boundary = mark_boundaries(temp / 255.0, mask)
    preds = predict_fn(slice_rgb[np.newaxis, :, :, :])
    pred_dict = {
        "class_0": float(preds[0, 0]),
        "class_1": float(preds[0, 1]) if preds.shape[1] > 1 else None,
        "class_2": float(preds[0, 2]) if preds.shape[1] > 2 else None,
        "predicted_class": int(np.argmax(preds[0])),
        "top_label": int(top_label)
    }
    return (img_boundary, pred_dict)

def explain_guided_backprop(model, volume_3d, device):
    # Placeholder: tu wstaw kod Guided Backpropagation
    d_idx = volume_3d.shape[0] // 2 if volume_3d.ndim == 3 else 0
    slice_2d = volume_3d[d_idx, :, :] if volume_3d.ndim == 3 else volume_3d
    # Zwróć obrazek z losowym szumem jako placeholder
    return np.random.rand(*slice_2d.shape)

def explain_saliency(model, volume_3d, device):
    # Placeholder: tu wstaw kod Saliency
    d_idx = volume_3d.shape[0] // 2 if volume_3d.ndim == 3 else 0
    slice_2d = volume_3d[d_idx, :, :] if volume_3d.ndim == 3 else volume_3d
    # Zwróć obrazek z losowym szumem jako placeholder
    return np.random.rand(*slice_2d.shape)


@st.cache_resource
def load_model(checkpoint_path: str):
    """Załaduj checkpoint ResNet3D."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet3d(num_classes=3, in_channels=1, device=device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        st.success(f"✓ Model załadowany z: {checkpoint_path}")
        return model, device
    except Exception as e:
        st.error(f"✗ Błąd przy załadowaniu modelu: {e}")
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
    """
    Przewiduj diagnozę na podstawie 3D volumenu.
    
    Returns:
        (predicted_class, probabilities_dict)
    """
    if model is None:
        return None, {}
    
    # Normalizuj volume do 0-1
    vol = volume_3d.astype(np.float32)
    vol_min, vol_max = vol.min(), vol.max()
    if vol_max > vol_min:
        vol = (vol - vol_min) / (vol_max - vol_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    
    # Przeskaluj do target shape za pomocą torch.nn.functional.interpolate
    vol_tensor = torch.tensor(vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    vol_resized = torch.nn.functional.interpolate(
        vol_tensor, 
        size=target_shape, 
        mode='trilinear', 
        align_corners=False
    )  # (1,1,32,128,128)
    
    # Predykcja
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
    return diagnoses.get(class_idx, "❓ Nieznany wynik")


def main():
    st.set_page_config(page_title="Knee Injury Scanner", layout="wide")
    st.title("🏥 Knee Injury Scanner — AI Diagnosis")
    st.write("Analiza zdjęć MRI kolana przy użyciu sieci neuronowej ResNet3D")
    
    # Sidebar
    st.sidebar.header("⚙️ Konfiguracja")
    model_path = st.sidebar.text_input(
        "Ścieżka do checkpointu modelu",
        value="/home/dominika/Projekty_MGR/Knee-injury-scanner/checkpoints/resnet3d_best_10_01_16:49.pt"
    )
    target_shape = (32, 128, 128)
    st.sidebar.info(f"Target shape dla modelu: {target_shape}")
    
    # Załaduj model
    model, device = load_model(model_path)
    
    st.markdown("---")
    
    # Upload sekcja
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

        # Wyciągnij array
        arr = None
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, dict):
            st.write("Klucze słownika:", list(data.keys()))
            for key in ("image", "volume", "img", "data"):
                if key in data:
                    arr = data[key]
                    st.write(f"Wyświetlam zawartość klucza '{key}' — typ: {type(arr)}")
                    break

        if arr is None:
            st.write("Nie znaleziono tablicy 2D/3D w pliku .pck — pokażę reprezentację:")
            st.write(repr(data)[:1000])
        else:
            arr = np.asarray(arr).astype(np.float32)
            st.write(f"📊 Kształt danych: {arr.shape}")
            st.write(f"📈 Zakres intensywności: [{arr.min():.2f}, {arr.max():.2f}]")
            
            # Wyświetl slice
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2D Preview")
                if arr.ndim == 2:
                    st.image(_normalize_for_display(arr), caption="Obraz 2D", use_column_width=True)
                elif arr.ndim == 3:
                    mid = arr.shape[0] // 2
                    st.image(_normalize_for_display(arr[mid]), caption=f"Środkowy przekrój (slice {mid})", use_column_width=True)
            
            # 3D Visualization
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
            
            st.markdown("---")
            
            # Diagnosis section
            st.header("🔬 Diagnoza AI")

            # --- AI session state ---
            if "ai_result" not in st.session_state:
                st.session_state.ai_result = None

            if model is not None and device is not None:
                if st.button("🚀 Uruchom analizę", key="diagnose_btn"):
                    with st.spinner("Analizuję obraz..."):
                        predicted_class, probs = predict_knee_diagnosis(
                            model, device, arr, target_shape=target_shape
                        )
                    st.session_state.ai_result = (predicted_class, probs)

            # Wyświetl wynik AI jeśli jest
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

                # --- XAI SECTION ---
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

                with xai_col2:
                    if st.button("Guided Backprop (slice)"):
                        st.session_state.xai_method = "gbp"
                        st.session_state.xai_result = None

                with xai_col3:
                    if st.button("Saliency (slice)"):
                        st.session_state.xai_method = "saliency"
                        st.session_state.xai_result = None

                # Wykonaj XAI tylko jeśli trzeba
                if st.session_state.xai_method and st.session_state.xai_result is None:
                    with st.spinner("Obliczanie wyjaśnienia..."):
                        if st.session_state.xai_method == "lime":
                            img, pred_dict = explain_lime(model, arr, device)
                            st.session_state.xai_result = ("LIME", img, pred_dict)
                        elif st.session_state.xai_method == "gbp":
                            img = explain_guided_backprop(model, arr, device)
                            st.session_state.xai_result = ("Guided Backprop", img, None)
                        elif st.session_state.xai_method == "saliency":
                            img = explain_saliency(model, arr, device)
                            st.session_state.xai_result = ("Saliency", img, None)

                # Wyświetl wynik XAI jeśli jest
                if st.session_state.xai_result:
                    method, img, pred_dict = st.session_state.xai_result
                    st.subheader(f"Wynik XAI: {method}")
                    st.image(img, caption=f"{method} (middle slice)", use_column_width=True)
                    if pred_dict:
                        st.write(pred_dict)
            else:
                st.error("❌ Model nie został załadowany — sprawdź ścieżkę checkpointu")


if __name__ == "__main__":
    main()