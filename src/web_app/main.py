import streamlit as st
import pickle
import numpy as np
from skimage import measure
import plotly.graph_objects as go


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


def main():
    st.title("Knee Injury Scanner — Demo")
    st.write("Witaj! To prosta aplikacja Streamlit — tutaj później wrzucisz MRI w formacie .pck.")
    st.write("To tylko prototyp interfejsu; funkcje wgrywania i wizualizacji dodamy później.")
    st.sidebar.header("Nawigacja")
    st.sidebar.info("Możesz wgrać plik .pck poniżej (prototyp).")

    uploaded = st.file_uploader("Wgraj plik .pck", type=["pck"])
    if uploaded is not None:
        try:
            b = uploaded.read()
            data = pickle.loads(b)
        except Exception as e:
            st.error(f"Błąd przy wczytywaniu pickle: {e}")
            return

        st.success("Plik .pck wczytany pomyślnie (prototyp)")
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
            arr = np.asarray(arr)
            st.write("Kształt danych:", arr.shape)
            if arr.ndim == 2:
                st.image(_normalize_for_display(arr), caption="Obraz 2D", use_column_width=True)
            elif arr.ndim == 3:
                mid = arr.shape[0] // 2
                st.image(_normalize_for_display(arr[mid]), caption=f"Środkowy przekrój (slice {mid})", use_column_width=True)

                st.markdown("### Wizualizacja 3D (prototyp)")
                if measure is None or go is None:
                    st.warning("Brak wymaganych pakietów do wizualizacji 3D (skimage lub plotly). Zainstaluj 'scikit-image' i 'plotly'.")
                else:
                    vmin, vmax = float(arr.min()), float(arr.max())
                    st.write(f"Zakres intensywności: [{vmin:.2f}, {vmax:.2f}]")
                    level = st.slider("Poziom izosurface (wartość intensywności)", min_value=vmin, max_value=vmax, value=(vmin + vmax) / 2.0)
                    if st.button("Pokaż 3D"):
                        try:
                            fig = _plot_volume_3d_plotly(arr, level=level)
                            if fig is None:
                                st.warning("Nie udało się wygenerować figury 3D.")
                            else:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Błąd podczas tworzenia wizualizacji 3D: {e}")

        st.markdown("---")
        st.subheader("Diagnoza (prototyp)")
        st.info("Brak modelu — tutaj pojawi się wynik diagnozy po implementacji.")
        st.subheader("Wyjaśnialność (prototyp)")
        st.write("Tutaj pokażemy wyjaśnienia/model interpretowalności (np. heatmapy, ważność obszarów).")

if __name__ == "__main__":
    main()