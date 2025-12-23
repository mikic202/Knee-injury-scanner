import numpy as np
import pytest
from src.web_app.main import _plot_volume_3d_plotly
import src.web_app.main as main_mod


def _make_sphere_volume(size=32, radius=8):
    c = (np.array((size, size, size)) - 1) / 2.0
    zz, yy, xx = np.indices((size, size, size))
    dist2 = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2
    vol = (dist2 <= radius ** 2).astype(np.float32)
    return vol


def test_returns_figure_with_valid_volume():
    vol = _make_sphere_volume(size=32, radius=8)
    fig = _plot_volume_3d_plotly(vol, level=0.5)
    assert fig is not None, "Expected a plotly Figure for a valid volume"
    assert hasattr(fig, "data")
    assert len(fig.data) >= 1
    first = fig.data[0]
    assert getattr(first, "type", None) == "mesh3d"


def test_raises_if_marching_cubes_and_lewiner_fail(monkeypatch):
    vol = _make_sphere_volume(size=16, radius=4)

    def mc_fail(volume, level=None):
        raise RuntimeError("marching_cubes failed")

    def lew_fail(volume, level):
        raise ValueError("marching_cubes_lewiner failed")

    monkeypatch.setattr(main_mod, "measure", main_mod.measure, raising=False)
    monkeypatch.setattr(main_mod.measure, "marching_cubes", mc_fail, raising=True)
    monkeypatch.setattr(main_mod.measure, "marching_cubes_lewiner", lew_fail, raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        _plot_volume_3d_plotly(vol, level=0.5)
    assert "marching_cubes failed" in str(excinfo.value)


def test_returns_none_when_deps_missing(monkeypatch):
    vol = _make_sphere_volume(size=16, radius=4)
    monkeypatch.setattr(main_mod, "measure", None, raising=True)
    monkeypatch.setattr(main_mod, "go", None, raising=True)

    result = _plot_volume_3d_plotly(vol, level=0.5)
    assert result is None