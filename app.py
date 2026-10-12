import json
import os
import tempfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Tutupan Lahan Gambut Pulang Pisau",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(tempfile.gettempdir()) / "pp_dashboard_cache"
CACHE_DIR.mkdir(exist_ok=True)

st.title("Peta Tutupan Lahan Gambut Kab. Pulang Pisau")
st.caption("Klasifikasi Tutupan Lahan Menggunakan U-Net (Fusi Sentinel-1 & Sentinel-2), Q3 2024")

# ---------------------------------------------------------
# 2. DEFINISI KELAS & LAYER
# ---------------------------------------------------------
CLASS_INFO = {
    0: {"name": "Hutan", "color": "#1b5e20"},
    1: {"name": "Perkebunan dan Pertanian Lahan Kering", "color": "#fbc02d"},
    2: {"name": "Semak/Belukar", "color": "#8d6e63"},
    3: {"name": "Lahan Terbuka dan Terbangun", "color": "#d32f2f"},
}

# nama menu -> kunci data
LAYERS = {
    # "RGB Sentinel-2": "rgb",
    "Ground Truth": "gt",
    "Prediksi U-Net S1+S2": "s1s2",
    "Prediksi U-Net S2 saja": "s2",
}

LAYER_CAPTION = {
    "rgb": "Komposit median Sentinel-2 (Juli-September 2024), tampilan warna asli (R-G-B).",
    "gt": "Peta acuan hasil rasterisasi shapefile tutupan lahan. ",
    "s1s2": "Hasil klasifikasi U-Net dengan fusi 12 band (Sentinel-2 + Sentinel-1).",
    "s2": "Hasil klasifikasi U-Net dengan 10 band Sentinel-2 saja.",
}


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ---------------------------------------------------------
# 3. SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("PENGATURAN LAYER & FILTER")

selected_layer = st.sidebar.radio("Pilih Layer Visualisasi:", options=list(LAYERS.keys()))
layer_key = LAYERS[selected_layer]
is_rgb = layer_key == "rgb"

opacity = st.sidebar.slider(
    "Transparansi Layer (Opacity):", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="0 = transparan penuh, 1 = pekat penuh",
)

st.sidebar.markdown("---")
st.sidebar.header("LEGENDA & FILTER KELAS")
st.sidebar.caption(
    "Filter kelas tidak berlaku untuk layer RGB." if is_rgb
    else "Centang kelas yang ingin ditampilkan pada peta:"
)

selected_classes = []
for cid, info in CLASS_INFO.items():
    col_color, col_check = st.sidebar.columns([1, 6])
    with col_color:
        st.markdown(
            f'<div style="background-color:{info["color"]};width:20px;height:20px;'
            f'border-radius:3px;margin-top:5px;"></div>',
            unsafe_allow_html=True,
        )
    with col_check:
        if st.checkbox(info["name"], value=True, key=f"chk_class_{cid}", disabled=is_rgb):
            selected_classes.append(cid)

# ---------------------------------------------------------
# 4. MEMUAT DATA (hasil preprocess_data.py)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Memuat data...")
def load_arrays():
    with np.load(DATA_DIR / "layers.npz") as z:
        return {k: z[k] for k in z.files}


@st.cache_data
def load_meta():
    return json.loads((DATA_DIR / "meta.json").read_text())


@st.cache_data
def load_areas():
    return pd.read_csv(DATA_DIR / "luas_per_layer.csv")


if not (DATA_DIR / "meta.json").exists():
    st.error("Folder `data/` belum ada. Jalankan `python preprocess_data.py` terlebih dahulu.")
    st.stop()

meta = load_meta()
bounds = meta["bounds"]          # [[south, west], [north, east]]
arrays = load_arrays()
df_all = load_areas()


def class_overlay_path(key: str, classes: tuple) -> Path | None:
    """Buat PNG transparan berisi hanya kelas yang dicentang. Di-cache ke disk."""
    if key not in arrays:
        return None
    path = CACHE_DIR / f"{key}_{''.join(map(str, classes)) or 'none'}.png"
    if not path.exists():
        arr = arrays[key]
        rgba = np.zeros((*arr.shape, 4), np.uint8)
        for cid in classes:
            m = arr == cid
            rgba[m, :3] = hex_to_rgb(CLASS_INFO[cid]["color"])
            rgba[m, 3] = 255
        tmp = path.with_suffix(f".{os.getpid()}.tmp.png")
        Image.fromarray(rgba, "RGBA").save(tmp, compress_level=3)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------
# 5. PETA
# ---------------------------------------------------------
st.subheader(f"Visualisasi Layer: {selected_layer}")
st.caption(LAYER_CAPTION[layer_key])

overlay_path = None
if is_rgb:
    p = DATA_DIR / "rgb.png"
    overlay_path = p if p.exists() else None
else:
    overlay_path = class_overlay_path(layer_key, tuple(sorted(selected_classes)))

if overlay_path is None:
    st.warning(
        f"Data untuk layer '{selected_layer}' belum tersedia di folder `data/`. "
        "Periksa hasil preprocess_data.py."
    )

# Peta dasar dibuat statis; layer data dikirim lewat feature_group_to_add
# supaya zoom/posisi peta TIDAK reset saat slider atau centang diubah.
base_map = folium.Map(tiles="CartoDB positron", control_scale=True, zoom_start=9)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(base_map)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, Maxar, Earthstar Geographics", name="Citra Satelit (Esri)",
).add_to(base_map)
base_map.fit_bounds(bounds)
folium.LayerControl(collapsed=True).add_to(base_map)

fg = folium.FeatureGroup(name=selected_layer)
if overlay_path is not None:
    folium.raster_layers.ImageOverlay(
        image=str(overlay_path), bounds=bounds, opacity=opacity,
        interactive=False, zindex=2,
    ).add_to(fg)

st_folium(
    base_map,
    feature_group_to_add=fg,
    key="peta_tutupan_lahan",
    height=620,
    use_container_width=True,
    returned_objects=[],
)
st.caption(
    "Tampilan peta disederhanakan (~33 m/piksel) agar ringan. "
    "Luas pada tabel dihitung dari raster resolusi penuh 10 m."
)

# ---------------------------------------------------------
# 6. TABEL ATRIBUT
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Tabel Atribut Layer Saat Ini")

if is_rgb:
    st.info("Layer RGB Sentinel-2 adalah citra komposit, tidak memiliki atribut kelas atau luas.")
else:
    df_layer = df_all[df_all["layer"] == layer_key]
    if df_layer.empty:
        st.warning("Tabel luas untuk layer ini belum tersedia.")
    else:
        filtered = (
            df_layer[df_layer["id_kelas"].isin(selected_classes)]
            [["id_kelas", "nama_kelas", "luas_ha"]]
            .reset_index(drop=True)
        )
        col_tb1, col_tb2 = st.columns([3, 1])
        with col_tb1:
            st.dataframe(
                filtered,
                column_config={
                    "id_kelas": "ID Kelas",
                    "nama_kelas": "Nama Kelas Tutupan Lahan",
                    "luas_ha": st.column_config.NumberColumn("Luas (Hektar)", format="%.2f Ha"),
                },
                hide_index=True,
                use_container_width=True,
            )
        with col_tb2:
            st.metric("Total Luas Terpilih", f"{filtered['luas_ha'].sum():,.2f} Ha")
            st.metric("Jumlah Kelas Aktif", f"{len(filtered)} / {len(CLASS_INFO)}")