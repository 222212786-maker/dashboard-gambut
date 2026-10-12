"""
preprocess_data.py
==================
Jalankan SEKALI (di laptop atau Colab) sebelum menjalankan dashboard.

Tugasnya: mengubah GeoTIFF besar (19.718 x 8.514 piksel, EPSG:32749, 10 m)
menjadi file ringan di folder `data/` yang bisa dibaca dashboard Streamlit:

  data/layers.npz         -> array kelas (Ground Truth, Prediksi S1+S2, Prediksi S2), EPSG:4326, disederhanakan
  data/rgb.png            -> komposit RGB Sentinel-2 (RGBA, transparan di luar AOI)
  data/meta.json          -> batas peta & parameter
  data/luas_per_layer.csv -> luas (ha) per kelas, dihitung dari RESOLUSI PENUH 10 m (bukan dari peta yang disederhanakan)

Dependensi: rasterio, numpy, pandas, pillow
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from PIL import Image

# =====================================================================
# EDIT BAGIAN INI
# =====================================================================
# Salin folder dari Google Drive ke komputer, lalu sesuaikan path-nya.
BASE = Path(r"D:/ROAD TO WISUDA ZHIFA/Bahan Sidang/GEE_Skripsi_PulangPisau_Fix")

LAYERSTACK_GLOB = str(BASE / "GEE_LayerStack_PP" / "*.tif")           # 6 pecahan, 12 band
GT_TIF          = BASE / "output" / "Landcover_LabelTarget_4Kelas.tif"
PRED_S1S2_TIF   = BASE / "output" / "Peta_Tutupan_Lahan_PulangPisau_2024_run1_fusi_s1s2.tif"
PRED_S2_TIF     = BASE / "output" / "Peta_Tutupan_Lahan_PulangPisau_2024_run2_optik_s2.tif"

OUT_DIR = Path("data")

# Resolusi peta tampilan (derajat). 0.0003 deg ~ 33 m. Makin kecil = makin tajam tapi makin berat.
TARGET_RES_DEG = 0.0003

# Urutan band RGB (1-based: Red, Green, Blue) di GeoTIFF LayerStack.
# PERIKSA! Lihat catatan band di jawaban. Skrip lebih dulu mencoba membaca nama band
# dari metadata GeoTIFF; nilai ini hanya dipakai bila metadata kosong.
RGB_BANDS_FALLBACK = (1, 2, 3)
RGB_MAX = 0.3          # sama dengan visualisasi di GEE (min 0, max 0.3)

NODATA = 255
CLASS_NAMES = [
    "Hutan",
    "Perkebunan dan Pertanian Lahan Kering",
    "Semak/Belukar",
    "Lahan Terbuka dan Terbangun",
]
# id_merge_n pada shapefile = id_kelas model + 1
# =====================================================================


def area_stats(tif_path: Path, n_classes: int = 4, rows_per_block: int = 1024):
    """Hitung jumlah piksel per kelas pada resolusi penuh, per blok baris (hemat memori)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    with rasterio.open(tif_path) as src:
        assert src.crs is not None and src.crs.is_projected, f"{tif_path.name}: CRS harus terproyeksi (meter)."
        pixel_ha = abs(src.res[0] * src.res[1]) / 10_000.0
        for r in range(0, src.height, rows_per_block):
            h = min(rows_per_block, src.height - r)
            a = src.read(1, window=Window(0, r, src.width, h))
            a = a[a < n_classes]  # buang NODATA (255) dan nilai di luar kelas
            counts += np.bincount(a.ravel(), minlength=n_classes)[:n_classes]
    return counts, pixel_ha


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Grid tujuan (EPSG:4326) dari grid label ----------
    with rasterio.open(GT_TIF) as gt:
        src_crs, src_shape, src_transform = gt.crs, (gt.height, gt.width), gt.transform
        dst_transform, dst_w, dst_h = calculate_default_transform(
            gt.crs, "EPSG:4326", gt.width, gt.height, *gt.bounds, resolution=TARGET_RES_DEG
        )
    west, north = dst_transform.c, dst_transform.f
    east = west + dst_w * dst_transform.a
    south = north + dst_h * dst_transform.e
    print(f"Grid tampilan: {dst_h} x {dst_w} piksel | batas (S,W,N,E) = {south:.4f}, {west:.4f}, {north:.4f}, {east:.4f}")

    # ---------- 2. Layer kelas: luas penuh + reproject (nearest) ----------
    layers = {"gt": GT_TIF, "s1s2": PRED_S1S2_TIF, "s2": PRED_S2_TIF}
    arrays, area_rows = {}, []
    for key, path in layers.items():
        if not Path(path).exists():
            print(f"[LEWATI] {key}: file tidak ditemukan -> {path}")
            continue
        with rasterio.open(path) as src:
            assert (src.height, src.width) == src_shape and src.transform == src_transform and src.crs == src_crs, (
                f"Grid {Path(path).name} berbeda dari Ground Truth. Semua raster harus satu grid."
            )
            dst = np.full((dst_h, dst_w), NODATA, np.uint8)
            reproject(
                source=rasterio.band(src, 1), destination=dst,
                src_transform=src.transform, src_crs=src.crs, src_nodata=NODATA,
                dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=NODATA,
                resampling=Resampling.nearest,
            )
        arrays[key] = dst

        counts, pixel_ha = area_stats(Path(path), len(CLASS_NAMES))
        for cid, name in enumerate(CLASS_NAMES):
            area_rows.append(dict(
                layer=key, id=cid + 1, id_kelas=cid, nama_kelas=name,
                piksel=int(counts[cid]), luas_ha=round(float(counts[cid] * pixel_ha), 2),
            ))
        print(f"[OK] {key}: total {counts.sum() * pixel_ha:,.2f} ha")

    np.savez_compressed(OUT_DIR / "layers.npz", **arrays)
    pd.DataFrame(area_rows).to_csv(OUT_DIR / "luas_per_layer.csv", index=False)

    # ---------- 3. RGB Sentinel-2 ----------
    tiles = sorted(glob.glob(LAYERSTACK_GLOB))
    assert tiles, f"LayerStack tidak ditemukan: {LAYERSTACK_GLOB}"
    rgb = np.full((3, dst_h, dst_w), np.nan, np.float32)
    used_bands = None
    for t in tiles:
        with rasterio.open(t) as src:
            desc = list(src.descriptions)
            if {"B4", "B3", "B2"} <= set(desc):
                bands = [desc.index("B4") + 1, desc.index("B3") + 1, desc.index("B2") + 1]
                how = "dari metadata nama band"
            else:
                bands = list(RGB_BANDS_FALLBACK)
                how = "FALLBACK manual (metadata band kosong)"
            if used_bands is None:
                used_bands = bands
                print(f"Band RGB (R,G,B) = {bands} -> {how}")
            tmp = np.full((3, dst_h, dst_w), np.nan, np.float32)
            reproject(
                source=rasterio.band(src, bands), destination=tmp,
                src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                dst_transform=dst_transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        fill = np.isnan(rgb)
        rgb[fill] = tmp[fill]
        del tmp

    valid = np.isfinite(rgb).all(axis=0) & (rgb.sum(axis=0) > 0)
    img = np.clip(np.nan_to_num(rgb) / RGB_MAX, 0, 1)
    rgba = np.zeros((dst_h, dst_w, 4), np.uint8)
    rgba[..., :3] = (np.moveaxis(img, 0, -1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)
    Image.fromarray(rgba, "RGBA").save(OUT_DIR / "rgb.png", optimize=True)

    # ---------- 4. Metadata ----------
    meta = dict(
        bounds=[[south, west], [north, east]],
        target_res_deg=TARGET_RES_DEG,
        rgb_bands_used=used_bands,
        layers_available=list(arrays.keys()),
    )
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Selesai. Isi folder data/:", sorted(p.name for p in OUT_DIR.iterdir()))


if __name__ == "__main__":
    main()