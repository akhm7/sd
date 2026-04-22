import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path("data/sentinel2")
RES = 20
BANDS = {"blue":"B02", "green":"B03", "red":"B04",
         "nir":"B08", "swir1":"B11", "swir2":"B12"}

ndbi_t = 0.0
ndvi_t = 0.2
mndwi_t = 0.0


def get_dates():
    db = defaultdict(set)
    for f in DATA_DIR.glob("s2_B*.tif"):
        p = f.stem.split("_")
        if len(p)==3:
            db[p[2]].add(p[1])
    # оставляем только даты где есть все нужные каналы
    return sorted([d for d,b in db.items()
                   if all(x in b for x in ("B11","B08","B04","B03"))])


def load_band(date, band, shape=None):
    path = DATA_DIR / f"s2_{band}_{date}.tif"
    with rasterio.open(path) as src:
        if shape and src.shape != shape:
            data = src.read(1, out_shape=(1,*shape),
                          resampling=Resampling.bilinear).astype(np.float32)
        else:
            data = src.read(1).astype(np.float32)
        nd = src.nodata
        if nd is not None: data[data==nd] = np.nan
        data[data==0] = np.nan
    return data


def norm_idx(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a-b) / (a+b)
    r[~np.isfinite(r)] = np.nan
    return r


def calc_area(date, thresh=0.0):
    with rasterio.open(DATA_DIR / f"s2_B11_{date}.tif") as src:
        ref = src.shape

    b11 = load_band(date, BANDS["swir1"], ref)
    b08 = load_band(date, BANDS["nir"],  ref)
    b04 = load_band(date, BANDS["red"],  ref)
    b03 = load_band(date, BANDS["green"],ref)

    ndbi = norm_idx(b11, b08)
    ndvi = norm_idx(b08, b04)
    mndwi= norm_idx(b03, b11)

    mask = ((ndbi > thresh) & (ndvi < ndvi_t) &
            (mndwi < mndwi_t) & np.isfinite(ndbi))

    area = float(mask.sum() * (RES**2) / 10000)
    return area, mask, {"ndbi":ndbi, "ndvi":ndvi, "mndwi":mndwi, "shape":ref}


def load_rgb(date, shape=None):
    r = load_band(date, BANDS["red"], shape)
    g = load_band(date, BANDS["green"], shape)
    b = load_band(date, BANDS["blue"], shape)
    rgb = np.dstack([r,g,b]).astype(float)
    p2, p98 = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    rgb = np.clip((rgb-p2)/(p98-p2+1e-10), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def show_map(date, thresh=0.0):
    try:
        area, mask, idx = calc_area(date, thresh)
    except FileNotFoundError:
        print(f"нет файлов для {date}")
        return
    rgb = load_rgb(date, idx["shape"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(rgb); ax[0].set_title(f"True Color\n{date}"); ax[0].axis("off")
    im = ax[1].imshow(idx["ndbi"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
    ax[1].set_title("NDBI"); ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ov = rgb.copy()
    ov[mask,0]=1.0; ov[mask,1]=0.2; ov[mask,2]=0.2
    ax[2].imshow(ov); ax[2].set_title(f"маска\n{area:.1f} га"); ax[2].axis("off")
    plt.suptitle(f"Ташкент {date}", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()
    return fig


def render_map(date, thresh=0.0):
    # то же что show_map но без plt.show - чтоб сохранять fig в файл
    try:
        area, mask, idx = calc_area(date, thresh)
    except FileNotFoundError:
        return None
    rgb = load_rgb(date, idx["shape"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].imshow(rgb); ax[0].set_title(f"True Color\n{date}"); ax[0].axis("off")
    im = ax[1].imshow(idx["ndbi"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
    ax[1].set_title("NDBI"); ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ov = rgb.copy()
    ov[mask,0]=1.0; ov[mask,1]=0.2; ov[mask,2]=0.2
    ax[2].imshow(ov); ax[2].set_title(f"маска\n{area:.1f} га"); ax[2].axis("off")
    plt.suptitle(f"Ташкент {date}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
