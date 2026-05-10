import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from collections import defaultdict
from skimage.exposure import match_histograms


DATA_DIR = Path("data/sentinel2")
RES = 20
BANDS = {"blue":"B02", "green":"B03", "red":"B04",
         "nir":"B08", "swir1":"B11", "swir2":"B12"}

ndbi_t = 0.0
ndvi_t = 0.2
mndwi_t = 0.0

# эталонная сцена для нормализации - лето 2021, чистое небо
# препод на консультации сказал применять histogram matching
# чтоб убрать разницу в освещенности между датами
REF_DATE = "2021-07-13"
_ref_cache = {}


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


def _get_ref(band, shape):
    # кешируем чтоб не загружать референс каждый раз
    key = (band, shape)
    if key not in _ref_cache:
        _ref_cache[key] = load_band(REF_DATE, band, shape)
    return _ref_cache[key]


def load_band_norm(date, band, shape=None):
    # подгоняем гистограмму к эталонной сцене (skimage histogram matching)
    raw = load_band(date, band, shape)
    if date == REF_DATE:
        return raw
    ref = _get_ref(band, raw.shape)
    # NaN сначала заполняем медианой - match_histograms не любит nan
    raw_f = np.where(np.isnan(raw), np.nanmedian(raw), raw)
    ref_f = np.where(np.isnan(ref), np.nanmedian(ref), ref)
    out = match_histograms(raw_f, ref_f).astype(np.float32)
    out[np.isnan(raw)] = np.nan
    return out


def calc_area(date, thresh=0.0, normalize=False):
    loader = load_band_norm if normalize else load_band
    with rasterio.open(DATA_DIR / f"s2_B11_{date}.tif") as src:
        ref = src.shape

    b11 = loader(date, BANDS["swir1"], ref)
    b08 = loader(date, BANDS["nir"],  ref)
    b04 = loader(date, BANDS["red"],  ref)
    b03 = loader(date, BANDS["green"],ref)

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
