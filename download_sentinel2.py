"""
Скачиваем снимки Sentinel-2 для задачи NDBI/NDVI по Ташкенту.

Используем COG (Cloud Optimized GeoTIFF) — rasterio читает только нужный
кусок прямо с S3 по HTTP, без скачивания всего файла (~150 МБ → ~50 КБ).

Что качаем:
  B02 blue, B03 green, B04 red          — RGB (True Color)
  B05 rededge1                           — NDVIre
  B08 nir, B8A nir08                    — NDBI, NDVI, BUI, BSI
  B11 swir16, B12 swir22                — NDBI, MNDWI, NDTI

Индексы считаются потом отдельно:
  NDBI  = (B11 - B08) / (B11 + B08)
  NDVI  = (B08 - B04) / (B08 + B04)
  BUI   = NDBI - NDVI
  BSI   = ((B11+B04) - (B08+B02)) / ((B11+B04) + (B08+B02))
  MNDWI = (B03 - B11) / (B03 + B11)
  NDTI  = (B11 - B12) / (B11 + B12)
  NDVIre= (B8A - B05) / (B8A + B05)

Запуск:
  1. setup.bat  (один раз)
  2. .venv/Scripts/activate
  3. python download_sentinel2.py

STAC: Element84 Earth Search v1 (бесплатно, без токенов)
"""

import time
from datetime import date, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta
from shapely.wkt import loads as wkt_loads
from shapely.geometry import mapping

import rasterio
from rasterio.env import Env
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS
from rasterio.warp import transform_geom

from pystac_client import Client
from tqdm import tqdm


# ── настройки ────────────────────────────────────────────────────────────────

ROI_WKT = (
    "POLYGON ((69.290686 41.27213, 69.325104 41.27213, "
    "69.325104 41.290964, 69.290686 41.290964, 69.290686 41.27213))"
)

DATE_START   = date(2019, 1, 1)
DATE_END     = date.today()
CHUNK_MONTHS = 6    # дробим по полгода — чтобы STAC не завис
MAX_CLOUD    = 15   # % облачности

# ключи asset-ов проверены по живому API Element84 (GeoTIFF/COG формат)
BANDS = {
    "B02": "blue",      # Blue   490 nm
    "B03": "green",     # Green  560 nm
    "B04": "red",       # Red    665 nm
    "B05": "rededge1",  # RE1    705 nm
    "B08": "nir",       # NIR    842 nm
    "B8A": "nir08",     # NIR-n  865 nm
    "B11": "swir16",    # SWIR1 1610 nm
    "B12": "swir22",    # SWIR2 2190 nm
}

STAC_URL   = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

OUTPUT_DIR = Path("data/sentinel2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GDAL-настройки для чтения COG по HTTP с публичного S3
# AWS_NO_SIGN_REQUEST — файлы публичные, подпись не нужна
# GDAL_HTTP_MERGE_CONSECUTIVE_REQUESTS — склеивает HTTP-запросы, быстрее
# GDAL_DISABLE_READDIR_ON_OPEN — не сканирует папку при открытии файла
COG_ENV = {
    "AWS_NO_SIGN_REQUEST":              "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_REQUESTS": "YES",
    "GDAL_HTTP_MULTIPLEX":              "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN":     "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY":              "3",
    "GDAL_HTTP_RETRY_DELAY":            "2",
}


# ── функции ──────────────────────────────────────────────────────────────────

def date_chunks(start, end, months):
    chunks = []
    cur = start
    while cur < end:
        chunk_end = min(cur + relativedelta(months=months) - timedelta(days=1), end)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def fetch_cog_window(url, roi_geojson, out_path):
    """
    Читает только нужный кусок COG-файла прямо с S3 (без скачивания целого).
    Трансформирует геометрию ROI в СК растра если нужно, кропает и сохраняет.
    """
    wgs84 = CRS.from_epsg(4326)

    with Env(**COG_ENV):
        with rasterio.open(url) as src:
            # если растр не в WGS84 — переводим геометрию ROI в его СК
            if src.crs != wgs84:
                geom = transform_geom(wgs84, src.crs, roi_geojson)
            else:
                geom = roi_geojson

            clipped, transform = rio_mask(
                src, [geom], crop=True, filled=True,
                nodata=src.nodata if src.nodata is not None else 0,
            )

            profile = src.profile.copy()
            profile.update({
                "driver":    "GTiff",
                "height":    clipped.shape[1],
                "width":     clipped.shape[2],
                "transform": transform,
                "compress":  "lzw",
            })

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(clipped)

    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Sentinel-2 → NDBI + RGB  |  Ташкент  (COG-режим)")
    print(f"  {DATE_START} → {DATE_END}")
    print(f"  Облачность < {MAX_CLOUD}%")
    print(f"  Каналы: {', '.join(BANDS.keys())}  ({len(BANDS)} шт.)")
    print(f"  Выход: {OUTPUT_DIR.resolve()}")
    print("=" * 55 + "\n")

    roi_geojson = mapping(wkt_loads(ROI_WKT))

    print(f"Подключение: {STAC_URL}")
    catalog = Client.open(STAC_URL)
    print(f"OK: {catalog.title}\n")

    chunks = date_chunks(DATE_START, DATE_END, CHUNK_MONTHS)
    print(f"Чанков дат: {len(chunks)}\n")

    saved = skipped = errors = 0

    for i, (d_from, d_to) in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}]  {d_from} → {d_to}")

        try:
            search = catalog.search(
                collections=[COLLECTION],
                intersects=roi_geojson,
                datetime=f"{d_from}/{d_to}",
                query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
                max_items=200,
            )
            items = list(search.items())
        except Exception as e:
            print(f"  [!] STAC-запрос упал: {e}\n")
            errors += 1
            time.sleep(3)
            continue

        print(f"  Найдено сцен: {len(items)}")

        for item in tqdm(items, desc="  сцены", unit="шт", leave=False):
            scene_date = str(item.datetime)[:10]
            cloud = item.properties.get("eo:cloud_cover", "?")

            for band_name, asset_key in BANDS.items():
                out_path = OUTPUT_DIR / f"s2_{band_name}_{scene_date}.tif"

                if out_path.exists() and out_path.stat().st_size > 0:
                    skipped += 1
                    continue

                if asset_key not in item.assets:
                    tqdm.write(
                        f"  [?] {scene_date} {band_name}: нет ключа '{asset_key}'. "
                        f"Есть: {list(item.assets.keys())}"
                    )
                    errors += 1
                    continue

                url = item.assets[asset_key].href

                try:
                    fetch_cog_window(url, roi_geojson, out_path)
                    kb = out_path.stat().st_size // 1024
                    tqdm.write(f"  ✓ {out_path.name}  (облака: {cloud}%, {kb} KB)")
                    saved += 1

                except Exception as e:
                    tqdm.write(f"  [!] {scene_date} {band_name}: {e}")
                    errors += 1
                    if out_path.exists():
                        out_path.unlink()

        time.sleep(1)

    print(f"\n{'=' * 55}")
    print(f"  Готово!")
    print(f"  Сохранено : {saved}")
    print(f"  Пропущено : {skipped}")
    print(f"  Ошибок    : {errors}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
