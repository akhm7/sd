import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from utils import get_dates, calc_area, load_rgb


OUT = Path("streamlit_data")
OUT.mkdir(exist_ok=True)
(OUT / "maps").mkdir(exist_ok=True)


def main():
    dates = get_dates()
    print(f"всего дат: {len(dates)}")

    rows = []
    for d in tqdm(dates, desc="площади"):
        try:
            a0, _, idx = calc_area(d, thresh=0.0)
            a5, _, _ = calc_area(d, thresh=0.05)
            a0n, _, _ = calc_area(d, thresh=0.0, normalize=True)
            a5n, _, _ = calc_area(d, thresh=0.05, normalize=True)
            ndbi = idx["ndbi"]
            ndvi = idx["ndvi"]
            veg_mask = (ndvi > 0.3) & np.isfinite(ndvi)
            a_veg = float(veg_mask.sum() * 400 / 10000)
            rows.append({
                "date": d,
                "year": int(d[:4]),
                "month": int(d[5:7]),
                "area_t0": round(a0, 1),
                "area_t005": round(a5, 1),
                "area_t0_norm": round(a0n, 1),
                "area_t005_norm": round(a5n, 1),
                "area_veg": round(a_veg, 1),
                "ndbi_mean": round(float(np.nanmean(ndbi)), 4),
                "ndbi_std": round(float(np.nanstd(ndbi)), 4),
                "ndvi_mean": round(float(np.nanmean(ndvi)), 4),
            })
        except Exception as e:
            print(f"пропуск {d}: {e}")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"results.csv: {len(rows)} строк")

    summer = [r for r in rows if r["month"] in (6,7,8) and r["area_t0"] > 50]
    by_year = defaultdict(list)
    by_year5 = defaultdict(list)
    by_year_n = defaultdict(list)
    by_year_n5 = defaultdict(list)
    by_year_v = defaultdict(list)
    for r in summer:
        by_year[r["year"]].append(r["area_t0"])
        by_year5[r["year"]].append(r["area_t005"])
        by_year_n[r["year"]].append(r["area_t0_norm"])
        by_year_n5[r["year"]].append(r["area_t005_norm"])
        by_year_v[r["year"]].append(r["area_veg"])

    summary = []
    for y in sorted(by_year.keys()):
        vals = by_year[y]
        vals5 = by_year5[y]
        valsn = by_year_n[y]
        valsn5 = by_year_n5[y]
        valsv = by_year_v[y]
        summary.append({
            "year": y,
            "median": round(np.median(vals), 1),
            "min": round(np.min(vals), 1),
            "max": round(np.max(vals), 1),
            "q25": round(np.percentile(vals, 25), 1),
            "q75": round(np.percentile(vals, 75), 1),
            "n_scenes": len(vals),
            "median_t005": round(np.median(vals5), 1),
            "min_t005": round(np.min(vals5), 1),
            "max_t005": round(np.max(vals5), 1),
            "median_norm": round(np.median(valsn), 1),
            "min_norm": round(np.min(valsn), 1),
            "max_norm": round(np.max(valsn), 1),
            "median_norm_t005": round(np.median(valsn5), 1),
            "min_norm_t005": round(np.min(valsn5), 1),
            "max_norm_t005": round(np.max(valsn5), 1),
            "median_veg": round(np.median(valsv), 1),
        })
        print(f"  {y}: {summary[-1]['median']:.0f} га / зел {summary[-1]['median_veg']:.0f} ({len(vals)} сцен)")

    with open(OUT / "yearly_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    by_year = defaultdict(list)
    for d in dates:
        by_year[d[:4]].append(d)

    combos = [
        ("t0", 0.0, False),
        ("t005", 0.05, False),
        ("t0_norm", 0.0, True),
        ("t005_norm", 0.05, True),
    ]

    for year, yd in sorted(by_year.items()):
        s, w = None, None
        for mo in ("07","06","08","05","04","03"):
            cand = [d for d in yd if d[5:7]==mo]
            if cand: s = cand[len(cand)//2]; break
        for mo in ("01","12","02","11","03"):
            cand = [d for d in yd if d[5:7]==mo]
            if cand: w = cand[len(cand)//2]; break

        for d, lbl in [(s, "summer"), (w, "winter")]:
            if not d: continue
            for key, th, norm in combos:
                try:
                    print(f"карта {year} {lbl} {key}: {d}")
                    area, mask, idx = calc_area(d, thresh=th, normalize=norm)
                    rgb = load_rgb(d, idx["shape"])
                    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
                    ax[0].imshow(rgb); ax[0].set_title(f"True Color\n{d}"); ax[0].axis("off")
                    im = ax[1].imshow(idx["ndbi"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
                    ax[1].set_title("NDBI"); ax[1].axis("off")
                    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
                    ov = rgb.copy()
                    ov[mask,0]=1.0; ov[mask,1]=0.2; ov[mask,2]=0.2
                    ax[2].imshow(ov); ax[2].set_title(f"маска\n{area:.1f} га"); ax[2].axis("off")
                    suf = "норм." if norm else ""
                    plt.suptitle(f"Ташкент {d} | порог {th} {suf}", fontsize=13, fontweight="bold")
                    plt.tight_layout()
                    fig.savefig(OUT / f"maps/map_{lbl}_{key}_{d}.png", dpi=120)
                    plt.close(fig)
                except Exception as e:
                    print(f"ошибка {d} {key}: {e}")

    print("готово")


if __name__ == "__main__":
    main()
