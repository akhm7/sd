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
            ndbi = idx["ndbi"]
            rows.append({
                "date": d,
                "year": int(d[:4]),
                "month": int(d[5:7]),
                "area_t0": round(a0, 1),
                "area_t005": round(a5, 1),
                "ndbi_mean": round(float(np.nanmean(ndbi)), 4),
                "ndbi_std": round(float(np.nanstd(ndbi)), 4),
            })
        except Exception as e:
            print(f"пропуск {d}: {e}")

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"results.csv: {len(rows)} строк")

    # годовая сводка по лету
    summer = [r for r in rows if r["month"] in (6,7,8) and r["area_t0"] > 50]
    by_year = defaultdict(list)
    by_year5 = defaultdict(list)
    for r in summer:
        by_year[r["year"]].append(r["area_t0"])
        by_year5[r["year"]].append(r["area_t005"])

    summary = []
    for y in sorted(by_year.keys()):
        vals = by_year[y]
        vals5 = by_year5[y]
        summary.append({
            "year": y,
            "median": round(np.median(vals), 1),
            "min": round(np.min(vals), 1),
            "max": round(np.max(vals), 1),
            "q25": round(np.percentile(vals, 25), 1),
            "q75": round(np.percentile(vals, 75), 1),
            "n_scenes": len(vals),
            "median_t005": round(np.median(vals5), 1),
        })
        print(f"  {y}: {summary[-1]['median']:.0f} га ({len(vals)} сцен)")

    with open(OUT / "yearly_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    # карты по годам (берем середину июля)
    all_dates_by_year = defaultdict(list)
    for d in dates:
        all_dates_by_year[d[:4]].append(d)

    for year, yd in sorted(all_dates_by_year.items()):
        best = None
        for mo in ("07","06","08","05","04","03"):
            c = [d for d in yd if d[5:7]==mo]
            if c:
                best = c[len(c)//2]
                break
        if not best:
            best = yd[-1]

        print(f"карта {year}: {best}")
        try:
            area, mask, idx = calc_area(best, thresh=0.0)
            rgb = load_rgb(best, idx["shape"])

            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            ax[0].imshow(rgb)
            ax[0].set_title(f"True Color\n{best}")
            ax[0].axis("off")

            im = ax[1].imshow(idx["ndbi"], cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
            ax[1].set_title("NDBI")
            ax[1].axis("off")
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

            ov = rgb.copy()
            ov[mask,0]=1.0; ov[mask,1]=0.2; ov[mask,2]=0.2
            ax[2].imshow(ov)
            ax[2].set_title(f"маска\n{area:.1f} га")
            ax[2].axis("off")

            plt.suptitle(f"Ташкент {best}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            fig.savefig(OUT / f"maps/map_{best}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"ошибка: {e}")

    print("готово")


if __name__ == "__main__":
    main()
