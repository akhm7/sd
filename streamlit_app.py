import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


DATA_DIR = Path("streamlit_data")
MAPS_DIR = DATA_DIR / "maps"


@st.cache_data
def load_results():
    df = pd.read_csv(DATA_DIR / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_summary():
    return pd.read_csv(DATA_DIR / "yearly_summary.csv")


def get_map_dates():
    maps = sorted(MAPS_DIR.glob("map_*.png"))
    return {p.stem.replace("map_",""): p for p in maps}


st.set_page_config(
    page_title="Застройка Ташкента",
    page_icon="🏗️",
    layout="wide"
)

# боковая панель
st.sidebar.title("Анализ застройки")
st.sidebar.markdown("Sentinel-2 NDBI, 2019-2026")
st.sidebar.markdown("---")

thresh = st.sidebar.radio(
    "Порог NDBI",
    ["NDBI > 0 (стандарт)", "NDBI > 0.05"],
    index=0
)
col_area = "area_t0" if "0 " in thresh else "area_t005"

season = st.sidebar.selectbox(
    "Сезон",
    ["Лето (июнь-август)", "Все сезоны", "Зима (дек-фев)"]
)

df = load_results()
summary = load_summary()

# фильтр сезона
if "Лето" in season:
    df_f = df[df["month"].isin([6,7,8])]
elif "Зима" in season:
    df_f = df[df["month"].isin([12,1,2])]
else:
    df_f = df.copy()

# фильтр по площади (убираем мусор)
df_f = df_f[df_f[col_area] > 50]

year_min = int(df_f["year"].min())
year_max = int(df_f["year"].max())
yr_range = st.sidebar.slider("Годы", year_min, year_max, (year_min, year_max))
df_f = df_f[(df_f["year"] >= yr_range[0]) & (df_f["year"] <= yr_range[1])]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Территория**: ЖК Assalom Sohil\n\n"
    "Яшнабадский р-н, ~3x2 км\n\n"
    "**Разрешение**: 20м/пиксель"
)
st.sidebar.markdown(
    "[GitHub](https://github.com/akhm7/sd)"
)


tab1, tab2, tab3, tab4 = st.tabs(["Динамика", "Карты", "Данные", "Методология"])


with tab1:
    st.header("Динамика застройки")

    # годовая сводка
    col_med = "median" if "0 " in thresh else "median_t005"

    if "Лето" in season:
        # используем предрассчитанную сводку
        sm = summary[(summary["year"] >= yr_range[0]) & (summary["year"] <= yr_range[1])]

        if len(sm) >= 2:
            c1, c2, c3 = st.columns(3)
            first = sm.iloc[0][col_med]
            last = sm.iloc[-1][col_med]
            c1.metric("Старт", f"{first:.0f} га", delta=None)
            c2.metric("Финиш", f"{last:.0f} га", delta=f"{last-first:+.0f} га")
            c3.metric("Рост", f"{(last-first)/first*100:+.1f}%")

            # столбчатая диаграмма с error bars
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(sm))
            meds = sm[col_med].values
            if col_med == "median":
                yerr_low = meds - sm["min"].values
                yerr_high = sm["max"].values - meds
            else:
                yerr_low = np.zeros(len(sm))
                yerr_high = np.zeros(len(sm))

            ax.bar(x, meds, color="#e74c3c", alpha=0.8)
            if col_med == "median":
                ax.errorbar(x, meds, yerr=[yerr_low, yerr_high],
                           fmt="none", color="black", capsize=4, lw=1.5)
            for i, m in enumerate(meds):
                ax.text(i, m+3, f"{m:.0f}", ha="center", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(sm["year"].values)
            ax.set_xlabel("год")
            ax.set_ylabel("площадь, га")
            ax.set_title(f"застройка по годам ({thresh}, лето)")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        # для не-лета считаем сводку из данных
        by_yr = df_f.groupby("year")[col_area].agg(["median","min","max","count"])

        if len(by_yr) >= 2:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(by_yr))
            meds = by_yr["median"].values
            ax.bar(x, meds, color="#e74c3c", alpha=0.8)
            yerr_low = meds - by_yr["min"].values
            yerr_high = by_yr["max"].values - meds
            ax.errorbar(x, meds, yerr=[yerr_low, yerr_high],
                       fmt="none", color="black", capsize=4, lw=1.5)
            for i, m in enumerate(meds):
                ax.text(i, m+3, f"{m:.0f}", ha="center", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(by_yr.index)
            ax.set_xlabel("год")
            ax.set_ylabel("площадь, га")
            ax.set_title(f"застройка по годам ({thresh})")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # временной ряд
    if len(df_f) > 5:
        st.subheader("Временной ряд")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_f["date"], df_f[col_area],
                "o-", color="#e74c3c", lw=1, ms=4, alpha=0.7, label="площадь")

        # тренд
        xn = mdates.date2num(df_f["date"])
        z = np.polyfit(xn, df_f[col_area].values, 1)
        ax.plot(df_f["date"], np.poly1d(z)(xn),
                "--", color="#c0392b", lw=2, label="тренд")

        # скользящее среднее
        w = min(10, len(df_f)//3)
        if w >= 3:
            sm_vals = np.convolve(df_f[col_area].values,
                                  np.ones(w)/w, mode="valid")
            off = w//2
            dates_sm = df_f["date"].values[off:off+len(sm_vals)]
            ax.plot(dates_sm, sm_vals, "-", color="#2ecc71",
                    lw=2, alpha=0.8, label=f"скольз.среднее ({w})")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("год")
        ax.set_ylabel("площадь, га")
        ax.set_title(f"динамика застройки ({thresh})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


with tab2:
    st.header("Карты по годам")
    st.markdown("True Color + NDBI + маска застройки (NDBI > 0)")

    map_dates = get_map_dates()
    if map_dates:
        years_available = sorted(map_dates.keys())
        selected = st.selectbox("Дата снимка", years_available,
                               format_func=lambda d: f"{d[:4]} ({d})")

        if selected in map_dates:
            st.image(str(map_dates[selected]), use_container_width=True)

            # показать площадь для этой даты
            row = df[df["date"] == pd.Timestamp(selected)]
            if len(row) > 0:
                r = row.iloc[0]
                st.markdown(
                    f"**Площадь**: {r['area_t0']:.1f} га (NDBI > 0) / "
                    f"{r['area_t005']:.1f} га (NDBI > 0.05)"
                )
    else:
        st.warning("Карты не найдены в streamlit_data/maps/")


with tab3:
    st.header("Данные")

    st.subheader("Все наблюдения")
    st.dataframe(
        df_f[["date","year","month",col_area,"ndbi_mean","ndbi_std"]].rename(
            columns={col_area: "площадь_га", "ndbi_mean": "NDBI_ср", "ndbi_std": "NDBI_std"}
        ),
        use_container_width=True,
        height=400
    )

    st.download_button(
        "Скачать CSV",
        df_f.to_csv(index=False).encode("utf-8"),
        "sentinel2_results.csv",
        "text/csv"
    )

    st.subheader("Сводка по годам (лето)")
    st.dataframe(summary, use_container_width=True)

    # базовая стата
    st.subheader("Статистика")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего сцен", len(df))
    c2.metric("Отфильтровано", len(df_f))
    c3.metric("Период", f"{df['date'].min().year}-{df['date'].max().year}")
    c4.metric("Медиана (все)", f"{df_f[col_area].median():.0f} га")


with tab4:
    st.header("Методология")

    st.subheader("Индексы")
    st.markdown("""
**NDBI** (Normalized Difference Built-up Index):
```
NDBI = (B11 - B08) / (B11 + B08)
```
SWIR vs NIR. Положительные значения - застройка и голая почва.
Порог > 0 - стандарт (Zha et al. 2003), > 0.05 - строже.

**NDVI** (Normalized Difference Vegetation Index):
```
NDVI = (B08 - B04) / (B08 + B04)
```
Растительность (> 0.3). Используется как маска: NDVI < 0.2 = не зелень.

**MNDWI** (Modified Normalized Difference Water Index):
```
MNDWI = (B03 - B11) / (B03 + B11)
```
Вода (> 0). Маска: MNDWI < 0 = не вода.
    """)

    st.subheader("Маска застройки")
    st.code("mask = (NDBI > порог) & (NDVI < 0.2) & (MNDWI < 0.0)", language="python")

    st.subheader("Ограничения метода")
    st.markdown("""
1. **Разрешение 20м** - один пиксель = 400 м2. Отдельные здания не различимы,
   маска показывает общую картину импервиозных поверхностей

2. **Аридный климат** - в Ташкенте сухая почва спектрально похожа на бетон.
   NDBI > 0 ловит и здания и голую землю. Разделить одним порогом нереально

3. **Погрешность ~15-20%** - разброс 100-150 га между снимками одного сезона.
   Это реальная вариативность условий съемки а не ошибка метода

4. **Тренд работает** - доля шума от голой почвы стабильна,
   а новая застройка добавляет пиксели. Рост площади реальный

Для точного картирования зданий нужен ML или данные 1-5м (Planet, WorldView).
    """)

    st.subheader("Источники")
    st.markdown("""
- Zha et al. (2003) - NDBI, *Int. J. Remote Sensing*
- Xu H. (2008) - IBI, *Int. J. Remote Sensing*
- Osgouei et al. (2019) - Istanbul, *MDPI Remote Sensing* 11(3):345
- Teshome et al. (2022) - Addis Ababa, *Environmental Challenges*
- Bhatt et al. (2021) - обзор индексов, *Arabian J. Geosciences*
    """)

    st.subheader("Территория")
    st.markdown("""
**ЖК Assalom Sohil**, Яшнабадский район, Ташкент

Полный цикл трансформации земного покрова (2019-2026):
- **до 2020**: промзона, бывший завод УзБум
- **лето 2020**: снос, котлованы (Gazeta.uz, 2 июня 2020)
- **2021+**: высотки 9-16 этажей, 13 корпусов
- **2024-2026**: новые очереди строительства
    """)
