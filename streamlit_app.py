import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

normalize = st.sidebar.checkbox(
    "Нормализация гистограммы",
    value=False,
    help="по совету препода - подгонка снимков под эталон 2021-07-13 чтоб убрать атмосферный шум"
)

if normalize:
    col_area = "area_t0_norm"
else:
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
    if normalize:
        col_med = "median_norm"
        col_lo, col_hi = "min_norm", "max_norm"
    elif "0 " in thresh:
        col_med = "median"
        col_lo, col_hi = "min", "max"
    else:
        col_med = "median_t005"
        col_lo, col_hi = None, None

    color = "#3498db" if normalize else "#e74c3c"

    if "Лето" in season:
        sm = summary[(summary["year"] >= yr_range[0]) & (summary["year"] <= yr_range[1])]

        if len(sm) >= 2:
            c1, c2, c3 = st.columns(3)
            first = sm.iloc[0][col_med]
            last = sm.iloc[-1][col_med]
            c1.metric("Старт", f"{first:.0f} га", delta=None)
            c2.metric("Финиш", f"{last:.0f} га", delta=f"{last-first:+.0f} га")
            c3.metric("Рост", f"{(last-first)/first*100:+.1f}%")

            # бар чарт через plotly
            fig = go.Figure()
            err_args = {}
            if col_lo is not None:
                err_args = {"error_y": dict(
                    type="data", symmetric=False,
                    array=sm[col_hi]-sm[col_med],
                    arrayminus=sm[col_med]-sm[col_lo],
                    color="black", thickness=1.5, width=6,
                )}
            fig.add_trace(go.Bar(
                x=sm["year"], y=sm[col_med],
                marker_color=color, opacity=0.85,
                text=[f"{m:.0f}" for m in sm[col_med]],
                textposition="outside",
                hovertemplate="год %{x}<br>%{y:.0f} га<extra></extra>",
                **err_args
            ))
            fig.update_layout(
                title=f"застройка по годам ({thresh}, лето)",
                xaxis_title="год", yaxis_title="площадь, га",
                showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        by_yr = df_f.groupby("year")[col_area].agg(["median","min","max","count"]).reset_index()

        if len(by_yr) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=by_yr["year"], y=by_yr["median"],
                marker_color=color, opacity=0.85,
                error_y=dict(
                    type="data", symmetric=False,
                    array=by_yr["max"]-by_yr["median"],
                    arrayminus=by_yr["median"]-by_yr["min"],
                    color="black", thickness=1.5, width=6,
                ),
                text=[f"{m:.0f}" for m in by_yr["median"]],
                textposition="outside",
                hovertemplate="год %{x}<br>%{y:.0f} га<extra></extra>",
            ))
            fig.update_layout(
                title=f"застройка по годам ({thresh})",
                xaxis_title="год", yaxis_title="площадь, га",
                showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    if len(df_f) > 5:
        st.subheader("Временной ряд")
        df_s = df_f.sort_values("date").reset_index(drop=True)

        fig = go.Figure()
        # точки
        fig.add_trace(go.Scatter(
            x=df_s["date"], y=df_s[col_area],
            mode="markers+lines",
            name="площадь",
            marker=dict(size=5, color=color),
            line=dict(width=1, color=color),
            opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f} га<extra></extra>",
        ))

        # тренд
        xn = (df_s["date"].astype(np.int64) // 10**9).astype(float).values
        z = np.polyfit(xn, df_s[col_area].values, 1)
        fig.add_trace(go.Scatter(
            x=df_s["date"], y=np.poly1d(z)(xn),
            mode="lines", name="тренд",
            line=dict(dash="dash", width=2, color="#c0392b"),
        ))

        # скользящее среднее
        w = min(10, len(df_s)//3)
        if w >= 3:
            sm_vals = np.convolve(df_s[col_area].values, np.ones(w)/w, mode="valid")
            off = w//2
            dates_sm = df_s["date"].values[off:off+len(sm_vals)]
            fig.add_trace(go.Scatter(
                x=dates_sm, y=sm_vals,
                mode="lines",
                name=f"скольз.среднее ({w})",
                line=dict(width=2.5, color="#2ecc71"),
                opacity=0.85,
            ))

        fig.update_layout(
            title=f"динамика застройки ({thresh})",
            xaxis_title="год", yaxis_title="площадь, га",
            hovermode="x unified", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)


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
