import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing


DATA_DIR = Path("streamlit_data")
MAPS_DIR = DATA_DIR / "maps"


def load_results():
    df = pd.read_csv(DATA_DIR / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_map_files(season_key, combo_key):
    # season_key: summer/winter, combo_key: t0/t005/t0_norm/t005_norm
    # glob с [0-9] чтоб t0 не ловил t0_norm и tt005 не ловил t005_norm
    maps = sorted(MAPS_DIR.glob(f"map_{season_key}_{combo_key}_[0-9]*.png"))
    out = {}
    for p in maps:
        date = p.stem.replace(f"map_{season_key}_{combo_key}_", "")
        out[date[:4]] = (date, p)
    return out


st.set_page_config(page_title="Застройка Ташкента", layout="wide")

st.sidebar.title("анализ застройки")
st.sidebar.write("Sentinel-2, NDBI, 2019-2026")

thresh = st.sidebar.radio(
    "порог NDBI",
    ["NDBI > 0 (стандарт)", "NDBI > 0.05"],
)

normalize = st.sidebar.checkbox(
    "нормализация гистограммы",
    help="по совету препода - подгонка под эталон 2021-07-13"
)

is_t0 = "0 " in thresh
if normalize:
    col_area = "area_t0_norm" if is_t0 else "area_t005_norm"
else:
    col_area = "area_t0" if is_t0 else "area_t005"

season = st.sidebar.selectbox(
    "сезон",
    ["лето (июнь-август)", "все сезоны", "зима (дек-фев)"]
)

df = load_results()
summary = pd.read_csv(DATA_DIR / "yearly_summary.csv")

if "лето" in season:
    df_f = df[df["month"].isin([6,7,8])]
elif "зима" in season:
    df_f = df[df["month"].isin([12,1,2])]
else:
    df_f = df.copy()

df_f = df_f[df_f[col_area] > 50]

year_min = int(df_f["year"].min())
year_max = int(df_f["year"].max())
yr_range = st.sidebar.slider("годы", year_min, year_max, (year_min, year_max))
df_f = df_f[(df_f["year"] >= yr_range[0]) & (df_f["year"] <= yr_range[1])]

st.sidebar.write("---")
st.sidebar.write("ЖК Assalom Sohil")
st.sidebar.write("Яшнабадский р-н, ~3x2 км")
st.sidebar.write("[репо](https://github.com/akhm7/sd)")


tab1, tab2, tab3, tab4 = st.tabs(["динамика", "карты", "данные", "описание"])


with tab1:
    if normalize and is_t0:
        col_med, col_lo, col_hi = "median_norm", "min_norm", "max_norm"
    elif normalize:
        col_med, col_lo, col_hi = "median_norm_t005", "min_norm_t005", "max_norm_t005"
    elif is_t0:
        col_med, col_lo, col_hi = "median", "min", "max"
    else:
        col_med, col_lo, col_hi = "median_t005", "min_t005", "max_t005"

    color = "#3498db" if normalize else "#e74c3c"

    if "лето" in season:
        sm = summary[(summary["year"] >= yr_range[0]) & (summary["year"] <= yr_range[1])]

        if len(sm) >= 2:
            first = sm.iloc[0][col_med]
            last = sm.iloc[-1][col_med]
            st.write(f"рост {sm.iloc[0]['year']}→{sm.iloc[-1]['year']}: "
                     f"**{first:.0f} → {last:.0f} га** ({(last-first):+.0f} га, "
                     f"{(last-first)/first*100:+.1f}%)")

            fig = go.Figure()
            err = {}
            if col_lo is not None:
                err = {"error_y": dict(
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
                hovertemplate="%{x}: %{y:.0f} га<extra></extra>",
                **err
            ))
            fig.update_layout(
                title=f"площадь по годам ({thresh}, лето)",
                xaxis_title="год", yaxis_title="га",
                showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        by_yr = df_f.groupby("year")[col_area].agg(["median","min","max"]).reset_index()
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
                hovertemplate="%{x}: %{y:.0f} га<extra></extra>",
            ))
            fig.update_layout(
                title=f"площадь по годам ({thresh})",
                xaxis_title="год", yaxis_title="га",
                showlegend=False, height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    if len(df_f) > 5:
        df_s = df_f.sort_values("date").reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_s["date"], y=df_s[col_area],
            mode="markers+lines", name="площадь",
            marker=dict(size=5, color=color),
            line=dict(width=1, color=color),
            opacity=0.7,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.0f} га<extra></extra>",
        ))

        xn = (df_s["date"].astype(np.int64) // 10**9).astype(float).values
        z = np.polyfit(xn, df_s[col_area].values, 1)
        fig.add_trace(go.Scatter(
            x=df_s["date"], y=np.poly1d(z)(xn),
            mode="lines", name="тренд",
            line=dict(dash="dash", width=2, color="#c0392b"),
        ))

        w = min(10, len(df_s)//3)
        if w >= 3:
            sm_vals = np.convolve(df_s[col_area].values, np.ones(w)/w, mode="valid")
            off = w//2
            dates_sm = df_s["date"].values[off:off+len(sm_vals)]
            fig.add_trace(go.Scatter(
                x=dates_sm, y=sm_vals,
                mode="lines", name=f"скольз.среднее ({w})",
                line=dict(width=2.5, color="#2ecc71"),
                opacity=0.85,
            ))

        fig.update_layout(
            title=f"временной ряд ({thresh})",
            xaxis_title="год", yaxis_title="га",
            hovermode="x unified", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ============= прогноз =============
    if "лето" in season:
        sm_all = pd.read_csv(DATA_DIR / "yearly_summary.csv")
        if len(sm_all) >= 4:
            st.markdown("### прогноз на 2026-2028")
            st.caption("Holt exponential smoothing на годовых медианах")

            yrs = sm_all["year"].values
            meds = sm_all[col_med if col_med in sm_all.columns else "median"].values

            try:
                model = ExponentialSmoothing(meds, trend="add", seasonal=None).fit()
                pred = model.forecast(3)
                future_yrs = np.arange(yrs[-1]+1, yrs[-1]+4)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yrs, y=meds, mode="lines+markers",
                    name="факт", line=dict(width=2.5, color=color),
                    marker=dict(size=8),
                    hovertemplate="%{x}: %{y:.0f} га<extra></extra>",
                ))
                # пунктир от последней точки факта до первой прогноза
                fig.add_trace(go.Scatter(
                    x=[yrs[-1], future_yrs[0]], y=[meds[-1], pred[0]],
                    mode="lines", line=dict(dash="dash", width=1.5, color="#9b59b6"),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=future_yrs, y=pred, mode="lines+markers",
                    name="прогноз",
                    line=dict(dash="dash", width=2.5, color="#9b59b6"),
                    marker=dict(size=10, symbol="square"),
                    hovertemplate="%{x}: %{y:.0f} га<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="год", yaxis_title="га",
                    height=400, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.write(f"2026: **{pred[0]:.0f} га**")
                c2.write(f"2027: **{pred[1]:.0f} га**")
                c3.write(f"2028: **{pred[2]:.0f} га**")
            except Exception as e:
                st.warning(f"модель не сошлась: {e}")

    # ============= корреляция застройка vs зелень =============
    if "лето" in season and "median_veg" in summary.columns:
        st.markdown("### корреляция застройка ↔ зелень")
        st.caption("растет ли застройка за счет зеленых зон? z-нормализация чтоб масштабы совпали")

        sm_corr = summary.copy()
        bld = sm_corr[col_med].values
        veg = sm_corr["median_veg"].values

        # z-norm
        bz = (bld - bld.mean()) / bld.std()
        vz = (veg - veg.mean()) / veg.std()

        # корреляция Пирсона
        r = float(np.corrcoef(bld, veg)[0,1])

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sm_corr["year"], y=bz, mode="lines+markers",
                name="застройка (z)", line=dict(width=2, color="#e74c3c"),
            ))
            fig.add_trace(go.Scatter(
                x=sm_corr["year"], y=vz, mode="lines+markers",
                name="зелень (z)", line=dict(width=2, color="#27ae60"),
            ))
            fig.add_hline(y=0, line=dict(color="gray", width=1))
            fig.update_layout(
                title="z-нормализованные ряды",
                xaxis_title="год", yaxis_title="z-score",
                height=380, hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bld, y=veg, mode="markers+text",
                text=sm_corr["year"], textposition="top center",
                marker=dict(size=12, color="#3498db",
                           line=dict(width=1, color="black")),
                showlegend=False,
                hovertemplate="%{text}<br>застр: %{x:.0f}<br>зел: %{y:.0f}<extra></extra>",
            ))
            # линия тренда
            z = np.polyfit(bld, veg, 1)
            xs = np.linspace(bld.min(), bld.max(), 50)
            fig.add_trace(go.Scatter(
                x=xs, y=np.poly1d(z)(xs),
                mode="lines", line=dict(dash="dash", color="red"),
                name=f"r = {r:.2f}",
            ))
            fig.update_layout(
                title=f"scatter (Пирсон r = {r:.2f})",
                xaxis_title="застройка, га", yaxis_title="зелень, га",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

        sign = "отрицательная" if r < 0 else "положительная"
        strength = "сильная" if abs(r) > 0.7 else "умеренная" if abs(r) > 0.4 else "слабая"
        st.write(f"**{strength} {sign} связь** (r = {r:.2f}) - "
                 f"{'застройка растет, зелень падает' if r < -0.4 else 'связь не очень выражена'}")


with tab2:
    # карты зависят от sidebar (сезон, порог, нормализация)
    season_key = "winter" if "зима" in season else "summer"
    combo_key = ("t0" if is_t0 else "t005") + ("_norm" if normalize else "")

    map_files = get_map_files(season_key, combo_key)
    if not map_files:
        st.warning("карт нет, надо запустить precompute.py")
    else:
        years_av = sorted(map_files.keys())

        left, right = st.columns([5, 1])
        with right:
            st.caption(f"{season_key} | {thresh}{' + норм' if normalize else ''}")
            sel_year = st.radio("год", years_av,
                                index=len(years_av)-1,
                                label_visibility="collapsed")
            sel_date, sel_path = map_files[sel_year]
            row = df[df["date"] == pd.Timestamp(sel_date)]
            if len(row) > 0:
                r = row.iloc[0]
                st.write(f"**{r[col_area]:.0f} га**")
                st.caption(thresh + (" + норм" if normalize else ""))

        with left:
            st.image(str(sel_path), use_container_width=True)
            st.caption("True Color | NDBI | маска")


with tab3:
    st.write(f"всего сцен в данных: {len(df)}, после фильтра: {len(df_f)}")
    st.dataframe(
        df_f[["date","year","month",col_area,"ndbi_mean","ndbi_std"]].rename(
            columns={col_area: "площадь_га"}
        ),
        use_container_width=True, height=400
    )
    st.download_button(
        "скачать csv",
        df_f.to_csv(index=False).encode("utf-8"),
        "sentinel2_results.csv",
        "text/csv"
    )

    st.write("сводка по годам (лето):")
    st.dataframe(summary, use_container_width=True)


with tab4:
    st.write("""
короче, что тут происходит. качаю снимки Sentinel-2 со STAC (Element84)
за 2019-2026 по моему участку в ташкенте (ЖК Assalom Sohil, ~3x2 км).
считаю индексы NDBI, NDVI, MNDWI - они нормализованные разности каналов.

формулы:
""")
    st.code("""NDBI  = (B11 - B08) / (B11 + B08)   # застройка
NDVI  = (B08 - B04) / (B08 + B04)   # растительность
MNDWI = (B03 - B11) / (B03 + B11)   # вода

маска застройки = (NDBI > 0) & (NDVI < 0.2) & (MNDWI < 0)""", language="python")

    st.write("""
про пороги. NDVI > 0.3 - это активная растительность (деревья, газон),
0.1-0.3 - редкая или сухая растительность, < 0.1 - почти нет зелени.
взял **NDVI < 0.2** чтоб точно отсечь зеленые пиксели но не быть слишком
жестким - в городе много "полузеленых" дворов которые формально не лес.
MNDWI > 0 - вода, поэтому **MNDWI < 0** = не вода (стандарт).

дальше считаю площадь маски в гектарах (пиксель = 20м, площадь = 400 м2),
строю временной ряд по годам. лето беру отдельно тк зимой NDBI завышен -
голая почва зимой сухая дает похожий сигнал что и бетон.

проблема: NDBI в аридном климате путает застройку с голой почвой.
один пиксель 20x20м не показывает отдельные здания - только общую плотность.
маска получается "размазанная". это известное ограничение метода
(см. источники ниже).

после консультации с преподом добавил нормализацию гистограммы
(skimage.match_histograms) к эталонной сцене 2021-07-13. за счёт этого
разброс между датами упал с ~30 до ~8 га - сезонный шум ушёл.

почему этот участок: за 2019-2026 он прошел полный цикл - сначала промзона
(старый завод УзБум), летом 2020 снос и котлованы (Gazeta.uz писал),
с 2021 новые высотки, 13 корпусов по 9-16 этажей. идеальная история
для NDBI - старая → разрытая → новая застройка.
""")

    st.markdown("---")
    st.write("**источники:**")
    st.markdown("""
- Zha et al. (2003) - оригинальный NDBI:
  [Use of normalized difference built-up index in automatically mapping urban areas from TM imagery](https://www.tandfonline.com/doi/abs/10.1080/01431160304987)
- Xu H. (2008) - IBI (улучшенный NDBI):
  [A new index for delineating built-up land features in satellite imagery](https://www.tandfonline.com/doi/abs/10.1080/01431160802039957)
- Osgouei et al. (2019) - Стамбул, лучшая связка индексов:
  [Separating Built-Up Areas from Bare Land in Mediterranean Cities](https://www.mdpi.com/2072-4292/11/3/345)
- Teshome & Halefom (2022) - Аддис-Абеба, аридный климат как Ташкент:
  [Evaluation of NDBI for built-up area extraction](https://www.sciencedirect.com/journal/environmental-challenges)
- Bhatt et al. (2021) - обзор индексов застройки:
  [Spectral indices for extraction of built-up land: a review](https://link.springer.com/journal/12517)
- по нормализации гистограммы:
  [scikit-image match_histograms docs](https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html)
""")
