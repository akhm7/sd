# анализ застройки ташкента

Sentinel-2, NDBI, период 2019-2026.
участок - ЖК Assalom Sohil и окрестности (Яшнабадский р-н), ~3x2 км.
спецкурс "методы анализа спутниковых данных", защита май 2026.

## что внутри

- `analysis.ipynb` - юпитер с пошаговым анализом по этапам
- `streamlit_app.py` - веб-интерфейс, на share.streamlit.io
- `utils.py` - функции (загрузка каналов, NDBI/NDVI/MNDWI, маски, нормализация)
- `precompute.py` - готовит csv и карты для стримлита
- `download_sentinel2.py` - качает .tif со STAC (Element84)
- `streamlit_data/` - то что читает стримлит (csv + png)
- `output/` - картинки которые генерит ноутбук (в gitignore)

## как запустить локально

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python download_sentinel2.py   # качаем данные (~482 сцены, занимает минут 30)
```

потом или ноутбук:
```
jupyter notebook analysis.ipynb
```

или стримлит:
```
python precompute.py           # один раз, готовит csv + карты
streamlit run streamlit_app.py
```

## методы кратко

```
NDBI  = (B11 - B08) / (B11 + B08)   # застройка
NDVI  = (B08 - B04) / (B08 + B04)   # растительность
MNDWI = (B03 - B11) / (B03 + B11)   # вода

маска = (NDBI > 0) & (NDVI < 0.2) & (MNDWI < 0)
```

лето берем отдельно тк зимой NDBI завышен (голая почва дает похожий
сигнал что и бетон в аридном климате).

после консультации добавил нормализацию гистограммы через
`skimage.match_histograms` к эталонной сцене 2021-07-13 - разброс
между датами упал с ~30 до ~8 га.

прогноз на 2026-2028 через `statsmodels.ExponentialSmoothing` (Holt).
корреляция застройки и зелени через коэф Пирсона + z-нормализация.

## данные

Sentinel-2 L2A через STAC API Element84 Earth Search.
~3800 .tif файлов (8 каналов x 482 сцены), облачность < 15%.
COG-режим - скачиваем только нужный кусок прямо с S3.

## деплой

https://tashkent-ndbi.streamlit.app

## источники

индексы NDBI/IBI/BUI:
- Zha et al. 2003 - NDBI, Int. J. Remote Sensing
- Xu H. 2008 - IBI, Int. J. Remote Sensing
- Osgouei et al. 2019 - Istanbul, MDPI Remote Sensing 11(3):345
- Teshome et al. 2022 - Addis Ababa (аридный климат), Environmental Challenges

про участок:
- Gazeta.uz, 2 июня 2020 (снос и котлованы)
- Gazeta.uz, 18 августа 2021 (ЖК сформирован)
- Yangiuylar.uz, Domtut.uz (очереди до 2026)
