@echo off
setlocal
echo =====================================================
echo  Sentinel-2 Download — setup venv (Python 3.13)
echo =====================================================
echo.

python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python не найден в PATH!
    pause & exit /b 1
)

if not exist .venv (
    echo Создаю venv...
    python -m venv .venv
) else (
    echo .venv уже есть.
)

call .venv\Scripts\activate.bat

echo.
echo Обновляю pip...
python -m pip install --upgrade pip --quiet

echo.
echo Ставлю пакеты...

:: rasterio на Windows 3.13 — с версии 1.4.x есть готовые wheels
pip install rasterio
if errorlevel 1 (
    echo.
    echo [!] rasterio не встал через pip.
    echo     Попробуй: conda install -c conda-forge rasterio
    echo     Или скачай wheel с https://github.com/cgohlke/rasterio-builds/releases
    echo.
    pause & exit /b 1
)

pip install pystac-client shapely requests tqdm numpy python-dateutil

echo.
echo =====================================================
echo  Проверка:
echo =====================================================
python -c "import rasterio; print('rasterio   ', rasterio.__version__)"
python -c "import pystac_client; print('pystac_client  OK')"
python -c "import shapely; print('shapely    ', shapely.__version__)"
python -c "import tqdm; print('tqdm       ', tqdm.__version__)"

echo.
echo =====================================================
echo  Всё готово. Запускай:
echo    .venv\Scripts\activate
echo    python download_sentinel2.py
echo =====================================================
echo.
pause
