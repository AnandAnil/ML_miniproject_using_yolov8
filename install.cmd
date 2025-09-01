@echo off
REM YOLO Drowsiness Detection Project - Installation Script for Windows
REM This script creates a virtual environment and installs all required dependencies

setlocal enabledelayedexpansion

echo YOLO Drowsiness Detection - Installation Script (Windows)
echo ==============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python found: !PYTHON_VERSION!

REM Check if pip is installed
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed. Please install pip first.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -m pip --version 2^>^&1') do set PIP_VERSION=%%i
echo pip found: !PIP_VERSION!

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q .venv
)

python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)
echo Virtual environment created successfully!

REM Activate virtual environment
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing required packages...
echo    This may take a few minutes depending on your internet connection...

if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install packages from requirements.txt
        pause
        exit /b 1
    )
    echo All packages installed successfully!
) else (
    echo requirements.txt not found!
    pause
    exit /b 1
)

REM Verify installation
echo.
echo Verifying installation...
python -c "import sys; print(f'Python version: {sys.version}'); print(f'Virtual environment: {sys.prefix}'); import ultralytics, cv2, sklearn, matplotlib, serial; print('All main libraries imported successfully!'); print(''); print('Installed package versions:'); print(f'   ultralytics: {ultralytics.__version__}'); print(f'   opencv-python: {cv2.__version__}'); print(f'   scikit-learn: {sklearn.__version__}'); print(f'   matplotlib: {matplotlib.__version__}'); print(f'   pyserial: {serial.VERSION}')"

if errorlevel 1 (
    echo Package verification failed.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo ==============================================================
echo.
echo Usage Instructions:
echo    1. Activate the virtual environment:
echo       .venv\Scripts\activate.bat
echo.
echo    2. Run your Python scripts:
echo       python "1) split_dataset.py"
echo       python "2) train_yolov8_drowsiness.py"
echo       python "3) confusion_matrix.py"
echo       python "4) test_realtime_detection.py"
echo       python serialtest.py
echo.
echo    3. Deactivate when done:
echo       deactivate
echo.
echo Troubleshooting:
echo    - If you get permission errors, run as Administrator
echo    - If CUDA is not detected, PyTorch will use CPU (this is normal)
echo    - For ESP32 connection issues, check USB drivers and COM port access
echo    - Make sure Python is added to your system PATH
echo.
echo Happy coding!
echo.
pause
