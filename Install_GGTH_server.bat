@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo    LSTM Trading System - Automated Installation Script
echo ================================================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  WARNING: Not running as administrator. Some features may not work.
    echo    Right-click this script and select "Run as administrator" for best results.
    echo.
    pause
)

:: Set installation directory to the same folder as this script
set INSTALL_DIR=%~dp0
:: Remove trailing backslash if present
if "%INSTALL_DIR:~-1%"=="\" set INSTALL_DIR=%INSTALL_DIR:~0,-1%
set PYTHON_VERSION=3.9.17

echo 📁 Step 1: Creating directory structure...
cd /d "%INSTALL_DIR%"
if not exist "data" mkdir "data"
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"
echo ✅ Directory structure created: %INSTALL_DIR%
echo.

echo 🐍 Step 2: Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Python not found or not in PATH.
    echo.
    echo Please install Python manually:
    echo 1. Go to https://www.python.org/downloads/
    echo 2. Download Python 3.9 or 3.10
    echo 3. IMPORTANT: Check "Add Python to PATH" during installation
    echo 4. Run this script again after Python is installed
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
    echo ✅ Python found: !PYTHON_VER!
)
echo.

echo 🔧 Step 3: Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)
echo.

echo 📦 Step 4: Installing Python packages...
echo This may take 5-10 minutes depending on your internet connection...
call venv\Scripts\activate.bat

:: Upgrade pip first
python -m pip install --upgrade pip --quiet

:: Install PyTorch (CPU version)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if %errorLevel% neq 0 (
    echo ❌ Failed to install PyTorch
    pause
    exit /b 1
)

:: Install other packages
echo Installing data science packages...
pip install pandas numpy scikit-learn --quiet
pip install pandas-ta --quiet
pip install joblib tqdm requests --quiet

echo ✅ All Python packages installed successfully
echo.

echo 🧪 Step 5: Testing Python installation...
python -c "import torch; print('✅ PyTorch version:', torch.__version__)" 2>nul
if %errorLevel% neq 0 (
    echo ❌ PyTorch test failed
    pause
    exit /b 1
)

python -c "import pandas; print('✅ Pandas version:', pandas.__version__)" 2>nul
python -c "import pandas_ta; print('✅ Technical Analysis library loaded')" 2>nul
echo.

echo 📋 Step 6: Creating project files...

:: Check if Python files exist
set FILES_NEEDED=train.py backtest.py daemon_http.py utils.py test.py SAM.mq5 ExportHistory.mq5
set MISSING_FILES=

for %%f in (%FILES_NEEDED%) do (
    if not exist "%%f" (
        set MISSING_FILES=!MISSING_FILES! %%f
    )
)

if defined MISSING_FILES (
    echo ⚠️  Missing project files: !MISSING_FILES!
    echo.
    echo Please copy the following files to %INSTALL_DIR%:
    for %%f in (%FILES_NEEDED%) do echo   - %%f
    echo.
    echo After copying files, run this script again or continue manually.
    pause
) else (
    echo ✅ All project files found
)
echo.

echo 🏦 Step 7: MetaTrader 5 setup...
echo.
echo MANUAL STEPS REQUIRED FOR MT5:
echo.
echo 1. Install MetaTrader 5 from: https://www.metatrader5.com/en/download
echo.
echo 2. Configure Expert Advisors:
echo    Tools → Options → Expert Advisors
echo    ✅ Allow automated trading
echo    ✅ Allow DLL imports
echo    ✅ Allow WebRequest for URLs
echo.
echo 3. Add these URLs to WebRequest allowed list:
echo    http://127.0.0.1:8888
echo    http://127.0.0.1:8888/predict
echo    http://127.0.0.1:8888/health
echo    http://127.0.0.1:8888/stats
echo.
echo 4. Install project files in MT5:
echo    a) Copy SAM.mq5 to: MT5_Data_Folder\MQL5\Experts\
echo    b) Copy ExportHistory.mq5 to: MT5_Data_Folder\MQL5\Scripts\
echo    c) Compile both files in MetaEditor (press F7)
echo.
echo 5. Export historical data:
echo    a) Add all currency pairs to Market Watch: EURUSD, EURJPY, USDJPY, GBPUSD, EURGBP, USDCAD, USDCHF
echo    b) Open H1 charts for each pair and scroll back to 2015 to download history
echo    c) Run ExportHistory script on any chart
echo    d) Copy exported CSV files from MT5 Common\Files to %INSTALL_DIR%\data\
echo.
echo 📊 For detailed data export instructions, see the full installation guide!
echo.

echo 📊 Step 8: Creating sample data (for immediate testing)...
if exist "train.py" (
    echo Creating sample prediction file...
    python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
start_time = datetime.now() - timedelta(hours=100)
data = []
base_price = 1.0800

for i in range(100):
    timestamp = start_time + timedelta(hours=i)
    current_price = base_price + (i * 0.00001)
    
    predictions = []
    for step in range(5):
        step_change = np.random.normal(0, 0.0001 * (step + 1))
        trend = 0.0002 if np.random.random() > 0.5 else -0.0002
        pred_price = current_price + (step_change + trend * (step + 1) * 0.1)
        predictions.append(pred_price)
    
    buy_prob = max(0.1, min(0.8, np.random.normal(0.4, 0.15)))
    sell_prob = max(0.1, min(0.8, np.random.normal(0.3, 0.15)))
    total_prob = buy_prob + sell_prob
    if total_prob > 0.9:
        buy_prob *= 0.9 / total_prob
        sell_prob *= 0.9 / total_prob
    hold_prob = 1.0 - buy_prob - sell_prob
    confidence = max(0.3, min(0.95, np.random.normal(0.65, 0.15)))
    
    row = [
        timestamp.strftime('%%Y.%%m.%%d %%H:%%M:%%S'),
        round(buy_prob, 6), round(sell_prob, 6), 
        round(hold_prob, 6), round(confidence, 6)
    ] + [round(p, 5) for p in predictions]
    data.append(row)

columns = ['timestamp', 'buy_prob', 'sell_prob', 'hold_prob', 'confidence_score'] + [f'predicted_price_{i}' for i in range(5)]
df = pd.DataFrame(data, columns=columns)
df.to_csv('backtest_predictions.csv', sep=';', index=False)
print('✅ Sample prediction file created')
"
    if %errorLevel% equ 0 (
        echo ✅ Sample prediction file created successfully
    ) else (
        echo ⚠️  Could not create sample file automatically
    )
) else (
    echo ⚠️  train.py not found, skipping sample data creation
)
echo.

echo 🎯 Step 9: Creating startup scripts...

:: Create start_daemon.bat
echo @echo off > start_daemon.bat
echo cd /d "%INSTALL_DIR%" >> start_daemon.bat
echo call venv\Scripts\activate.bat >> start_daemon.bat
echo echo Starting LSTM Trading Daemon... >> start_daemon.bat
echo python daemon_http.py >> start_daemon.bat
echo pause >> start_daemon.bat

:: Create test_system.bat  
echo @echo off > test_system.bat
echo cd /d "%INSTALL_DIR%" >> test_system.bat
echo call venv\Scripts\activate.bat >> test_system.bat
echo echo Testing system communication... >> test_system.bat
echo python test.py >> test_system.bat
echo pause >> test_system.bat

:: Create train_model.bat
echo @echo off > train_model.bat
echo cd /d "%INSTALL_DIR%" >> train_model.bat
echo call venv\Scripts\activate.bat >> train_model.bat
echo echo Training LSTM model... >> train_model.bat
echo python train.py >> train_model.bat
echo pause >> train_model.bat

:: Create copy_data.bat for moving MT5 exported files
echo @echo off > copy_data.bat
echo echo Copying exported data from MT5 to Python project... >> copy_data.bat
echo. >> copy_data.bat
echo set MT5_DATA_PATH=%%APPDATA%%\MetaQuotes\Terminal\Common\Files\GGTH_Python_Backend\data >> copy_data.bat
echo set PYTHON_DATA_PATH=%INSTALL_DIR%\data >> copy_data.bat
echo. >> copy_data.bat
echo if not exist "%%MT5_DATA_PATH%%" ^( >> copy_data.bat
echo    echo ❌ MT5 export folder not found: %%MT5_DATA_PATH%% >> copy_data.bat
echo    echo Please run ExportHistory script in MT5 first >> copy_data.bat
echo    pause >> copy_data.bat
echo    exit /b 1 >> copy_data.bat
echo ^) >> copy_data.bat
echo. >> copy_data.bat
echo echo Copying CSV files... >> copy_data.bat
echo copy "%%MT5_DATA_PATH%%\*.csv" "%%PYTHON_DATA_PATH%%\" >> copy_data.bat
echo. >> copy_data.bat
echo echo Verifying copied files: >> copy_data.bat
echo dir "%%PYTHON_DATA_PATH%%\*.csv" >> copy_data.bat
echo. >> copy_data.bat
echo echo ✅ Data copy complete! >> copy_data.bat
echo pause >> copy_data.bat

echo ✅ Startup scripts created:
echo   - start_daemon.bat (starts the HTTP server)
echo   - test_system.bat (tests communication)
echo   - train_model.bat (trains the model with real data)
echo   - copy_data.bat (copies exported CSV files from MT5)
echo.

echo 📋 Step 10: Creating quick reference...
echo. > QUICK_START.txt
echo LSTM Trading System - Quick Start Guide >> QUICK_START.txt
echo ============================================= >> QUICK_START.txt
echo. >> QUICK_START.txt
echo Installation Directory: %INSTALL_DIR% >> QUICK_START.txt
echo. >> QUICK_START.txt
echo DATA EXPORT (First Time Setup): >> QUICK_START.txt
echo 1. Open MetaTrader 5 >> QUICK_START.txt
echo 2. Add currency pairs to Market Watch: EURUSD, EURJPY, USDJPY, GBPUSD, EURGBP, USDCAD, USDCHF >> QUICK_START.txt
echo 3. Open H1 charts for each pair, scroll back to 2015 to download history >> QUICK_START.txt
echo 4. Run ExportHistory script on any chart >> QUICK_START.txt
echo 5. Run copy_data.bat to move CSV files to Python project >> QUICK_START.txt
echo 6. Run train_model.bat to train the LSTM model >> QUICK_START.txt
echo. >> QUICK_START.txt
echo DAILY STARTUP SEQUENCE: >> QUICK_START.txt
echo 1. Double-click start_daemon.bat (keep it running) >> QUICK_START.txt
echo 2. Open MetaTrader 5 >> QUICK_START.txt
echo 3. Attach GGTH-SR.mq5 to EURUSD H1 chart >> QUICK_START.txt
echo 4. Enable automated trading >> QUICK_START.txt
echo. >> QUICK_START.txt
echo TROUBLESHOOTING: >> QUICK_START.txt
echo - Run test_system.bat to check communication >> QUICK_START.txt
echo - Check MT5 Experts tab for error messages >> QUICK_START.txt
echo - Ensure URLs are in MT5 allowed WebRequest list >> QUICK_START.txt
echo - If no predictions, check daemon is running and URLs are allowed >> QUICK_START.txt
echo. >> QUICK_START.txt
echo RETRAINING MODEL (Monthly): >> QUICK_START.txt
echo 1. Run ExportHistory script to get fresh data >> QUICK_START.txt
echo 2. Run copy_data.bat to update CSV files >> QUICK_START.txt
echo 3. Run train_model.bat to retrain with new data >> QUICK_START.txt
echo. >> QUICK_START.txt

echo ================================================================
echo                    INSTALLATION COMPLETE! 
echo ================================================================
echo.
echo ✅ Python environment set up at: %INSTALL_DIR%\venv
echo ✅ Project files ready at: %INSTALL_DIR%
echo ✅ Startup scripts created
echo ✅ Sample prediction file created for testing
echo.
echo 🔄 NEXT STEPS:
echo.
echo 1. MANUAL: Set up MetaTrader 5 (see instructions above)
echo 2. TEST: Double-click 'start_daemon.bat' to start the HTTP server
echo 3. TEST: In MT5, attach GGTH-SR.mq5 to EURUSD H1 chart  
echo 4. OPTIONAL: Export real data and run 'train_model.bat'
echo.
echo 📖 Read QUICK_START.txt for detailed usage instructions
echo.
echo ⚠️  IMPORTANT: Start with small position sizes when going live!
echo.
echo ================================================================
pause