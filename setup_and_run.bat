@echo off
REM Sports vs Politics Classifier - Windows Setup and Execution Script
REM This script sets up the environment and runs the complete pipeline

echo ==========================================
echo Sports vs Politics Classifier Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Step 1: Create virtual environment
echo 📦 Step 1: Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo ✅ Virtual environment created
) else (
    echo ℹ️  Virtual environment already exists
)
echo.

REM Step 2: Activate virtual environment
echo 🔧 Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✅ Virtual environment activated
echo.

REM Step 3: Install dependencies
echo 📥 Step 3: Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ✅ Dependencies installed
echo.

REM Step 4: Download NLTK data
echo 📚 Step 4: Downloading NLTK stopwords...
python -c "import nltk; nltk.download('stopwords', quiet=True)"
echo ✅ NLTK data downloaded
echo.

REM Step 5: Collect data
echo 📡 Step 5: Collecting data from RSS feeds...
if not exist "data\dataset.csv" (
    python src\collect_data.py
    echo ✅ Data collection complete
) else (
    echo ℹ️  Dataset already exists. Skipping collection.
    set /p REPLY="Do you want to re-collect data? (y/n): "
    if /i "%REPLY%"=="y" (
        python src\collect_data.py
        echo ✅ Data re-collected
    )
)
echo.

REM Step 6: Run the full pipeline
echo 🚀 Step 6: Running the full ML pipeline...
echo This will:
echo   - Preprocess the text data
echo   - Build BoW and TF-IDF features
echo   - Train Naive Bayes, Logistic Regression, and SVM models
echo.
python main.py
echo.

echo ==========================================
echo ✅ Pipeline completed successfully!
echo ==========================================
echo.
echo 📊 Results have been saved to:
echo   - Models: models\*.pkl
echo   - Logs: logs\pipeline.log
echo.
echo To view the log file:
echo   type logs\pipeline.log
echo.
pause
