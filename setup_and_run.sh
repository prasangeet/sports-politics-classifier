#!/bin/bash

# Sports vs Politics Classifier - Setup and Execution Script
# This script sets up the environment and runs the complete pipeline

set -e  # Exit on error

echo "=========================================="
echo "Sports vs Politics Classifier Setup"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Step 1: Create virtual environment
echo "📦 Step 1: Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Step 2: Activate virtual environment
echo "🔧 Step 2: Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Step 3: Install dependencies
echo "📥 Step 3: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Step 4: Download NLTK data (required for preprocessing)
echo "📚 Step 4: Downloading NLTK stopwords..."
python3 -c "import nltk; nltk.download('stopwords', quiet=True)"
echo "✅ NLTK data downloaded"
echo ""

# Step 5: Collect data
echo "📡 Step 5: Collecting data from RSS feeds..."
if [ ! -f "data/dataset.csv" ]; then
    python3 src/collect_data.py
    echo "✅ Data collection complete"
else
    echo "ℹ️  Dataset already exists. Skipping collection."
    read -p "Do you want to re-collect data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 src/collect_data.py
        echo "✅ Data re-collected"
    fi
fi
echo ""

# Step 6: Run the full pipeline
echo "🚀 Step 6: Running the full ML pipeline..."
echo "This will:"
echo "  - Preprocess the text data"
echo "  - Build BoW and TF-IDF features"
echo "  - Train Naive Bayes, Logistic Regression, and SVM models"
echo ""
python3 main.py
echo ""

echo "=========================================="
echo "✅ Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "📊 Results have been saved to:"
echo "  - Models: models/*.pkl"
echo "  - Logs: logs/pipeline.log"
echo ""
echo "To view the log file:"
echo "  cat logs/pipeline.log"
echo ""
