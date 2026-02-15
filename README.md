# Sports vs. Politics: A Text Classification Project

Welcome! This project is a simple yet effective exploration into Natural Language Processing (NLP). The goal was to build a system capable of reading a text document and deciding whether it's about **Sports** or **Politics**.

It's fascinating how distinct the language is between these two worlds, and this project uses machine learning to quantify those differences.

## 🌟 Project Overview

This classifier was built to compare different machine learning techniques on a real-world dataset collected from RSS feeds. We explored how different ways of representing text (like counting words vs. measuring their uniqueness) impact the ability of a computer to distinguish between a game recap and a political debate.

### Key Features
- **Real-world Data**: Articles scraped from BBC, ESPN, and NYT.
- **Text Preprocessing**: Cleaned raw text to remove noise while preserving signal.
- **Multiple Models**: Compared Naive Bayes, Logistic Regression, and SVM.
- **Feature Analysis**: Evaluated Bag of Words (BoW) and TF-IDF representations.

## 🚀 How to Run

If you'd like to run this analysis yourself, follow these steps:

### Option 1: Automated Setup (Recommended)

Simply run the setup script which handles everything automatically:

**For Linux/macOS:**
```bash
git clone https://github.com/yourusername/sports-politics-classifier.git
cd sports-politics-classifier
./setup_and_run.sh
```

**For Windows:**
```cmd
git clone https://github.com/yourusername/sports-politics-classifier.git
cd sports-politics-classifier
setup_and_run.bat
```

This script will:
- Create and activate a virtual environment
- Install all dependencies
- Download NLTK data
- Collect data from RSS feeds
- Run the complete ML pipeline

### Option 2: Manual Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sports-politics-classifier.git
    cd sports-politics-classifier
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  
    # or on fish shell
    source .venv/bin/activate.fish
    # On Windows use `venv\Scripts\activate`
    ```
3. **Download the dataset**
    ```bash
    python src/collect_data.py
    ```

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the full pipeline**:
    This will collect data (if not already present), preprocess it, build features, and train the models.
    ```bash
    python main.py
    ```


## 📊 Methodology & Results

We tested three different classification algorithms using two different feature extraction methods. Here is a summary of our findings.

### Data Collection
We collected **372 articles** (213 Sports, 159 Politics) using RSS feeds from major news outlets. The data is available in `data/dataset.csv`.

### Performance Comparison (Accuracy)

| Model | Bag of Words (BoW) | TF-IDF (1-2 grams) |
| :--- | :---: | :---: |
| **Naive Bayes** | 0.97 | 0.96 |
| **Logistic Regression** | **0.99** | 0.97 |
| **Support Vector Machine (SVM)** | **0.99** | 0.97 |

### Analysis
- **Logistic Regression & SVM with Bag of Words** achieved the highest accuracy of **98.67%**. This demonstrates that for these categories, simple word counts provide enough signal for linear separators to work nearly perfectly.
- **Improved Dataset Stability**: With 372 articles, the TF-IDF representation became much more stable compared to smaller initial runs, showing high accuracy (96-97%) across all models.
- **Lexical Distinction**: The vocabulary remains the key driver of performance—distinct terms like "ballot" and "touchdown" allow for very clean class separation.

## 📁 Project Structure

- `src/collect_data.py`: Scripts to scrape RSS feeds.
- `src/preprocess.py`: Text cleaning and normalization.
- `src/features.py`: Generating BoW and TF-IDF features.
- `src/train_*.py`: Training scripts for each model.
- `main.py`: The orchestrator that runs the entire pipeline.
- `notebooks/`: Jupyter notebooks for exploratory analysis.

## 📝 Limitations

- **Dataset Size**: While expanded to ~370 articles, real-world generalization would benefit from thousands of documents.
- **Class Imbalance**: There is a slight bias towards sports articles (57%), though stratified splitting helps mitigate this in testing.

---
**Author**: Prasangeet Dongre (B23CH1033)  
*Created for the NLU Assignment.*
