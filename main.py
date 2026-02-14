from src.preprocess import TextPreprocessor
from src.features import BowFeatureBuilder, TfidfFeatureBuilder
from src.train_nb import NaiveBayesTrainer
from src.train_logreg import LogisticRegressionTrainer
from src.train_svm import SVMTrainer

import sys
from pathlib import Path


"""
this file runs the whole project pipeline

now it also writes full execution logs to a file
so model outputs and metrics are saved automatically
useful later for report tables and comparisons
"""


class TeeLogger:
    """
    simple stdout duplicator

    whatever gets printed will go to both
    terminal and log file
    """

    def __init__(self, logfile_path: str):
        Path("logs").mkdir(exist_ok=True)
        self.file = open(logfile_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


class ProjectRunner:

    # -------- run text preprocessing --------
    def run_preprocessing(self):
        """
        cleans raw dataset and writes cleaned csv
        """

        print("\nStarting preprocessing step...")
        pre = TextPreprocessor()
        pre.process_and_save()

    # -------- build both feature types --------
    def run_feature_building(self):
        """
        generates BoW and TFIDF features
        saved to disk for reuse by all models
        """

        print("\nStarting feature building step...")

        BowFeatureBuilder().build_and_save()
        TfidfFeatureBuilder().build_and_save()

    # -------- train all classifiers --------
    def run_models(self):
        """
        trains all three classifiers
        each prints metrics into the log
        """

        print("\nStarting model training step...")

        NaiveBayesTrainer().train_and_save()
        LogisticRegressionTrainer().train_and_save()
        SVMTrainer().train_and_save()

    # -------- full pipeline runner --------
    def run_all(self):
        """
        executes full project pipeline
        """

        print("\nPipeline started\n")

        self.run_preprocessing()
        self.run_feature_building()
        self.run_models()

        print("\nPipeline completed successfully")


# -------- entry trigger with logging --------

if __name__ == "__main__":

    sys.stdout = TeeLogger("logs/pipeline.log")

    runner = ProjectRunner()
    runner.run_all()
