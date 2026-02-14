import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from typing import List


"""
this class handles full dataset text cleaning

goal is simple:
take raw scraped articles → normalize text → remove junk →
drop too-small docs → print dataset stats → save clean file

kept intentionally simple and readable
"""

nltk.download("stopwords")
STOP_WORDS: set[str] = set(stopwords.words("english"))


class TextPreprocessor:

    # -------- normalize and clean a single document --------
    def clean_text(self, text: str) -> str:
        """
        basic normalization — no heavy NLP here
        just enough cleanup to help vectorizers
        """

        text = text.lower()
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens: List[str] = text.split()

        # remove stopwords and tiny tokens
        tokens = [
            tok for tok in tokens
            if tok not in STOP_WORDS and len(tok) > 2
        ]

        return " ".join(tokens)

    # -------- print simple dataset statistics --------
    def dataset_stats(self, df: pd.DataFrame) -> None:
        """
        quick sanity checks before feature building
        """

        print("\n===== DATASET STATS =====")
        print("Total rows:", int(len(df)))

        print("\nLabel counts:")
        print(df["label"].value_counts())

        lengths = df["text"].str.split().str.len()
        avg_len = int(lengths.mean())

        print("\nAvg words per doc:", avg_len)
        print("=========================\n")

    # -------- full preprocessing pipeline --------
    def process_and_save(
        self,
        input_path: str = "data/dataset.csv",
        output_path: str = "data/dataset_clean.csv",
        min_words: int = 30,
    ) -> pd.DataFrame:
        """
        full flow:
        load → clean → filter → stats → save
        returns dataframe so caller can reuse it
        """

        print("Loading dataset...")
        df: pd.DataFrame = pd.read_csv(input_path)

        # drop rows where text missing
        df = df.dropna(subset=["text"]).copy()

        print("Cleaning text...")

        df["text"] = (
            df["text"]
            .astype(str)
            .apply(self.clean_text)
        )

        # filter very short documents
        lengths = df["text"].str.split().str.len()
        df = df[lengths > min_words].copy()

        self.dataset_stats(df)

        df.to_csv(output_path, index=False)

        print(f"Saved cleaned dataset → {output_path}")

        return df
