import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)


"""
this module builds numeric text features

we generate two representations:
bag of words and tfidf ngrams

both use the same train/test split so
all models compare fairly later
"""

RANDOM_STATE = 42


class DatasetSplitLoader:

    # -------- load cleaned dataset and split --------
    def load_and_split(
        self,
        path: str = "data/dataset_clean.csv",
        test_size: float = 0.2,
    ):
        """
        loads cleaned dataset and performs stratified split
        returns raw text splits (not vectorized yet)
        """

        print("Loading cleaned dataset...")
        df: pd.DataFrame = pd.read_csv(path)

        X = df["text"].astype(str)
        y = df["label"].astype(str)

        print("Splitting train/test...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        return X_train, X_test, y_train, y_test


class BowFeatureBuilder:

    """
    builds bag-of-words features

    simple frequency counts
    good baseline representation
    """

    def __init__(self, max_features: int = 40000):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
        )

    # -------- build and save bow features --------
    def build_and_save(self):
        loader = DatasetSplitLoader()
        X_train, X_test, y_train, y_test = loader.load_and_split()

        print("\nBuilding Bag of Words features...")

        X_train_bow = self.vectorizer.fit_transform(X_train)
        X_test_bow = self.vectorizer.transform(X_test)

        print("BoW vocab size:", len(self.vectorizer.vocabulary_))
        print("BoW train shape:", X_train_bow.shape)

        # save vectorizer and matrices
        joblib.dump(
            self.vectorizer,
            "models/vectorizer_bow.pkl",
        )

        joblib.dump(
            (X_train_bow, X_test_bow, y_train, y_test),
            "models/features_bow.pkl",
        )

        print("BoW features saved")

        return X_train_bow, X_test_bow, y_train, y_test


class TfidfFeatureBuilder:

    """
    builds tfidf ngram features

    includes bigrams because phrases
    carry strong topic signals
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

    # -------- build and save tfidf features --------
    def build_and_save(self):
        loader = DatasetSplitLoader()
        X_train, X_test, y_train, y_test = loader.load_and_split()

        print("\nBuilding TFIDF (1–2 gram) features...")

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        print("TFIDF vocab size:", len(self.vectorizer.vocabulary_))
        print("TFIDF train shape:", X_train_tfidf.shape)

        # save vectorizer and matrices
        joblib.dump(
            self.vectorizer,
            "models/vectorizer_tfidf.pkl",
        )

        joblib.dump(
            (X_train_tfidf, X_test_tfidf, y_train, y_test),
            "models/features_tfidf.pkl",
        )

        print("TFIDF features saved")

        return X_train_tfidf, X_test_tfidf, y_train, y_test
