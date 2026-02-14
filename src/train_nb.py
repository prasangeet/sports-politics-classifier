import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


"""
this class trains multinomial naive bayes models
on both BoW and TFIDF features

naive bayes is our fast baseline model
usually performs surprisingly well for text
"""


class NaiveBayesTrainer:

    # -------- evaluation helper --------
    def evaluate(self, name, model, X_test, y_test):
        """
        prints accuracy, report and confusion matrix
        keeping same format across all models
        """

        preds = model.predict(X_test)

        print(f"\n===== {name} =====")
        print("Accuracy:", accuracy_score(y_test, preds))

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    # -------- load precomputed features --------
    def load_features(self):
        """
        loads saved sparse matrices from disk
        avoids recomputing vectorizers each run
        """

        print("Loading features...")

        bow_pack = joblib.load("models/features_bow.pkl")
        tfidf_pack = joblib.load("models/features_tfidf.pkl")

        X_train_bow, X_test_bow, y_train, y_test = bow_pack
        X_train_tfidf, X_test_tfidf, _, _ = tfidf_pack

        return (
            X_train_bow,
            X_test_bow,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
        )

    # -------- train both variants --------
    def train_and_save(self):
        """
        trains two naive bayes models:
        one with BoW and one with TFIDF
        then saves both
        """

        (
            X_train_bow,
            X_test_bow,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test,
        ) = self.load_features()

        # -------- BoW model --------
        print("\nTraining Naive Bayes (BoW)...")
        nb_bow = MultinomialNB()
        nb_bow.fit(X_train_bow, y_train)

        self.evaluate(
            "NaiveBayes + BoW",
            nb_bow,
            X_test_bow,
            y_test
        )

        # -------- TFIDF model --------
        print("\nTraining Naive Bayes (TFIDF)...")
        nb_tfidf = MultinomialNB()
        nb_tfidf.fit(X_train_tfidf, y_train)

        self.evaluate(
            "NaiveBayes + TFIDF",
            nb_tfidf,
            X_test_tfidf,
            y_test
        )

        # -------- save models --------
        joblib.dump(
            nb_bow,
            "models/model_nb_bow.pkl"
        )

        joblib.dump(
            nb_tfidf,
            "models/model_nb_tfidf.pkl"
        )

        print("\nNaive Bayes models saved")
