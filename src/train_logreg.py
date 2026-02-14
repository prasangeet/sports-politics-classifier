import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


"""
this class trains and evaluates logistic regression models
on both BoW and TFIDF features

kept separate so later we can call it from a global
comparison runner and collect metrics in one place
"""


class LogisticRegressionTrainer:

    # -------- evaluate predictions in a readable way --------
    def evaluate(self, name, model, X_test, y_test):
        """
        prints standard classification metrics
        same format used across all models
        """

        preds = model.predict(X_test)

        print(f"\n===== {name} =====")
        print("Accuracy:", accuracy_score(y_test, preds))

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    # -------- load saved sparse feature matrices --------
    def load_features(self):
        """
        loads prebuilt features so we don't recompute
        vectorization every time we train a model
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

    # -------- create logistic regression model --------
    def build_model(self):
        """
        using liblinear because it behaves better
        for high-dimensional sparse text data
        """

        return LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        )

    # -------- train both feature variants --------
    def train_and_save(self):
        """
        trains two models:
        BoW version and TFIDF version
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

        # -------- train BoW model --------
        print("\nTraining Logistic Regression (BoW)...")
        model_bow = self.build_model()
        model_bow.fit(X_train_bow, y_train)

        self.evaluate(
            "LogReg + BoW",
            model_bow,
            X_test_bow,
            y_test
        )

        # -------- train TFIDF model --------
        print("\nTraining Logistic Regression (TFIDF)...")
        model_tfidf = self.build_model()
        model_tfidf.fit(X_train_tfidf, y_train)

        self.evaluate(
            "LogReg + TFIDF",
            model_tfidf,
            X_test_tfidf,
            y_test
        )

        # -------- persist models --------
        joblib.dump(
            model_bow,
            "models/model_lr_bow.pkl"
        )

        joblib.dump(
            model_tfidf,
            "models/model_lr_tfidf.pkl"
        )

        print("\nLogistic Regression models saved")
