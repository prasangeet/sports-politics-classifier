import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


"""
this class trains linear SVM models for text classification

linear SVM is usually one of the strongest classic
models for high-dimensional sparse text features
so this is our high-performance baseline
"""


class SVMTrainer:

    # -------- evaluation helper --------
    def evaluate(self, name, model, X_test, y_test):
        """
        prints full metric summary
        same format used for all models
        """

        preds = model.predict(X_test)

        print(f"\n===== {name} =====")
        print("Accuracy:", accuracy_score(y_test, preds))

        print("\nClassification Report:")
        print(classification_report(y_test, preds))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

    # -------- load saved feature matrices --------
    def load_features(self):
        """
        loads BoW and TFIDF sparse matrices
        built earlier during feature step
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

    # -------- build svm model --------
    def build_model(self):
        """
        linear SVC works well for sparse text vectors
        C controls margin softness
        """

        return LinearSVC(C=1.0)

    # -------- train both feature variants --------
    def train_and_save(self):
        """
        trains svm on BoW and TFIDF features
        then saves both trained models
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
        print("\nTraining SVM (BoW)...")
        svm_bow = self.build_model()
        svm_bow.fit(X_train_bow, y_train)

        self.evaluate(
            "SVM + BoW",
            svm_bow,
            X_test_bow,
            y_test
        )

        # -------- TFIDF model --------
        print("\nTraining SVM (TFIDF)...")
        svm_tfidf = self.build_model()
        svm_tfidf.fit(X_train_tfidf, y_train)

        self.evaluate(
            "SVM + TFIDF",
            svm_tfidf,
            X_test_tfidf,
            y_test
        )

        # -------- save models --------
        joblib.dump(
            svm_bow,
            "models/model_svm_bow.pkl"
        )

        joblib.dump(
            svm_tfidf,
            "models/model_svm_tfidf.pkl"
        )

        print("\nSVM models saved")
