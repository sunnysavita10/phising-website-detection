import joblib
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import print_header


def train_ann(X_train, X_test, y_train, y_test, params, save_path):
    print_header("TRAINING ANN MLPClassifier (MLflow Enabled)")

    # FIX: config compatibility
    params = params.copy()
    if "hidden_layers" in params:
        params["hidden_layer_sizes"] = tuple(params["hidden_layers"])
        del params["hidden_layers"]

    with mlflow.start_run(run_name="ANN_MLP_Phishing"):

        # 🔹 Log hyperparameters
        mlflow.log_params(params)

        ann = MLPClassifier(**params)
        ann.fit(X_train, y_train)

        preds = ann.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # 🔹 Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1_score", report["1"]["f1-score"])

        # 🔹 Save locally
        joblib.dump(ann, save_path)

        # 🔹 Log model to MLflow
        mlflow.sklearn.log_model(ann, artifact_path="ann_model")

        print("ANN Accuracy:", acc)
        print("Report:\n", classification_report(y_test, preds))
        print(f"ANN model saved → {save_path}")
