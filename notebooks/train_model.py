import os
import tempfile
from unittest.mock import patch

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "workspace.default.irisrfmodel_taha"


def train_model():
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, X_test


def main():
    on_databricks = bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))

    if on_databricks:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run() as run:
        model, accuracy, X_test = train_model()

        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metric("accuracy", accuracy)

        if on_databricks:
            with tempfile.TemporaryDirectory(dir="/tmp", prefix="iris-mlflow-") as tmp:
                with (
                    patch.object(tempfile, "tempdir", tmp),
                    patch(
                        "mlflow.utils.file_utils.get_databricks_local_temp_dir",
                        return_value=tmp,
                        create=True,
                    ),
                    patch(
                        "mlflow.utils.databricks_utils.get_databricks_local_temp_dir",
                        return_value=tmp,
                    ),
                ):
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        name="random_forest_model",
                        signature=infer_signature(X_test, model.predict(X_test)),
                        input_example=X_test.head(3),
                    )
                    registered = mlflow.register_model(
                        model_info.model_uri,
                        MODEL_NAME,
                    )

            print(f"Modèle : {registered.name}")
            print(f"Version : {registered.version}")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Run ID : {run.info.run_id}")
        return run.info.run_id, accuracy


if __name__ == "__main__":
    main()