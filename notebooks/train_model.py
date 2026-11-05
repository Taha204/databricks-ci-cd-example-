import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model():
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
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

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        # Registration is performed on Databricks.
        # CI tests only verify training and metric logging.
        if on_databricks:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model",
                signature=infer_signature(
                    X_test,
                    model.predict(X_test),
                ),
                input_example=X_test.head(3),
            )

            model_uri = f"runs:/{run.info.run_id}/random_forest_model"

            registered = mlflow.register_model(
                model_uri,
                "workspace.default.IrisRFModel",
            )

            print(f"Modèle : {registered.name}")
            print(f"Version : {registered.version}")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Run ID : {run.info.run_id}")

        return run.info.run_id, accuracy


if __name__ == "__main__":
    main()


def train_model():
    iris = load_iris(as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, X_test


def main():
    with mlflow.start_run() as run:
        model, accuracy, X_test = train_model()

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Run ID : {run.info.run_id}")

        return run.info.run_id, accuracy


if __name__ == "__main__":
    main()