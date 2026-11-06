# Databricks notebook source
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main(registered_model_name=None):
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and model with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="random_forest_model",
            signature=infer_signature(X_test, y_pred),
        )

        # Register the model when a Unity Catalog name is provided
        if registered_model_name:
            mlflow.set_registry_uri("databricks-uc")
            version = mlflow.register_model(
                model_info.model_uri,
                registered_model_name,
            )
            print(f"Model registered: {version.name}, version {version.version}")


if __name__ == "__main__":
    main("workspace.default.IrisRFModel")