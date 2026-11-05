import mlflow
from mlflow.tracking import MlflowClient

from notebooks.train_model import main, train_model


def test_model_training():
    model, accuracy, X_test = train_model()

    assert accuracy >= 0.90
    assert len(model.predict(X_test)) == len(X_test)


def test_mlflow_logging(tmp_path):
    previous_uri = mlflow.get_tracking_uri()

    try:
        database_path = (tmp_path / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database_path}")

        client = MlflowClient()
        experiment_id = client.create_experiment(
            "iris-test",
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        mlflow.set_experiment(experiment_id=experiment_id)

        run_id, accuracy = main()
        run = client.get_run(run_id)

        assert run.info.status == "FINISHED"
        assert run.data.metrics["accuracy"] == accuracy
        assert run.data.params["n_estimators"] == "100"

    finally:
        mlflow.set_tracking_uri(previous_uri)