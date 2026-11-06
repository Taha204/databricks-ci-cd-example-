from mlflow.tracking import MlflowClient

from notebooks.train_model import main


def test_model_training(tmp_path, monkeypatch):
    # Use a fresh local MLflow database for this test
    monkeypatch.chdir(tmp_path)
    tracking_uri = "sqlite:///mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    # Run the training script
    main()

    # Check that MLflow recorded an accuracy metric
    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(experiment_ids=["0"])

    assert len(runs) == 1, "Expected one MLflow run"
    assert "accuracy" in runs[0].data.metrics, "Accuracy metric not found"