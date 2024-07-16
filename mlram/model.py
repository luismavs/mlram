import polars as pl
from sklearn.ensemble import RandomForestClassifier
import joblib
from mlram.onnx_utils import load_onnx, save_onnx, classify_onnx


def train(model_type="joblib"):
    df = pl.read_parquet("data/data.parquet")
    print(model_type)
    y = df["classification"].to_numpy()
    X = df.select([col for col in df.columns if col != "classification"]).to_numpy()

    model = RandomForestClassifier(n_estimators=50)

    print("Training model ...")

    model.fit(X=X, y=y)
    print("Finished model training.")
    if model_type == "joblib":
        joblib.dump(model, "model/model.joblib")
    if model_type == "onnx":
        save_onnx(model, X)


class JoblibClassifier:
    def __init__(self) -> None:
        self.model: RandomForestClassifier = joblib.load("model/model.joblib")

    def classify(self, data: pl.DataFrame) -> pl.DataFrame:
        out = self.model.predict(data.to_numpy())

        return pl.from_numpy(out)


class OnnxClassifier:
    def __init__(self) -> None:
        self.model = load_onnx()

    def classify(self, data: pl.DataFrame) -> pl.DataFrame:
        out = classify_onnx(self.model, data.to_numpy())

        return pl.from_numpy(out)


Classifier = JoblibClassifier
# Classifier = OnnxClassifier

train("joblib")
