from fastapi import FastAPI
import polars as pl
from mlram.model import Classifier
from io import StringIO

app = FastAPI()

classifier = Classifier()


@app.post("/classify")
def classify(data: dict) -> str:

    data = pl.DataFrame.deserialize(StringIO(data["data"]), format="json")
    out = classifier.classify(data)

    return out.serialize(format="json")
