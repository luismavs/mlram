import polars as pl
import requests


def test_inference(url="http://0.0.0.0:8000"):
    data = pl.read_parquet("data/data.parquet").head(10)
    data = data.select([col for col in data.columns if col != "classification"])

    payload = data.serialize(format="json")

    out = requests.post(url=url + "/classify", json={"data": payload})

    print(out.status_code)

    print(out.json())


test_inference()
