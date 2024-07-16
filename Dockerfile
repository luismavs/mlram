FROM python:3.12-slim-bookworm

WORKDIR /home/mlram

COPY requirements.txt /home/mlram/requirements.txt

RUN pip install uv
RUN uv pip install -r /home/mlram/requirements.txt --system

COPY ./mlram /home/mlram/mlram
COPY ./model /home/mlram/model
COPY ./data /home/mlram/data
COPY ./pyproject.toml /home/mlram/pyproject.toml
# RUN uv pip install --no-deps --system -e /home/mlram

WORKDIR /home/mlram

EXPOSE 8000
CMD ["uvicorn", "mlram.fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]

