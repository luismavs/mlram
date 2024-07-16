FROM python:3.12-slim-bookworm

WORKDIR /home/mlram

COPY requirements.txt /home/mlram/requirements.txt

RUN pip install -r /home/mlram/requirements.txt

COPY ./mlram /home/mlram/mlram
COPY ./model /home/mlram/model
COPY ./data /home/mlram/data

WORKDIR /home/mlram

EXPOSE 8000
CMD ["uvicorn", "mlram.fastapi_api:app", "--host", "0.0.0.0", "--port", "8000"]

