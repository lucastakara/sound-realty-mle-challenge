FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.3 \
    pandas==2.2.0 \
    scikit-learn==1.4.0

COPY app/ ./app/
COPY model/ ./app/model/
COPY data/zipcode_demographics.csv ./app/data/zipcode_demographics.csv

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]