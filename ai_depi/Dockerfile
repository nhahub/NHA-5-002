FROM python:3.11-slim
WORKDIR /service
COPY pyproject.toml .
COPY app ./app
RUN pip install --no-cache-dir .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
