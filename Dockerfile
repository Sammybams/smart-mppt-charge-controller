FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY models ./models

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "smart_mppt.api:app", "--host", "0.0.0.0", "--port", "8000"]

