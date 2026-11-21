FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY llm_cache ./llm_cache
RUN useradd -m app
USER app
EXPOSE 8000
CMD ["uvicorn", "llm_cache.app:app", "--host", "0.0.0.0", "--port", "8000"]
