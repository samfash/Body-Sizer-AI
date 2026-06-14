FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything
COPY . .

# expose port
EXPOSE 8000

# start command uses uvicorn. In production use gunicorn + uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
