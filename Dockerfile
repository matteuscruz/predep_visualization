FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema exigidas por netCDF4/pyarrow em algumas plataformas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8050
EXPOSE 8050

CMD ["sh", "-c", "gunicorn app:server -b ${HOST}:${PORT} --workers 2 --timeout 120"]
