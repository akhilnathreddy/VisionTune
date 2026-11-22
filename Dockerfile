# GPU inference backend for the VisionTune demo (FastAPI).
# Build:  docker build -t visiontune-api .
# Run:    docker run --gpus all -p 8000:8000 -v $PWD/adapters:/app/adapters visiontune-api
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt requirements-app.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt
COPY src ./src
COPY backend ./backend
COPY configs ./configs
COPY results/reports ./results/reports
RUN pip install --no-cache-dir -e .
ENV VISIONTUNE_CONFIG=configs/serve.yaml HF_HOME=/app/.hf
EXPOSE 8000
CMD ["uvicorn", "--factory", "backend.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
