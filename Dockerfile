FROM python:3.11-slim

WORKDIR /srv/app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Default build: heuristic-only, CPU, no torch/transformers -- small image,
# no multi-GB model download. Pass `--build-arg INSTALL_ML=1` to also install
# torch/transformers and enable the ViT/CLIP-backed detector paths.
COPY requirements.txt requirements-ml.txt ./
ARG INSTALL_ML=0
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY app ./app
COPY benchmarking ./benchmarking

ENV USE_PRETRAINED_MODELS=0
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
