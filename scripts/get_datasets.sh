#!/usr/bin/env bash
# Download datasets used by visionforge.
#
# Usage:
#   bash scripts/download_datasets.sh             # download all
#   bash scripts/download_datasets.sh cifar10     # only CIFAR-10
#   bash scripts/download_datasets.sh flickr8k    # only Flickr8k
set -euo pipefail

DATASETS_DIR="${DATASETS_DIR:-datasets}"
mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

download() {
    local url="$1"
    local out="$2"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$out" "$url"
    else
        echo "Error: need wget or curl" >&2
        exit 1
    fi
}

download_cifar10() {
    if [ -d "cifar-10-batches-py" ]; then
        echo "[CIFAR-10] already exists, skipping"
        return
    fi
    echo "[CIFAR-10] downloading ..."
    download "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" "cifar-10-python.tar.gz"
    tar -xzf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    echo "[CIFAR-10] done"
}

download_flickr8k() {
    if [ -d "Flicker8k_Dataset" ]; then
        echo "[Flickr8k] already exists, skipping"
        return
    fi
    echo "[Flickr8k] downloading images (~1GB) ..."
    download "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip" "Flickr8k_Dataset.zip"
    echo "[Flickr8k] downloading captions ..."
    download "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip" "Flickr8k_text.zip"
    unzip -q Flickr8k_Dataset.zip
    unzip -q Flickr8k_text.zip
    rm Flickr8k_Dataset.zip Flickr8k_text.zip
    echo "[Flickr8k] done"
}

case "${1:-all}" in
    cifar10)  download_cifar10 ;;
    flickr8k) download_flickr8k ;;
    all)      download_cifar10; download_flickr8k ;;
    *)        echo "Usage: $0 [cifar10|flickr8k|all]"; exit 1 ;;
esac

echo ""
echo "All requested datasets are under $DATASETS_DIR/"