import os
from pathlib import Path
from tempfile import gettempdir

from .common import CACHE


def configure_hub():
    cache = CACHE / "huggingface"
    if os.name == "nt" and not str(cache).isascii():
        # Windows SentencePiece wheels cannot open some Unicode model paths.
        cache = Path(gettempdir()) / "duygu-analizi-huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]
