import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "result"
CACHE = ROOT / ".cache"
LANGUAGES = ("en", "tr")
LABELS = ("positive", "negative", "neutral")
ABSTAIN = "abstain"
PER_CLASS = 1_000
DATASETS = {"en": "ingilizce.csv", "tr": "turkce.csv"}
TRANSLATED = DATA / "turkce_ingilizce_ceviri.csv"
SOURCE_FIELDS = ("id", "text", "label", "source_file", "source_line")
TRANSLATION_FIELDS = (*SOURCE_FIELDS, "text_en")
PREDICTION_FIELDS = ("id", "label", "prediction", "score", "input_tokens", "output_tokens")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary.write(content)
        destination = Path(temporary.name)
    destination.replace(path)


def write_json(path, data):
    write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8-sig", newline="", delete=False
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        destination = Path(temporary.name)
    destination.replace(path)


def load_dataset(language):
    rows = read_rows(DATA / DATASETS[language])
    if Counter(row["label"] for row in rows) != Counter({label: PER_CLASS for label in LABELS}):
        raise ValueError(f"{language}: sınıf dağılımı geçersiz.")
    if len({row["id"] for row in rows}) != len(rows) or any(not row["text"].strip() for row in rows):
        raise ValueError(f"{language}: tekrarlanan kimlik veya boş metin var.")
    return rows


def load_translations(original):
    rows = read_rows(TRANSLATED)
    restored = [{field: row[field] for field in SOURCE_FIELDS} for row in rows]
    if restored != original or any(not row["text_en"].strip() for row in rows):
        raise ValueError("Çeviri dosyası özgün Türkçe kaynakla eşleşmiyor.")
    return rows


def prediction_path(language, method):
    name = Path(DATASETS[language]).stem
    return RESULTS / method / f"{name}_{method}.csv"


def prediction_row(row, prediction, **details):
    return {"id": row["id"], "label": row["label"], "prediction": prediction, **details}
