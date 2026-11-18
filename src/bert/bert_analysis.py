import argparse

from ..common import LANGUAGES, PREDICTION_FIELDS, load_dataset, prediction_path, prediction_row, write_rows
from ..model_utils import batches, configure_hub

BERT = {
    "en": ("j-hartmann/sentiment-roberta-large-english-3-classes", "81cdc0fe3eee1bc18d95ffdfb56b2151a39c9007"),
    "tr": ("gmzx/berturk-turkish-sentiment", "f44c2464cb1817fc5e779d5dcfb6c1a766e42384"),
}
BATCH_SIZE = 32
BERT_TOKENS = 256


def bert(language):
    configure_hub()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name, revision = BERT[language]
    dataset = load_dataset(language)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision).to(device).eval()
    labels = {key: value.casefold() for key, value in model.config.id2label.items()}
    labels = {key: "neutral" if value in {"notr", "nötr"} else value for key, value in labels.items()}
    if sorted(labels.values()) != ["negative", "neutral", "positive"]:
        raise ValueError(f"Model etiketleri geçersiz: {labels}")

    predictions = []
    with torch.inference_mode():
        for batch in batches(dataset, BATCH_SIZE):
            inputs = tokenizer(
                [row["text"] for row in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=BERT_TOKENS,
            )
            probabilities = model(**inputs.to(device)).logits.softmax(dim=-1)
            scores, indices = probabilities.max(dim=-1)
            for row, score, index in zip(batch, scores.tolist(), indices.tolist(), strict=True):
                predictions.append(prediction_row(row, labels[index], score=f"{score:.8f}"))
            if len(predictions) % 512 == 0 or len(predictions) == len(dataset):
                print(f"{language} BERT ({device}): {len(predictions)}/{len(dataset)}", flush=True)
    write_rows(prediction_path(language, "bert"), PREDICTION_FIELDS, predictions)


def main():
    parser = argparse.ArgumentParser(description="İngilizce ve Türkçe BERT analizlerini sırayla çalıştır.")
    parser.parse_args()
    for language in LANGUAGES:
        bert(language)
