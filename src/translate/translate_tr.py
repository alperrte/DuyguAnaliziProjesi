from ..common import TRANSLATED, TRANSLATION_FIELDS, load_dataset, sha256, write_json, write_rows
from ..model_utils import batches, configure_hub

TRANSLATOR = ("Helsinki-NLP/opus-mt-tr-en", "19c65427cc2af5f191337d4899e0348c4af25902")
BATCH_SIZE = 32
TRANSLATION_TOKENS = 256
TRANSLATION_BEAMS = 4
TRANSLATION_REPEAT_NGRAM = 3


def translate():
    configure_hub()
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    source = load_dataset("tr")
    name, revision = TRANSLATOR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision).to(device).eval()
    output = []
    fallback_ids = []
    retried_ids = []
    with torch.inference_mode():
        for batch in batches(source, BATCH_SIZE):
            inputs = tokenizer([row["text"] for row in batch], return_tensors="pt", padding=True)
            if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError("Bir Türkçe metin çeviri modelinin girdi sınırını aşıyor.")
            ids = model.generate(
                **inputs.to(device), num_beams=TRANSLATION_BEAMS, do_sample=False,
                no_repeat_ngram_size=TRANSLATION_REPEAT_NGRAM, early_stopping=True,
                max_new_tokens=TRANSLATION_TOKENS, forced_eos_token_id=None,
            )
            texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            for row, text, tokens in zip(batch, texts, ids.tolist(), strict=True):
                if model.config.eos_token_id not in tokens[1:]:
                    retried_ids.append(row["id"])
                    single = tokenizer(row["text"], return_tensors="pt").to(device)
                    retry = model.generate(
                        **single, num_beams=1, do_sample=False, no_repeat_ngram_size=TRANSLATION_REPEAT_NGRAM,
                        max_new_tokens=TRANSLATION_TOKENS, forced_eos_token_id=None,
                    )
                    if model.config.eos_token_id not in retry[0, 1:].tolist():
                        raise ValueError(f"{row['id']}: çeviri çıktı sınırında tamamlanamadı.")
                    text = tokenizer.decode(retry[0], skip_special_tokens=True)
                if not text.strip():
                    fallback_ids.append(row["id"])
                    text = row["text"]
                output.append({**row, "text_en": text.strip()})
            if len(output) % 512 == 0 or len(output) == len(source):
                print(f"OPUS-MT ({device}): {len(output)}/{len(source)}", flush=True)
    write_rows(TRANSLATED, TRANSLATION_FIELDS, output)
    write_json(TRANSLATED.with_name("ceviri_manifestosu.json"), {
        "model": name,
        "revision": revision,
        "rows": len(output),
        "beams": TRANSLATION_BEAMS,
        "max_new_tokens": TRANSLATION_TOKENS,
        "no_repeat_ngram_size": TRANSLATION_REPEAT_NGRAM,
        "greedy_retry_ids": retried_ids,
        "empty_output_original_text_ids": fallback_ids,
        "sha256": sha256(TRANSLATED.read_bytes()),
    })
