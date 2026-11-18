import random
import unicodedata
from collections import Counter, defaultdict
from urllib.request import urlopen

from .common import CACHE, DATA, DATASETS, LABELS, PER_CLASS, SOURCE_FIELDS, sha256, write_bytes, write_json, write_rows

EN_REV = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
TR_REV = "9647a08a9f0f6ac117d6f79a65578315b27682d9"
EN_ROOT = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/{EN_REV}/datasets/sentiment"
TR_ROOT = f"https://raw.githubusercontent.com/sercankulcu/sentiment-analysis-of-tweets-in-Turkish/{TR_REV}/training%20data"
SOURCES = {
    "en/test_text.txt": (f"{EN_ROOT}/test_text.txt", "09a93a55c63fd93f97485ef7302889d7edb4091cd49733aa37da094f0bfa0675"),
    "en/test_labels.txt": (f"{EN_ROOT}/test_labels.txt", "6afb4afe9374d1f983bcf9a7c79b108d0f37fdf020a83f30488309bed215db9d"),
    "en/mapping.txt": (f"{EN_ROOT}/mapping.txt", "c0572796fbe261d8ea5fd16c5061076fbba49bc69fc865d6abccaf60daacee24"),
    "tr/positive.txt": (f"{TR_ROOT}/positive.txt", "67c938b1da2345bc8ef40f607330f434b642bf3f9176642261e025458029f28d"),
    "tr/negative.txt": (f"{TR_ROOT}/negative.txt", "d8e8680470fde6d5d83397e7a4d1bd92f81be62b9df12307c96dd6c16a67ede2"),
    "tr/notr.txt": (f"{TR_ROOT}/notr.txt", "a8f0c1da85df82eeedb289896b3c147f858dd704efdbc82e86334f9e9eba78e6"),
}
SEED = 20261118


def load_source(name):
    url, expected = SOURCES[name]
    path = CACHE / "sources" / name
    if path.exists():
        content = path.read_bytes()
    else:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    if sha256(content) != expected:
        raise ValueError(f"Kaynak dosyanın SHA-256 değeri uyuşmuyor: {name}")
    if not path.exists():
        write_bytes(path, content)
    return content.decode("utf-8-sig").splitlines()


def source_rows(language):
    if language == "en":
        labels = dict(line.split("\t") for line in load_source("en/mapping.txt"))
        texts = load_source("en/test_text.txt")
        codes = load_source("en/test_labels.txt")
        for line, (text, code) in enumerate(zip(texts, codes, strict=True), start=1):
            yield "en/test_text.txt", line, text, labels[code]
    else:
        for filename, label in (("positive", "positive"), ("negative", "negative"), ("notr", "neutral")):
            source = f"tr/{filename}.txt"
            for line, text in enumerate(load_source(source), start=1):
                yield source, line, text, label


def balanced(language):
    groups = defaultdict(list)
    stats = Counter()
    source_counts = Counter()
    for source, line, text, label in source_rows(language):
        source_counts[label] += 1
        canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if not canonical:
            stats["empty"] += 1
            continue
        if label not in LABELS:
            raise ValueError(f"Bilinmeyen kaynak etiketi: {label}")
        groups[canonical].append({
            "id": f"{language}-{source.rsplit('/', 1)[-1].removesuffix('.txt')}-{line:05d}",
            "text": text.strip(),
            "label": label,
            "source_file": source,
            "source_line": line,
        })

    eligible = defaultdict(list)
    for entries in groups.values():
        labels = {row["label"] for row in entries}
        if len(labels) > 1:
            stats["conflicting_labels"] += len(entries)
            continue
        retained = entries[0]
        eligible[retained["label"]].append(retained)
        stats["duplicates"] += len(entries) - 1

    rng = random.Random(f"{SEED}/{language}")
    chosen = []
    for label in LABELS:
        pool = eligible[label]
        if len(pool) < PER_CLASS:
            raise ValueError(f"{language}/{label}: {len(pool)} uygun örnek.")
        chosen.extend(rng.sample(pool, PER_CLASS))
    rng.shuffle(chosen)
    return chosen, {
        "source_counts": dict(source_counts),
        "removed": dict(stats),
        "eligible": {label: len(eligible[label]) for label in LABELS},
        "selected": dict(Counter(row["label"] for row in chosen)),
    }


def run():
    metadata = {"seed": SEED, "sources": {}, "datasets": {}}
    for name, (url, checksum) in SOURCES.items():
        metadata["sources"][name] = {"url": url, "sha256": checksum}
    for language, filename in DATASETS.items():
        rows, details = balanced(language)
        path = DATA / filename
        write_rows(path, SOURCE_FIELDS, rows)
        metadata["datasets"][language] = {"file": filename, "sha256": sha256(path.read_bytes()), **details}
        print(f"{language}: {details['selected']}", flush=True)
    write_json(DATA / "kaynak_manifestosu.json", metadata)


if __name__ == "__main__":
    run()
