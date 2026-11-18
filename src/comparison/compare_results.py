import argparse
from collections import Counter

from ..common import (
    ABSTAIN, LABELS, LANGUAGES, RESULTS, load_dataset, load_translations, prediction_path, read_rows, write_rows,
)

METHODS = ("vader", "bert", "claude")
DISPLAY = {"en": "İngilizce", "tr": "Türkçe", "vader": "VADER", "bert": "BERT", "claude": "Claude"}
LABEL_TR = {"positive": "Pozitif", "negative": "Negatif", "neutral": "Nötr", ABSTAIN: "Yanıt yok"}
PREDICTED_LABELS = (*LABELS, ABSTAIN)


def score_predictions(language, method, rows):
    path = prediction_path(language, method)
    expected = {row["id"]: row["label"] for row in rows}
    predictions = {}
    for row in read_rows(path):
        row_id = row["id"]
        valid = PREDICTED_LABELS if method == "claude" else LABELS
        if row_id in predictions or expected.get(row_id) != row["label"] or row["prediction"] not in valid:
            raise ValueError(f"{path.name}: geçersiz kimlik, etiket veya tahmin: {row_id}")
        predictions[row_id] = row["prediction"]
    if predictions.keys() != expected.keys():
        raise ValueError(f"{path.name}: {len(expected.keys() - predictions.keys())} beklenen tahmin eksik.")

    counts = Counter((label, predictions[row_id]) for row_id, label in expected.items())
    matrix = [[counts[actual, predicted] for predicted in PREDICTED_LABELS] for actual in LABELS]
    total = len(rows)
    correct = sum(matrix[index][index] for index in range(len(LABELS)))
    per_class = []
    summary = {"language": language, "method": method, "total": total, "correct": correct, "errors": total - correct,
               "accuracy": correct / total, "abstentions": sum(row[-1] for row in matrix)}
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        errors = support - true_positive
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support
        f1 = 2 * true_positive / (support + predicted)
        per_class.append({"language": language, "method": method, "label": label, "support": support,
                          "correct": true_positive, "errors": errors, "precision": precision, "recall": recall, "f1": f1})
        summary[f"{label}_errors"] = errors
    summary["macro_f1"] = sum(row["f1"] for row in per_class) / len(LABELS)
    details = [
        {"language": language, "method": method, "actual": actual, "predicted": predicted,
         "count": counts[actual, predicted]}
        for actual in LABELS for predicted in PREDICTED_LABELS
    ]
    return summary, per_class, details, matrix, predictions


def charts(summaries, matrices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.right": False, "axes.spines.top": False})
    RESULTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), layout="constrained")
    color_scale = None
    for index, language in enumerate(LANGUAGES):
        for column, method in enumerate(METHODS):
            axis = axes[index, column]
            axis.set_axis_off()
            matrix = matrices.get((language, method))
            if matrix is None:
                continue
            axis.set_axis_on()
            color_scale = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1_000)
            axis.set_title(f"{DISPLAY[language]} · {DISPLAY[method]}", fontsize=11)
            axis.set_xticks(range(4), [LABEL_TR[label] for label in PREDICTED_LABELS], fontsize=9)
            axis.set_yticks(range(3), [LABEL_TR[label] for label in LABELS], fontsize=9)
            if not column:
                axis.set_ylabel("Gerçek etiket")
            if index == 1:
                axis.set_xlabel("Tahmin")
            for y, row in enumerate(matrix):
                for x, count in enumerate(row):
                    axis.text(x, y, str(count), ha="center", va="center",
                              color="white" if count > 500 else "#11243a", fontweight="bold")
    if color_scale is not None:
        figure.colorbar(color_scale, ax=axes, shrink=0.75, label="Örnek sayısı")
    figure.savefig(RESULTS / "karisiklik_matrisleri.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), sharey=True, layout="constrained")
    palette = {"vader": "#486581", "bert": "#147d72", "claude": "#a64d25"}
    for axis, language in zip(axes, LANGUAGES, strict=True):
        scores = [row for row in summaries if row["language"] == language]
        bars = axis.bar(
            [DISPLAY[row["method"]] for row in scores], [100 * row["accuracy"] for row in scores],
            color=[palette[row["method"]] for row in scores], width=0.55,
        )
        axis.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
        axis.set_ylim(0, 100)
        axis.set_title(DISPLAY[language])
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Doğruluk (%)")
    figure.savefig(RESULTS / "dogruluk.png", dpi=170)
    plt.close(figure)


def run():
    required = [prediction_path(language, method) for language in LANGUAGES for method in METHODS]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Karşılaştırma için eksik tahmin dosyaları: {', '.join(missing)}")

    summaries, classes, confusion, errors = [], [], [], []
    matrices = {}
    for language in LANGUAGES:
        rows = load_dataset(language)
        translations = {
            row["id"]: row["text_en"] for row in load_translations(rows)
        } if language == "tr" else {}
        by_method = {}
        for method in METHODS:
            summary, per_class, details, matrix, predictions = score_predictions(language, method, rows)
            summaries.append(summary)
            classes.extend(per_class)
            confusion.extend(details)
            matrices[language, method] = matrix
            by_method[method] = predictions
            print(f"{DISPLAY[language]} {DISPLAY[method]}: {summary['correct']}/{summary['total']} "
                  f"(%{100 * summary['accuracy']:.2f}), makro F1={summary['macro_f1']:.4f}")
        for row in rows:
            predictions = {method: by_method[method][row["id"]] for method in METHODS}
            wrong = [method for method, label in predictions.items() if label != row["label"]]
            if wrong:
                errors.append({"language": language, "id": row["id"], "label": row["label"],
                               "text": row["text"], "text_en": translations.get(row["id"], ""),
                               "wrong_methods": ", ".join(wrong), **predictions})

    for filename, rows in (
        ("yontem_ozeti.csv", summaries), ("sinif_bazinda_hatalar.csv", classes),
        ("karisiklik_matrisleri.csv", confusion), ("hata_ornekleri.csv", errors),
    ):
        fields = tuple(rows[0]) if rows else ("language", "id", "label", "text", "text_en", "wrong_methods", *METHODS)
        write_rows(RESULTS / filename, fields, rows)
    charts(summaries, matrices)


def main():
    parser = argparse.ArgumentParser(description="VADER, BERT ve Claude sonuçlarını iki dilde karşılaştır.")
    parser.parse_args()
    try:
        run()
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
