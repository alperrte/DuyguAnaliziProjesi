from ..common import (
    LANGUAGES, PREDICTION_FIELDS, load_dataset, load_translations, prediction_path, prediction_row, write_rows,
)


def vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    for language in LANGUAGES:
        dataset = load_dataset(language)
        rows, column = (dataset, "text") if language == "en" else (load_translations(dataset), "text_en")
        predictions = []
        for row in rows:
            score = analyzer.polarity_scores(row[column])["compound"]
            if score >= 0.05:
                label = "positive"
            elif score <= -0.05:
                label = "negative"
            else:
                label = "neutral"
            predictions.append(prediction_row(row, label, score=score))
        write_rows(prediction_path(language, "vader"), PREDICTION_FIELDS, predictions)
        print(f"{language} VADER: {len(predictions)} tahmin", flush=True)
