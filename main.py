#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def main():
    # 1. DATASET
    corpus = [
        # Sincere / Positive (Label = 0
        ("This movie was absolutely wonderful and inspiring", 0),
        ("Thank you so much for your kind help today", 0),
        ("I had a great time at the family dinner", 0),
        ("The customer support was fast helpful and friendly", 0),
        ("I really enjoyed reading this book from start to finish", 0),
        ("The weather outside is lovely and warm today", 0),
        ("Congratulations on your well deserved promotion", 0),
        ("This food tastes delicious and perfectly cooked", 0),
        ("I am genuinely excited for our upcoming vacation", 0),
        ("Your presentation was very clear and informative", 0),
        # Sarcastic (Label = 1)
        ("Oh fantastic my flight got canceled again love that", 1),
        ("Great job breaking the printer right before the meeting", 1),
        ("Nothing better than waiting two hours in the rain", 1),
        ("Oh wonderful another surprise billing charge on my card", 1),
        ("I just love getting stuck in traffic for three hours", 1),
        ("Brilliant work forgetting to save the document", 1),
        ("Oh super my phone battery died right when I needed GPS", 1),
        ("Yay another endless meeting that could have been an email", 1),
        ("Oh perfect my coffee spilled all over my laptop", 1),
        ("Thanks a lot for leaving the door wide open in winter", 1),
    ]

    texts, labels = zip(*corpus)

    # 2. FEATURE EXTRACTION
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # 3. MODEL TRAINING
    model = LogisticRegression(C=1.0, random_state=42)
    model.fit(X, y)

    print("=== Model Training Complete ===")
    print(
        f"Extracted Vocabulary Size: {len(vectorizer.get_feature_names_out())} unique words/phrases\n"
    )

    # 4. INSPECT LEARNED WEIGHTS
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    top_sarcastic_idx = np.argsort(coefficients)[-5:][::-1]
    top_sincere_idx = np.argsort(coefficients)[:5]

    print("--- Top Indicators for SARCASTIC Tone (Positive Weights) ---")
    for idx in top_sarcastic_idx:
        print(
            f"  Phrase: '{feature_names[idx]}': Weight = +{coefficients[idx]:.3f}"
        )

    print("\n--- Top Indicators for SINCERE Tone (Negative Weights) ---")
    for idx in top_sincere_idx:
        print(
            f"  Phrase: '{feature_names[idx]}': Weight = {coefficients[idx]:.3f}"
        )

    # 5. PREDICT ON UNSEEN DATA
    test_samples = [
        "Oh fantastic, another delay on the subway today",
        "I am so grateful for all your hard work on this project",
        "Great, my computer just crashed in the middle of saving",
        "The meal was delicious and the staff was super friendly",
    ]

    X_test = vectorizer.transform(test_samples)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\n=== Predictions on Unseen Test Samples ===")
    label_map = {0: "SINCERE", 1: "SARCASTIC"}
    for text, pred, prob in zip(test_samples, predictions, probabilities):
        confidence = prob[pred] * 100
        print(f'\nInput Text:  "{text}"')
        print(
            f"Prediction:  {label_map[pred]} (Confidence: {confidence:.1f}%)"
        )

    # 6. VISUALIZE FEATURE IMPORTANCE
    plt.figure(figsize=(10, 5))
    top_indices = np.hstack([top_sincere_idx, top_sarcastic_idx])
    top_words = feature_names[top_indices]
    top_weights = coefficients[top_indices]
    colors = ["#2b5c8f" if w < 0 else "#d95f02" for w in top_weights]

    plt.barh(top_words, top_weights, color=colors)
    plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
    plt.xlabel(
        "Model Weight (Negative = Sincere, Positive = Sarcastic)"
    )
    plt.title(
        "NLP Feature Importance: Key Words Driving Sarcasm vs. Sincerity"
    )
    plt.tight_layout()

    plt.savefig("feature_weights.png")
    print("\nSaved output visualization to 'feature_weights.png'")


if __name__ == "__main__":
    main()