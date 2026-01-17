import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def evaluate_model(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():
    wine = load_wine()
    data = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target, name="target")

    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, stratify=target
    )

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]

    results = []
    for name, model in models:
        model.fit(x_train, y_train)
        results.append(evaluate_model(name, model, x_test, y_test))

    results_df = pd.DataFrame(results)
    print("Model performance on the Wine dataset:")
    print(results_df.to_string(index=False))

    best_model = results_df.sort_values("f1_score", ascending=False).iloc[0]
    print(
        f"\nBest model by weighted F1-score: {best_model['model']} "
        f"(F1={best_model['f1_score']:.4f})"
    )


if __name__ == "__main__":
    main()
