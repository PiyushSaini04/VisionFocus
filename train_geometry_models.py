import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from constants import EXPECTED_FEATURE_DIM, GEOMETRY_CSV_PATH, GEOMETRY_MODEL_PATH, LABELS


def main():
    if EXPECTED_FEATURE_DIM is None:
        raise ValueError(
            "EXPECTED_FEATURE_DIM is None in constants.py. "
            "Run extract_mediapipe_features.py first, verify the feature count, then set EXPECTED_FEATURE_DIM."
        )

    df = pd.read_csv(GEOMETRY_CSV_PATH)
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values.astype(int)

    assert (
        X.shape[1] == EXPECTED_FEATURE_DIM
    ), f"Feature mismatch: CSV has {X.shape[1]} cols, expected {EXPECTED_FEATURE_DIM}"

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Class counts:", dict(zip(uniq.tolist(), cnt.tolist())))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    sample_weights = np.array([class_weight_dict[yi] for yi in y_train], dtype=np.float32)
    print("Class weights:", class_weight_dict)

    # CatBoost expects weights aligned to class indices. We train only labels 0-3 here.
    max_class = int(np.max(classes))
    cb_class_weights = [class_weight_dict.get(i, 1.0) for i in range(max_class + 1)]

    candidates = {
        "CatBoost": CatBoostClassifier(
            class_weights=cb_class_weights,
            eval_metric="TotalF1",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=100,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            objective="multi:softprob",
            num_class=len(np.unique(y)),
            eval_metric="mlogloss",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    candidates["CatBoost"].fit(X_train, y_train)
    candidates["XGBoost"].fit(X_train, y_train, sample_weight=sample_weights)
    candidates["RandomForest"].fit(X_train, y_train)

    results = {}
    for name, model in candidates.items():
        y_pred = model.predict(X_val)
        macro_f1 = float(f1_score(y_val, y_pred, average="macro"))

        t0 = time.perf_counter()
        for _ in range(1000):
            model.predict(X_val[:1])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results[name] = {"macro_f1": macro_f1, "latency_ms": latency_ms, "model": model}

        print(f"\n{'='*50}")
        print(f"{name}  macro-F1={macro_f1:.4f}  1000-call latency={latency_ms:.2f}ms")
        print(
            classification_report(
                y_val, y_pred, target_names=[LABELS[i] for i in range(4)]
            )
        )

    # Selection: highest macro-F1; if within 0.02 of best, pick lowest latency.
    best_f1 = max(v["macro_f1"] for v in results.values())
    near_best = {k: v for k, v in results.items() if (best_f1 - v["macro_f1"]) <= 0.02}
    best_name = min(near_best, key=lambda k: near_best[k]["latency_ms"])
    best_model = results[best_name]["model"]
    print(f"\nSelected: {best_name}")

    Path = __import__("pathlib").Path
    Path(GEOMETRY_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, GEOMETRY_MODEL_PATH)
    print(f"Saved to {GEOMETRY_MODEL_PATH}")


if __name__ == "__main__":
    main()

