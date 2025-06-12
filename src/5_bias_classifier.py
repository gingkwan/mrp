import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def train_classifier():
    # Load clean training data
    train_df = pd.read_csv("../data/4_train_data/train_metadata.csv")
    train_emb = np.load("../data/4_train_data/train_text_emb.npy")

    # Create binary label using heuristic: 1 = masculine bias, 0 = feminine bias
    train_df["label"] = train_df["count_score"].apply(lambda x: 1 if x > 0 else 0)

    X_train, X_val, y_train, y_val = train_test_split(train_emb, train_df["label"], test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred))

    return clf

if __name__ == "__main__":
    # Train classifier
    clf = train_classifier()

    # Apply classifier to filtered data
    for i in range(9):
        meta_file = f"../data/1_3_count_metadata/count_metadata_{i}.csv"
        emb_file = f"../data/2_1_filtered_text_emb/filtered_text_emb_{i}.npy"

        if os.path.exists(meta_file) and os.path.exists(emb_file):
            df = pd.read_csv(meta_file)
            emb = np.load(emb_file)

            # Predict probability and map to [-1, 1]
            preds = clf.predict_proba(emb)[:, 1] * 2 - 1
            df["bias_score_orig"] = preds

            output_file = f"../1_5_bias_metadata/bias_metadata_{i}.csv"
            df.to_csv(output_file, index=False)
            print(f"Processed and saved {output_file}")
        else:
            print(f"Skipping missing files: {meta_file} or {emb_file}")

    print("Processing complete.")