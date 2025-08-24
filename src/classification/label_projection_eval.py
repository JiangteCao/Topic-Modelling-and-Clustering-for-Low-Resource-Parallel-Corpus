import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def main():
    # load embeddings and projected labels
    all_hsb_embeddings = np.load("/content/drive/MyDrive/Colab Notebooks/labse/labse_hsb.npy")
    df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/labse/de_labeled_sentences.csv")
    projected_labels = df["predicted_label"].tolist()

    # label encoder
    le = LabelEncoder()
    y_hsb = le.fit_transform(projected_labels)

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        all_hsb_embeddings, y_hsb, test_size=0.2, random_state=42, stratify=y_hsb
    )

    # train logistic regression
    clf_hsb = LogisticRegression(C=2.48, max_iter=1000, multi_class='multinomial', solver='lbfgs')#up to previous result
    clf_hsb.fit(X_train, y_train)

    # predict all embeddings
    hsb_predicted_labels_encoded = clf_hsb.predict(all_hsb_embeddings)
    hsb_predicted_labels = le.inverse_transform(hsb_predicted_labels_encoded)

    # consistency evaluation
    label_agreement_rate = accuracy_score(projected_labels, hsb_predicted_labels)
    print("Label agreement accuracy (German vs Upper Sorbian):", round(label_agreement_rate, 4))

    # classification report
    y_test_pred = clf_hsb.predict(X_test)
    print("Classification report (HSB Logistic Regression):")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))


if __name__ == "__main__":
    main()
