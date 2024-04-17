# train_evaluate.py
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from prepare_data import load_data, preprocess_data, feature_selection

def train_model(X_train, y_train, model):
    pipeline = make_pipeline(SMOTE(k_neighbors=3), model)
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    print(classification_report(y_test, y_pred))
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == "__main__":
    # Assuming command-line arguments for file path
    data = load_data(sys.argv[1])
    data = preprocess_data(data, ['UnwantedColumn'])
    X, y = data.drop('Target', axis=1), data['Target']
    X_rfe, _ = feature_selection(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()  # Example model
    pipeline = train_model(X_train, y_train, model)
    evaluate_model(pipeline, X_test, y_test)
