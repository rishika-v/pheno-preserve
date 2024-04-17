import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_curve, auc, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
import logging

# Setup logging
logging.basicConfig(filename='model_evaluation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def load_data(file_path):
    return pd.read_csv(file_path)

def train_and_evaluate(X, y, models, cv, task_type='classification'):
    results = {name: {'y_test': [], 'y_pred': [], 'y_proba': [] if task_type == 'classification' else None} for name, _ in models.items()}
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for name, model in models.items():
            if task_type == 'classification':
                pipeline = make_pipeline(SMOTE(k_neighbors=3), model) if 'SMOTE' in str(model) else make_pipeline(model)
            else:
                pipeline = make_pipeline(model)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            results[name]['y_test'].extend(y_test)
            results[name]['y_pred'].extend(y_pred)
            if task_type == 'classification' and hasattr(model, 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                results[name]['y_proba'].extend(y_proba)

    return results

def plot_and_save_results(results, task_type='classification'):
    for name, data in results.items():
        if task_type == 'classification':
            logging.info(f"Classification Report for {name}:\n{classification_report(data['y_test'], data['y_pred'])}")
            cm = confusion_matrix(data['y_test'], data['y_pred'])
            plt.figure()
            ConfusionMatrixDisplay(cm, display_labels=np.unique(data['y_test'])).plot(cmap='Blues')
            plt.title(f'Confusion Matrix for {name}')
            plt.savefig(f'confusion_matrix_{name}.png')
            plt.close()

            if data['y_proba']:
                fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
                roc_auc = auc(fpr, tpr)
                plt.figure()
                RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name).plot()
                plt.title(f'AUROC for {name}')
                plt.savefig(f'auroc_{name}.png')
                plt.close()
        else:
            mse = mean_squared_error(data['y_test'], data['y_pred'])
            rmse = np.sqrt(mse)
            r2 = r2_score(data['y_test'], data['y_pred'])
            logging.info(f"Regression Results for {name}: MSE={mse}, RMSE={rmse}, R^2={r2}")
            plt.figure()
            plt.scatter(data['y_test'], data['y_pred'], alpha=0.5)
            plt.plot([min(data['y_test']), max(data['y_test'])], [min(data['y_test']), max(data['y_test'])], '--r')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'Regression Results for {name}')
            plt.savefig(f'regression_plot_{name}.png')
            plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Usage: python train_evaluate.py <X_path> <y_path> <task_type>")
        sys.exit(1)

    X_path, y_path, task_type = sys.argv[1], sys.argv[2], sys.argv[3]
    X = load_data(X_path)
    y = load_data(y_path)

    if task_type == 'classification':
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=115, max_depth=26, min_samples_split=4, min_samples_leaf=1, max_features='sqrt', class_weight={0: 0.2, 1: 0.8}, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='linear', C=1.0, probability=True),
            'Decision Tree': DecisionTreeClassifier(max_depth=5),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif task_type == 'regression':
        models = {
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor()
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    else:
        logging.error("Invalid task type specified. Please choose 'classification' or 'regression'.")
        sys.exit(1)

    results = train_and_evaluate(X, y, models, cv, task_type)
    plot_and_save_results(results, task_type)
