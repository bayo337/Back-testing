import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'post_predictor.joblib')
REPORT_FILE = os.path.join(SCRIPT_DIR, 'post_feature_analysis_report.txt')
CONFUSION_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'post_confusion_matrix.png')
FEATURE_IMPORTANCE_FILE = os.path.join(SCRIPT_DIR, 'post_feature_importance.png')

def analyze_feature_importance(model, X_train, y_train):
    """Analyzes and saves feature importance and statistics."""
    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
    
    report_content = "Feature Importance Analysis (Post-Peak)\n"
    report_content += "=====================================\n\n"
    report_content += "Top 20 Most Important Features:\n"
    report_content += feature_importances.head(20).to_string()
    report_content += "\n\n"

    X_train_winners = X_train[y_train == 1]
    X_train_losers = X_train[y_train == 0]

    report_content += "Statistical Comparison of Top 20 Features (Winners vs. Losers)\n"
    report_content += "-------------------------------------------------------------\n"
    for feature in feature_importances.head(20).index:
        stats_winners = X_train_winners[feature].describe()
        stats_losers = X_train_losers[feature].describe()
        comparison_df = pd.DataFrame({'Winners (Big Drop)': stats_winners, 'Losers (Small Drop)': stats_losers})
        report_content += f"\nFeature: {feature}\n"
        report_content += comparison_df.to_string()
        report_content += "\n"

    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=feature_importances.head(20).importance, y=feature_importances.head(20).index)
    plt.title('Top 20 Feature Importances (Post-Peak Model)')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FILE)
    plt.close()

def main():
    """Main function to train the model."""
    print("Loading data for Post-Peak model...")
    df = pd.read_csv(DATA_FILE)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    y = df['target']
    post_cols = [col for col in df.columns if col.endswith('_post')]
    post_cols = [col for col in post_cols if 'transactions' not in col]
    
    if not post_cols:
        print("Error: No columns with '_post' suffix found. Cannot train model.")
        return
        
    X = df[post_cols]
    print(f"Training model with {len(X.columns)} post-peak features...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loser', 'Winner'], yticklabels=['Loser', 'Winner'])
    plt.title('Confusion Matrix (Post-Peak Model)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()

    print(f"Analyzing and saving feature importance to {REPORT_FILE}...")
    analyze_feature_importance(model, X_train, y_train)

    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    print("\nPost-Peak model training complete!")

if __name__ == '__main__':
    main() 