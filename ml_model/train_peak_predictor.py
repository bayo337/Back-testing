import pandas as pd
import os
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'peak_predictor.joblib')
REPORT_FILE = os.path.join(SCRIPT_DIR, 'peak_feature_analysis_report.txt')
CONFUSION_MATRIX_FILE = os.path.join(SCRIPT_DIR, 'peak_confusion_matrix.png')
FEATURE_IMPORTANCE_FILE = os.path.join(SCRIPT_DIR, 'peak_feature_importance.png')
SHAP_SUMMARY_FILE = os.path.join(SCRIPT_DIR, 'peak_shap_summary.png')
FEATURE_DIST_DIR = os.path.join(SCRIPT_DIR, 'feature_distributions_peak')
TOP_N_FEATURES = 20

def generate_model_analysis(model, X_train, y_train, feature_names):
    """
    Generates a comprehensive analysis report including feature importance,
    SHAP summary, and distribution plots for top features.
    """
    print("Generating comprehensive model analysis for Peak model...")
    os.makedirs(FEATURE_DIST_DIR, exist_ok=True)

    # 1. Standard Feature Importance
    feature_importances = pd.DataFrame(
        model.feature_importances_,
        index=feature_names,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    top_features = feature_importances.head(TOP_N_FEATURES)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=top_features.importance, y=top_features.index)
    plt.title(f'Top {TOP_N_FEATURES} Feature Importances (At-Peak Model)')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FILE)
    plt.close()

    # 2. SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=TOP_N_FEATURES)
    plt.title(f'SHAP Summary (Top {TOP_N_FEATURES} Features, At-Peak Model)')
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_FILE)
    plt.close()

    # 3. Distribution Plots
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_train_df['target'] = y_train.values
    
    for feature in top_features.index:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='target', y=feature, data=X_train_df)
        plt.title(f'Distribution of "{feature}" (At-Peak Model)')
        plt.xticks([0, 1], ['Loser (Small Drop)', 'Winner (Big Drop)'])
        plt.savefig(os.path.join(FEATURE_DIST_DIR, f'{feature}_distribution.png'))
        plt.close()

    # 4. Text Report
    report_content = "Actionable Insights Report (At-Peak Model)\n"
    report_content += "=" * 60 + "\n\n"
    report_content += f"See SHAP summary: {SHAP_SUMMARY_FILE}\n"
    report_content += f"See distribution plots in: {FEATURE_DIST_DIR}/\n\n"
    report_content += f"Top {TOP_N_FEATURES} Features:\n" + top_features.to_string() + "\n\n"
    
    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)

def main():
    """Main function to train the at-peak model with hyperparameter tuning."""
    print("Loading data for Peak model...")
    df = pd.read_csv(DATA_FILE)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    y = df['target']
    peak_cols = [col for col in df.columns if col.endswith('_peak')]
    
    if not peak_cols:
        print("Error: No columns with '_peak' suffix found.")
        return
        
    X = df[peak_cols]
    print(f"Training model with {len(X.columns)} at-the-peak features...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Performing hyperparameter tuning...")
    param_grid = {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    print("\nEvaluating best model...")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loser', 'Winner'], yticklabels=['Loser', 'Winner'])
    plt.title('Confusion Matrix (Tuned At-Peak Model)')
    plt.savefig(CONFUSION_MATRIX_FILE)
    plt.close()

    generate_model_analysis(best_model, X_train, y_train, X.columns)

    print(f"\nSaving best model to {MODEL_FILE}...")
    joblib.dump(best_model, MODEL_FILE)
    print("\nPeak model training complete!")

if __name__ == '__main__':
    main()
