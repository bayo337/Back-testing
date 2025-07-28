import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib # For saving the model
import seaborn as sns
import matplotlib.pyplot as plt
import os # For robust path handling

# --- Configuration ---
# Get the directory where the script is located to build robust paths
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

ML_DATA_FILE = os.path.join(SCRIPT_DIR, 'ml_training_data.csv')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'trade_win_predictor.joblib')
TARGET_COLUMN = 'target'

def generate_feature_analysis_report(df, top_features, filename='feature_analysis_report.txt'):
    """
    Generates a detailed text report comparing the statistics of top features
    for winning vs. losing trades.
    """
    report_lines = []
    
    winners = df[df[TARGET_COLUMN] == 1]
    losers = df[df[TARGET_COLUMN] == 0]
    
    report_lines.append("="*80)
    report_lines.append("          Detailed Feature Analysis Report")
    report_lines.append("="*80)
    report_lines.append("\nThis report shows a statistical breakdown of the top 20 most important features,")
    report_lines.append("comparing their values for winning trades vs. losing trades.\n")

    for feature in top_features:
        report_lines.append("\n" + "-"*80)
        report_lines.append(f"Feature: {feature}")
        report_lines.append("-"*80)
        
        # Get descriptive statistics
        winner_stats = winners[feature].describe()
        loser_stats = losers[feature].describe()
        
        # Combine into a single DataFrame for pretty printing
        stats_df = pd.concat([winner_stats, loser_stats], axis=1)
        stats_df.columns = ['WINNERS', 'LOSERS']
        
        report_lines.append(stats_df.to_string())
        report_lines.append("\n")
        
    try:
        with open(filename, 'w') as f:
            f.write("\n".join(report_lines))
        print(f"Detailed feature analysis report saved to '{filename}'")
    except Exception as e:
        print(f"Could not save feature analysis report: {e}")


def train_model():
    """
    This function will:
    1. Load the prepared machine learning dataset.
    2. Split the data into training and testing sets.
    3. Train a Gradient Boosting Classifier.
    4. Evaluate the model's performance.
    5. Save the trained model to a file.
    """
    # 1. Load the data
    try:
        df = pd.read_csv(ML_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: The training data file was not found at '{ML_DATA_FILE}'.")
        print("Please run the 'create_ml_dataset.py' script first.")
        return
        
    print(f"Loaded dataset with {len(df)} samples.")
    
    # Drop any non-numeric columns that might have been accidentally included
    df = df.select_dtypes(include=['number'])
    
    # Fill any potential missing values with the median of the column
    df = df.fillna(df.median())

    # --- NEW: Remove all transaction-related features ---
    transaction_cols = [col for col in df.columns if 'transactions' in col]
    if transaction_cols:
        X = df.drop(columns=[TARGET_COLUMN] + transaction_cols)
        print(f"\nRemoved {len(transaction_cols)} transaction-related features.")
    else:
        X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # 4. Initialize and train the model
    print("\nTraining the Gradient Boosting Classifier...")
    # These are good starting parameters for this type of problem
    model = GradientBoostingClassifier(
        n_estimators=150, 
        learning_rate=0.1, 
        max_depth=4, 
        subsample=0.8,
        random_state=42,
        verbose=1 # This will show the training progress
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # --- Classification Report ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loser (0)', 'Winner (1)']))
    
    # --- ROC AUC Score ---
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loser', 'Winner'], yticklabels=['Loser', 'Winner'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix.png'))
    print("\nConfusion Matrix saved to 'confusion_matrix.png'")
    
    # --- Feature Importance ---
    feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    top_20_features = feature_importance.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_20_features)
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'feature_importance.png'))
    print("Feature Importance plot saved to 'feature_importance.png'")

    # --- Generate Detailed Feature Report ---
    generate_feature_analysis_report(df, top_20_features['feature'], os.path.join(SCRIPT_DIR, 'feature_analysis_report.txt'))


    # 6. Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel has been saved to '{MODEL_FILE}'.")
    print("You can now use this file to make predictions on new trade setups.")

if __name__ == '__main__':
    train_model() 