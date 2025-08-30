from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    print(f"=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.show()  # Changed from plt.close() to plt.show() to see the plot
    plt.close()
    
    # Feature importance for tree-based models
    if hasattr(model.best_estimator_, 'feature_importances_'):
        importances = model.best_estimator_.feature_importances_
        feature_importances = pd.Series(importances, index=X_test.columns)
        
        plt.figure(figsize=(10, 6))
        feature_importances.sort_values(ascending=False).head(10).plot(kind='bar')
        plt.title(f"Feature Importances - {model_name}")
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importances.png")
        plt.show()  # Changed from plt.close() to plt.show() to see the plot
        plt.close()
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }