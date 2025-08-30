from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=0, n_estimators=50)  # Reduced n_estimators
    
    # Simplified parameter grid for faster training
    cv_params = {
        'max_depth': [5, None],
        'n_estimators': [50, 100]  # Reduced options
    }
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=3, refit='recall')  # Reduced CV folds
    rf_cv.fit(X_train, y_train)
    
    return rf_cv

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(objective='binary:logistic', random_state=0, n_estimators=50)  # Reduced n_estimators
    
    # Simplified parameter grid for faster training
    cv_params = {
        'max_depth': [4, 6],
        'learning_rate': [0.1],
        'n_estimators': [50, 100]  # Reduced options
    }
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=3, refit='recall')  # Reduced CV folds
    xgb_cv.fit(X_train, y_train)
    
    return xgb_cv