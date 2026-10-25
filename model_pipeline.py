import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_pipeline():
    # 1. Load Dataset
    df = pd.read_csv('Palo Alto Networks.csv')
    print(f"Dataset Loaded Successfully. Shape: {df.shape}")

    # 2. Feature Engineering (as per instructions)
    df['IncomeToExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['PromotionDelay'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['EngagementScore'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + 
                             df['JobInvolvement'] + df['RelationshipSatisfaction']) / 4
    df['WorkloadStress'] = np.where((df['OverTime'] == 'Yes') & (df['DistanceFromHome'] > 10), 1, 0)

    # 3. Encoding Categorical Variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = df.copy()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop(columns=['Attrition'])
    y = df_encoded['Attrition']

    # 4. Train-Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Scaling Numerical Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model Training & Evaluation
    models = {
        'Logistic Regression (Baseline)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = []
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results.append({
            'Model': name,
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
        })

    eval_df = pd.DataFrame(results)
    print("\n--- Model Performance Evaluation ---")
    print(eval_df.to_string(index=False))

    # 7. Risk Scoring Framework Assignment
    best_model = trained_models['Logistic Regression (Baseline)']
    all_scaled = scaler.transform(X)
    probabilities = best_model.predict_proba(all_scaled)[:, 1]
    
    df['Attrition_Probability'] = probabilities
    df['Risk_Category'] = pd.cut(
        df['Attrition_Probability'],
        bins=[-0.1, 0.3, 0.6, 1.0],
        labels=['Low Risk (<30%)', 'Medium Risk (30-60%)', 'High Risk (>60%)']
    )
    
    print("\n--- Risk Category Distribution ---")
    print(df['Risk_Category'].value_counts())
    
    return df, eval_df

if __name__ == '__main__':
    run_pipeline()