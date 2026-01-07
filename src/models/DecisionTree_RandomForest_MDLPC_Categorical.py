import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def main():
    filepath = r'C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\categorical\data_categorical_mdlpc.csv'
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    X = pd.get_dummies(df.drop('Y', axis=1))
    y = LabelEncoder().fit_transform(df['Y'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for name, model in [("Decision Tree", DecisionTreeClassifier(random_state=42)), 
                        ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=100))]:
        print(f"\n{'='*20} {name} (MDLPC) {'='*20}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
