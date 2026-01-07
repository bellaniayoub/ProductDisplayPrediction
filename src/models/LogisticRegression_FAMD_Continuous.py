import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def main():
    filepath = r'C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\continous\data_famd.csv'
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    X = df.drop('Y', axis=1)
    y = LabelEncoder().fit_transform(df['Y'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"\n{'='*20} Logistic Regression (FAMD) {'='*20}")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
