import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    # Load FAMD dataset
    filepath = r'C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\continous\data_famd.csv'
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    
    # Separation Features (X) / Target (Y)
    X = df[['FAMD_1', 'FAMD_2', 'FAMD_3', 'FAMD_4']]
    y = df['Y']
    
    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Instanciation du modèle SVC
    print("\nTraining SVM model (FAMD)...")
    svm = SVC(kernel='rbf', random_state=42) # Default RBF kernel

    # Entraînement du modèle
    svm.fit(X_train, y_train)

    # Prédictions
    y_pred = svm.predict(X_test)

    # Classification report
    print("\n=== SVM Evaluation (FAMD) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix heatmap
    print("\nSaving Confusion Matrix heatmap...")
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=svm.classes_, yticklabels=svm.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title("SVM Confusion Matrix (FAMD)")
    plt.savefig(r"C:\Users\PC\Desktop\ProductDisplayPrediction\svm_famd_confusion_matrix.png")
    # plt.show()
    print("\nGraph saved in C:\\Users\\PC\\Desktop\\ProductDisplayPrediction\\")

if __name__ == "__main__":
    main()
