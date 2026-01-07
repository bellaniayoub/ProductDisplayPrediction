import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --------------------------
# Load categorical dataset
# --------------------------
# Modified to local path
data_cat = pd.read_csv(r"C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\categorical\data_categorical_mdlpc.csv", sep=';')
print("Columns:", data_cat.columns.tolist())
print("Shape:", data_cat.shape)

# Features / target
X = data_cat.drop('Y', axis=1)
y = data_cat['Y']

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Encode categorical features (label encoding each column for CategoricalNB)
X_enc = X.apply(LabelEncoder().fit_transform)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_enc, test_size=0.2, random_state=42
)

# --------------------------
# Categorical Naive Bayes
# --------------------------
print("\nTraining Categorical Naive Bayes model (MDLPC)...")
nb = CategoricalNB()
nb.fit(X_train, y_train)

# Predictions
y_pred = nb.predict(X_test)

# --------------------------
# Evaluation
# --------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("=== Naive Bayes Evaluation - Categorical (MDLPC) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
print("\nSaving Confusion Matrix...")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
plt.title("Categorical Naive Bayes Confusion Matrix (MDLPC)")
plt.savefig(r"C:\Users\PC\Desktop\ProductDisplayPrediction\nb_mdlpc_confusion_matrix.png")
# plt.show()
print("\nGraph saved in C:\\Users\\PC\\Desktop\\ProductDisplayPrediction\\")
