import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --------------------------
# Load FAMD dataset
# --------------------------
# Modified to local path
data_famd = pd.read_csv(r"C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\continous\data_famd.csv", sep=';')
print("Columns:", data_famd.columns.tolist())
print("Shape:", data_famd.shape)

# Features / target
X = data_famd.drop('Y', axis=1)
y = data_famd['Y']

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42
)

# --------------------------
# Gaussian Naive Bayes
# --------------------------
print("\nTraining Gaussian Naive Bayes model (FAMD)...")
nb = GaussianNB()
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

print("=== Naive Bayes Evaluation - FAMD ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
print("\nSaving Confusion Matrix...")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
plt.title("Gaussian Naive Bayes Confusion Matrix (FAMD)")
plt.savefig(r"C:\Users\PC\Desktop\ProductDisplayPrediction\nb_famd_confusion_matrix.png")
# plt.show()
