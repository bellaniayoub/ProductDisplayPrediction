import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ==========================
# 1. Load dataset
# ==========================
# Path modified to local windows path
data_famd = pd.read_csv(r"C:\Users\PC\Desktop\ProductDisplayPrediction\Data\processed\continous\data_famd.csv", sep=';')

print("Columns:", data_famd.columns.tolist())
print("Shape:", data_famd.shape)
print(data_famd.head())

# ==========================
# 2. Features / Target
# ==========================
X = data_famd.drop('Y', axis=1).values
y = data_famd['Y']

# Encode target (No_Displ, Displ -> 0,1)
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Classes:", le.classes_)

# ==========================
# 3. Train / Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ==========================
# 4. Gradient Boosting Model
# ==========================
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

print("\nTraining Gradient Boosting model (FAMD)...")
gb.fit(X_train, y_train)

# ==========================
# 5. Predictions
# ==========================
y_pred = gb.predict(X_test)

# ==========================
# 6. Evaluation
# ==========================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== Gradient Boosting Evaluation (FAMD) ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ==========================
# 7. Confusion Matrix
# ==========================
print("\nPlotting Confusion Matrix...")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
plt.title("Confusion Matrix - Gradient Boosting (FAMD)")
plt.savefig(r"C:\Users\PC\Desktop\ProductDisplayPrediction\gb_famd_confusion_matrix.png")
# plt.show() # Disabled for headless execution

# ==========================
# 8. Feature Importance
# ==========================
feat_imp = pd.DataFrame({
    "Feature": data_famd.drop('Y', axis=1).columns,
    "Importance": gb.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importance:")
print(feat_imp.head(10))

# ==========================
# 9. Plot Feature Importance
# ==========================
print("\nPlotting Feature Importance...")
plt.figure(figsize=(10,6))
plt.barh(feat_imp["Feature"][:10], feat_imp["Importance"][:10])
plt.xlabel("Importance")
plt.title("Top Feature Importance - Gradient Boosting (FAMD)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(r"C:\Users\PC\Desktop\ProductDisplayPrediction\gb_famd_feature_importance.png")
# plt.show() # Disabled for headless execution

print("\nGraphs saved in C:\\Users\\PC\\Desktop\\ProductDisplayPrediction\\")
