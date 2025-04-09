import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/data/processed/simulated_features.csv")

X = df[["avg_sent_len", "num_pronouns", "num_nouns", "num_verbs"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (Linear)": SVC(kernel="linear"),
    "SVM (RBF)": SVC(kernel="rbf"),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print("=" * 40)

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="coolwarm")
plt.title("Model Accuracy Comparison")
plt.xlim(0, 1)
plt.show()