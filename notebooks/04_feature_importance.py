import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/data/processed/simulated_features.csv")

X = df[["avg_sent_len", "num_pronouns", "num_nouns", "num_verbs"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance – Cognitive Decline Prediction")
plt.xlabel("Importance")
plt.ylabel("Linguistic Feature")
plt.show()