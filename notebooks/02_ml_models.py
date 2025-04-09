import pandas as pd
import numpy as np
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import urllib3
import re

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize NLP
nlp = spacy.load("en_core_web_sm")


# Enhanced feature extraction functions (no Java dependency)
def calculate_entropy(items):
    item_counts = Counter(items)
    total = sum(item_counts.values())
    return entropy([count / total for count in item_counts.values()])


def count_grammar_issues(text):
    """Simplified grammar checking without LanguageTool"""
    # Count basic grammar issues using heuristics
    errors = 0
    # Check for sentence fragments (no verb)
    for sent in nlp(text).sents:
        if not any(token.pos_ == "VERB" for token in sent):
            errors += 1
    # Check for double spaces
    errors += len(re.findall(r'\s{2,}', text))
    return errors


def extract_enhanced_features(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]

    features = {
        # Basic features
        'avg_sent_len': sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
        'num_pronouns': sum(1 for token in doc if token.pos_ == "PRON"),
        'num_nouns': sum(1 for token in doc if token.pos_ == "NOUN"),
        'num_verbs': sum(1 for token in doc if token.pos_ == "VERB"),

        # Advanced features
        'hesitation_ratio': sum(1 for t in doc if t.text.lower() in ('um', 'uh')) / len(doc) if len(doc) > 0 else 0,
        'repetition_score': (len(tokens) - len(set(tokens))) / len(tokens) if tokens else 0,
        'grammar_issues': count_grammar_issues(text),
        'pos_entropy': calculate_entropy([token.pos_ for token in doc]),
        'pronoun_ratio': sum(1 for t in doc if t.pos_ == 'PRON') / len(doc) if len(doc) > 0 else 0,
        'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0
    }
    return pd.Series(features)


# Load and prepare data
df = pd.read_csv("/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/data/processed/simulated_features.csv")

# Extract all features
print("Extracting features...")
feature_df = df['text'].apply(extract_enhanced_features)
df = pd.concat([df, feature_df], axis=1)

# Prepare X and y
X = df.drop(['text', 'clean_text', 'label'], axis=1, errors='ignore')
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Parameter grid
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [5, 10, None],
    'clf__min_samples_split': [2, 5]
}

# Grid search with stratified cross-validation
print("Running grid search...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 folds for speed
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_clf = grid_search.best_estimator_

# Evaluation
y_pred = best_clf.predict(X_test)

print("\nBest Parameters:", grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Healthy", "Decline"],
            yticklabels=["Healthy", "Decline"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
feature_importances = pd.Series(best_clf.named_steps['clf'].feature_importances_,
                                index=X.columns)
feature_importances.sort_values().plot.barh(figsize=(10, 6))
plt.title("Feature Importances")
plt.show()

# Error analysis
df_test = X_test.copy()
df_test['text'] = df.loc[X_test.index, 'text']
df_test['label'] = y_test
df_test['prediction'] = y_pred
misclassified = df_test[df_test['label'] != df_test['prediction']]

print("\nSample Misclassified Texts:")
print(misclassified[['text', 'label', 'prediction']].sample(min(5, len(misclassified))))