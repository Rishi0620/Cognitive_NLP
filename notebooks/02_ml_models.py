import pandas as pd
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import urllib3
import re
import shap
from sklearn.utils.class_weight import compute_sample_weight
import os
import warnings

# Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize spaCy with TextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')


# --- Enhanced Feature Engineering Functions ---
def calculate_entropy(items):
    item_counts = Counter(items)
    total = sum(item_counts.values())
    return entropy([count / total for count in item_counts.values()])


def count_grammar_issues(text):
    errors = 0
    for sent in nlp(text).sents:
        if not any(token.pos_ == "VERB" for token in sent):
            errors += 1
    errors += len(re.findall(r'\s{2,}', text))
    return errors


def extract_enhanced_features(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    dep_relations = [token.dep_ for token in doc]
    sentiment = doc._.blob

    return pd.Series({
        'text_length': len(text),
        'word_count': len(tokens),
        'avg_word_length': np.mean([len(token) for token in tokens]) if tokens else 0,
        'unique_word_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
        'num_nouns': sum(1 for token in doc if token.pos_ == "NOUN"),
        'num_verbs': sum(1 for token in doc if token.pos_ == "VERB"),
        'num_adj': sum(1 for token in doc if token.pos_ == "ADJ"),
        'num_adv': sum(1 for token in doc if token.pos_ == "ADV"),
        'num_pronouns': sum(1 for token in doc if token.pos_ == "PRON"),
        'num_punct': sum(1 for token in doc if token.is_punct),
        'avg_sent_len': np.mean([len(sent) for sent in doc.sents]) if list(doc.sents) else 0,
        'pos_entropy': calculate_entropy([token.pos_ for token in doc]),
        'dep_entropy': calculate_entropy(dep_relations),
        'grammar_issues': count_grammar_issues(text),
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'num_unique_lemmas': len(set(lemmas)),
        'lemma_entropy': calculate_entropy(lemmas),
        'num_pauses': len(re.findall(r'[,;:-]', text)),
        'has_future_tense': int(any(token.tag_ == 'MD' for token in doc)),
        'has_past_tense': int(any(token.tag_ in ('VBD', 'VBN') for token in doc)),
        'quote_count': text.count('"') + text.count("'"),
        'hesitation_words': sum(1 for t in doc if t.text.lower() in ('um', 'uh', 'ah', 'er'))
    })


# --- Main Execution ---
if __name__ == "__main__":
    # --- Load Data ---
    df = pd.read_csv("/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/data/processed/simulated_features.csv")

    # Ensure 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("'text' column missing in input CSV")

    print("Extracting enhanced features...")
    enhanced_features = df['text'].apply(extract_enhanced_features)
    df = pd.concat([df, enhanced_features], axis=1)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    print("Columns after removing duplicates:", df.columns.tolist())

    # --- Train/Test Preparation ---
    X = df.drop(columns=['clean_text', 'label'], errors='ignore')  # Keep 'text' column
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Verify no duplicates in final features
    assert len(X_train.columns) == len(set(X_train.columns)), "Duplicate columns detected in features"

    # Build pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')),
        ('svd', TruncatedSVD(n_components=50, random_state=42))
    ])

    preprocessor = ColumnTransformer([
        ('text', text_pipeline, 'text'),
        ('num', 'passthrough', X.columns.drop('text'))
    ])

    # Simplify model if XGBoost issues persist
    base_models = [
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
        # Removed XGBoost temporarily to avoid library loading issues
    ]

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('scaler', StandardScaler()),
        ('clf', StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3
        ))
    ])

    # Simplified parameter grid
    param_grid = {
        'clf__rf__n_estimators': [100, 200],
        'clf__rf__max_depth': [5, 10],
        'pre__text__tfidf__max_features': [500, 1000],
        'pre__text__svd__n_components': [30, 50]
    }

    print("Running grid search...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_weighted',
        n_jobs=2,  # Reduced further for stability
        verbose=1
    )

    sample_weights = compute_sample_weight('balanced', y_train)
    grid_search.fit(X_train, y_train)

    # Evaluation
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)

    print("\nBest Parameters:", grid_search.best_params_)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualization code remains the same...

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Feature Importances (for Random Forest)
# Modify the feature importance section like this:
if hasattr(best_clf.named_steps['clf'].estimators_[0], 'feature_importances_'):
    plt.figure(figsize=(12, 8))

    # Get the preprocessor and feature info
    preprocessor = best_clf.named_steps['pre']
    svd = preprocessor.named_transformers_['text'].named_steps['svd']
    svd_components = svd.n_components  # Without underscore!

    # Create feature names
    tfidf_features = [f"tfidf_{i}" for i in range(svd_components)]
    num_feature_names = X.columns.drop('text').tolist()
    all_feature_names = tfidf_features + num_feature_names

    # Get importances
    importances = best_clf.named_steps['clf'].estimators_[0].feature_importances_

    # Create and plot the Series
    feature_importances = pd.Series(importances, index=all_feature_names)
    feature_importances.nlargest(20).plot.barh()
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()
    # SHAP Analysis (for XGBoost)
try:
    explainer = shap.TreeExplainer(best_clf.named_steps['clf'].estimators_[1])
    X_test_processed = best_clf.named_steps['pre'].transform(X_test)
    shap_values = explainer.shap_values(X_test_processed)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_processed,
                      feature_names=[f"tfidf_{i}" for i in range(50)] + list(X.columns.difference(['text'])),
                      plot_type="bar", max_display=20)
    plt.title("SHAP Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"SHAP analysis failed: {str(e)}")

# Misclassified Examples Analysis
df_test = X_test.copy()
df_test['text'] = df.loc[X_test.index, 'text']
df_test['label'] = y_test
df_test['prediction'] = y_pred
misclassified = df_test[df_test['label'] != df_test['prediction']]

print("\nSample Misclassified Texts:")
print(misclassified[['text', 'label', 'prediction']].sample(min(5, len(misclassified))))

# Feature Correlation Analysis
plt.figure(figsize=(12, 10))
corr_matrix = pd.concat([X_train, y_train], axis=1).corr()[['label']].sort_values('label')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlation with Label")
plt.tight_layout()
plt.show()