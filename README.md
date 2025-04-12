## Early Detection of Cognitive Decline through NLP

A Comparative Study of Linguistic Features and Deep Learning Models

## Project Overview

This project explores the use of Natural Language Processing (NLP) techniques to detect early signs of cognitive decline—such as Alzheimer’s or dementia—through the analysis of linguistic patterns in text. The goal is to identify key features in spoken or written language that signal neurodegenerative changes, and build both classical and deep learning models to classify these patterns.

## Objectives
	•	Understand linguistic and semantic indicators of cognitive decline
	•	Build a classical machine learning pipeline using engineered features
	•	Fine-tune transformer-based deep learning models (BERT) for comparison
	•	Visualize model predictions and feature importance
	•	Develop an interpretable and extendable NLP framework

## Project Roadmap

Phase 1: Get Oriented
	•	Learned NLP basics: tokenization, POS tagging, lemmatization, embeddings
	•	Selected dataset (or simulated similar data due to limited access)

Phase 2: Feature Engineering
	•	Extracted linguistic features:
	•	Average sentence length
	•	Lexical diversity
	•	Pronoun usage
	•	POS tag frequency
	•	Visualized key feature distributions

Phase 3: Classical ML Modeling
	•	Models Used:
	•	Logistic Regression
	•	Random Forest
	•	SVM
	•	Best macro F1-score: 0.654

Phase 4: Deep Learning with Transformers (In Progress)
	•	Setup for BERT fine-tuning using HuggingFace’s transformers
	•	Initial tokenization and model architecture defined

Phase 5: Packaging & Presentation (Planned)
	•	Build a Streamlit dashboard or Jupyter-based demo
	•	Highlight misclassifications and visual explanations

Results (so far)
	•	Best performing model: Random Forest (n_estimators=100)
	•	Average macro F1-score: 0.65
	•	Class 0 (Control) shows high precision/recall
	•	Confusion observed between progressive cognitive states

Best Parameters (via GridSearchCV):
- TF-IDF max features: 500  
- SVD components: 50  
- RF Estimators: 100  
- XGBoost Max Depth: 5

## Technologies Used
	•	Python
	•	Jupyter Notebook
	•	Libraries: nltk, spaCy, scikit-learn, transformers, xgboost, matplotlib, seaborn, pandas, shap

## Key Learnings
	•	Linguistic markers can serve as powerful indicators of cognitive decline
	•	Manual feature engineering helps build intuition and interpretability
	•	Classical models perform competitively with small datasets
	•	Transformer models offer promising generalization—pending more data

## Project Status: Suspended
This project is currently on hold due to limited access to large-scale, high-quality labeled datasets. Future continuation will focus on data acquisition, clinical partnerships, or synthetic dataset creation.
