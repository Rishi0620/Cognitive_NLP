import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load model + tokenizer
model_path = "/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/models/bert_cognitive_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

st.title("Cognitive Decline Classifier")
st.markdown("This tool analyzes writing samples for signs of early cognitive decline.")

user_input = st.text_area("Enter a sentence or paragraph:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy().flatten()
            pred = np.argmax(probs)

        st.subheader("🔍 Result:")
        st.write(f"**Prediction:** {'Cognitive Decline' if pred == 1 else 'Healthy'}")
        st.write(f"**Confidence:** {probs[pred]*100:.2f}%")