import streamlit as st
from email_classifier_using_roberta import RoBERTaEnhancedClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_classifier():
    clf = RoBERTaEnhancedClassifier()
    model_folder = "/Users/adeshnarayanatellakua/Documents/mini_project/roberta_enhanced_RoBERTa-Enhanced-v1.0-20250930_1718"
    clf.tokenizer = AutoTokenizer.from_pretrained(model_folder)
    clf.model = AutoModelForSequenceClassification.from_pretrained(model_folder).to(clf.device)
    return clf


st.set_page_config("Email Category Classifier (RoBERTa)", page_icon="📧", layout="centered")
st.title("📧 Email Category Classifier (RoBERTa Enhanced)")
st.write("Enter the **subject** and **body** of any email and get instant category prediction with RoBERTa!")

subject = st.text_input("Subject")
body = st.text_area("Body", height=200)

if st.button("Classify"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter subject and/or body to classify.")
    else:
        with st.spinner("Predicting..."):
            clf = load_classifier()
            result = clf.enhanced_classify(subject, body)
        st.success(f"**Prediction:** {result.category} ({result.confidence:.2%} confidence)")
        st.json(result.all_scores)
