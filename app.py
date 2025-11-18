import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load model (use your trained one or a pretrained HF model)
MODEL_DIR = "facebook/bart-large-cnn"  # or "./summarizer_model"

@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

summarizer = load_summarizer()

st.title("📝 Text Summarization with Transformers")
st.write("Paste the article below to get an AI-generated summary:")

text = st.text_area("Article text", height=300)

if st.button("Summarize"):
    if text.strip():
        summary = summarizer(text, max_length=128, min_length=30, do_sample=False)
        st.subheader("Summary:")
        st.success(summary[0]["summary_text"])
    else:
        st.warning("Please paste some text first!")
