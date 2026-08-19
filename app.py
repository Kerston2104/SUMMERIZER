import io
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import nltk
from textblob import TextBlob
from newspaper import Article

st.set_page_config(
    page_title="Text Summarizer AI",
    page_icon="📝",
    layout="wide"
)

# Download required NLTK datasets once
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.warning(f"NLTK download notice: {e}")

setup_nltk()

# Load HuggingFace Summarization Model (Bypasses C++ compilation issues)
@st.cache_resource
def load_summary_pipeline():
    # Uses a fast, lightweight summarization model
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    if not text or not text.strip():
        return "Please provide valid text to summarize."
    
    try:
        summarizer = load_summary_pipeline()
        
        # Safely truncate long input text to prevent token length errors
        max_chars = 3000
        safe_text = text[:max_chars] if len(text) > max_chars else text
        
        # Calculate dynamic max/min summary lengths based on text size
        input_length = len(safe_text.split())
        if input_length < 30:
            return safe_text  # Text is already too short to summarize
            
        max_len = min(130, max(30, int(input_length * 0.5)))
        min_len = min(30, max(10, int(input_length * 0.2)))
        
        summary = summarizer(safe_text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {str(e)}"

def extract_text_from_pdf_bytes(pdf_bytes):
    extracted_text = ""
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                extracted_text += content + "\n"
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
    return extracted_text.strip()

st.title("📝 Multi-Source Text Summarizer")

choice = st.sidebar.selectbox("Select Mode", ["Summarize Text", "Summarize Document", "URL to Summarize"])

if choice == "Summarize Text":
    st.subheader("Summarize Direct Text")
    input_text = st.text_area("Enter your text here", height=200)
    
    if st.button("Summarize Text"):
        if input_text and input_text.strip():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Your Input Text**")
                st.info(input_text)
            with col2:
                st.markdown("**Summary Result**")
                with st.spinner("Generating summary..."):
                    result = generate_summary(input_text)
                st.success(result)
        else:
            st.warning("Please enter some text first.")

elif choice == "Summarize Document":
    st.subheader("Summarize PDF Document")
    uploaded_file = st.file_uploader("Upload your document here", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Summarize Document"):
            with st.spinner("Extracting text from PDF..."):
                pdf_bytes = uploaded_file.read()
                extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
            
            if extracted_text:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.info("File uploaded and extracted successfully.")
                    st.markdown("**Extracted Text Preview:**")
                    preview = extracted_text[:1200] + ("..." if len(extracted_text) > 1200 else "")
                    st.text_area("Extracted Preview", preview, height=250, disabled=True)
                
                with col2:
                    st.markdown("**Summary Result**")
                    with st.spinner("Generating document summary..."):
                        doc_summary = generate_summary(extracted_text)
                    st.success(doc_summary)
            else:
                st.error("Could not extract any selectable text from this PDF. It might be scanned or image-only.")

elif choice == "URL to Summarize":
    st.subheader("Summarize Web Article")
    url = st.text_input('Enter article URL here')

    if st.button('Summarize URL'):
        if url and url.strip():
            try:
                with st.spinner("Fetching and parsing URL..."):
                    article = Article(url)
                    article.download()
                    article.parse()
                    article.nlp()

                st.markdown(f'**Title:** {article.title or "N/A"}')
                st.markdown(f'**Author(s):** {", ".join(article.authors) if article.authors else "N/A"}')
                st.markdown(f'**Publishing Date:** {article.publish_date or "N/A"}')
                
                st.subheader("Summary")
                st.success(article.summary if article.summary else "No summary could be generated.")

                if article.text:
                    analysis = TextBlob(article.text)
                    polarity = round(analysis.polarity, 2)
                    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                    st.markdown(f'**Sentiment Analysis:** Polarity ({polarity}) - **{sentiment}**')
            except Exception as e:
                st.error(f"Failed to extract article from URL: {e}")
        else:
            st.warning("Please enter a valid URL.")