
<div align="center">

<!-- Top Animated Teal Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=008080&height=120&section=header&text=Multi-Source%20Text%20Summarizer%20AI&fontSize=32&fontColor=ffffff&animation=twinkling" width="100%" />

[![Python](https://img.shields.io/badge/Python-3.9%2B-008080?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Website](https://img.shields.io/badge/Portfolio-kerstonanto.in-008080?style=for-the-badge&logo=google-chrome&logoColor=white)](https://kerstonanto.in)

> **A minimal build by Kerston Anto Singh**  
> *Turn hours of reading into seconds of action. Extract instant summaries and sentiment insights across raw text, PDFs, and web links.*

</div>

---

## ⚡ Overview

**Multi-Source Text Summarizer AI** is a lightweight, minimal NLP application built to process and condense unstructured text without losing core context. 

* **3-in-1 Ingestion**: Seamlessly summarize raw text input, multi-page PDF files, or web article URLs.
* **Fast Abstractive Summarization**: Powered by Hugging Face's lightweight `sshleifer/distilbart-cnn-12-6` model for quick local inference.
* **Real-time Sentiment Metrics**: Automatically calculates sentiment polarity for web articles using `TextBlob`.
* **Dynamic Bounds**: Automatically adjusts minimum and maximum summary length based on input size to prevent token clipping.

---


## 🏗️ Architecture & Workflow

```
[ Input Source ] ──> ( Direct Text / PDF Stream / Article URL )
│
▼
[ Ingestion Layer ] ──> ( PyPDF2 / Newspaper3k / String Sanitization )
│
▼
[ NLP Pipeline ] ───> ( DistilBART Transformer + TextBlob Sentiment )
│
▼
[ Presentation ] ───> ( Streamlit Dual-Column Interface )
```

---

## 🛠️ Tech Stack & Dependencies

* **Core Language**: Python 3.9+
* **User Interface**: Streamlit
* **ML & NLP Models**: Hugging Face `transformers`, `nltk`, `textblob`, `newspaper3k`
* **File Parsing**: `PyPDF2`

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/kerston2104/News_Article_Summerizer.git](https://github.com/kerston2104/News_Article_Summerizer.git)
cd News_Article_Summerizer

```

### 2. Set Up Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run Application

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## 💡 Operating Modes

1. **Summarize Text**: Paste raw text directly for quick abstractive summaries.
2. **Summarize Document**: Upload `.pdf` files to extract text, preview content, and generate key takeaways.
3. **URL to Summarize**: Input public news or blog URLs to extract metadata (Title, Author, Date), summaries, and sentiment analysis.

---

## ⚙️ Technical Details

* **Token Safeguards**: Inputs are trimmed to a safe 3,000-character boundary before passing to the model context window ($512$ positional tokens).
* **PDF Ingestion**: Extracts text via standard digital PDF streams (`PyPDF2`). Scanned image PDFs require an external OCR pre-processor.

---

## 👤 Author & Contact

**Kerston Anto Singh**

* Minimal AI/ML Tools & Web Development
* For Production Ready Code Contact.

🌐 **Website**: [kerstonanto.in](https://kerstonanto.in)

**GitHub**: [@kerston2104](https://www.google.com/search?q=https://github.com/kerston2104)

---
