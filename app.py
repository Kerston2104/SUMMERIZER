import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="Davi AI - Data Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .stApp { color: #F8FAFC; }
    .chat-bubble-user {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #F8FAFC;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background-color: #0F172A;
        border: 1px solid #3B82F6;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #F8FAFC;
        max-width: 85%;
        margin-right: auto;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 6px 20px;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def check_conversational_intent(query):
    """Handles greetings, identity questions, and general guidance."""
    q = query.lower().strip()
    
    greetings = [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgreetings\b", r"\bgood morning\b", r"\bgood afternoon\b"]
    identity = [r"who are you", r"what is your name", r"what are you", r"tell me about yourself"]
    capabilities = [r"what can you do", r"how to use", r"help", r"options", r"commands", r"what do you do"]
    gratitude = [r"thank", r"thanks", r"awesome", r"great job", r"cool"]
    farewell = [r"\bbye\b", r"\bgoodbye\b", r"\bsee ya\b"]

    for pattern in greetings:
        if re.search(pattern, q):
            return "Hello! I am **Davi AI**, your local data visualization and analytics guide. How can I assist you with your dataset today?"
            
    for pattern in identity:
        if re.search(pattern, q):
            return "I am **Davi AI**—an offline data assistant designed to help you analyze datasets, generate charts, inspect missing values, and calculate summary statistics."
            
    for pattern in capabilities:
        if re.search(pattern, q):
            return (
                "Here is what I can help you with:\n\n"
                "- **Summary Stats:** Ask for `describe`, `shape`, `data types`, or `missing values`.\n"
                "- **Single Column:** Ask for `histogram of Column`, `value counts of Column`, or `boxplot of Column`.\n"
                "- **Relationships:** Ask for `scatter Column1 vs Column2`, `line chart Column1 vs Column2`, or `heatmap`.\n"
                "- **Cleaning:** Ask to `drop missing rows` or `drop missing columns`."
            )

    for pattern in gratitude:
        if re.search(pattern, q):
            return "You're welcome! Let me know if you need any more visualizations or data summaries."

    for pattern in farewell:
        if re.search(pattern, q):
            return "Goodbye! Feel free to upload a new dataset whenever you're ready to analyze more data."

    return None

def generate_kb(df):
    """Builds an expanded knowledge base mapped dynamically to the dataset's columns."""
    cols = df.columns.tolist()
    
    kb = [
        ("dataset shape rows count summary dimensions size total entries", "shape", []),
        ("describe statistics summary overview metrics mean std", "describe", []),
        ("object categorical columns text fields list strings", "object_cols", []),
        ("numeric numerical float int integer metrics columns", "numeric_cols", []),
        ("null missing empty missing values columns list nones", "null_cols", []),
        ("count null missing columns missing values tally", "count_nulls", []),
        ("isnull summary count missing per column breakdown", "isnull_all", []),
        ("data types dtypes schema structure variables types", "dtypes", []),
        ("rows with missing values drop rows corrupt data", "missing_rows", []),
        ("missing values percentage ratio missing proportion", "missing_perc", []),
        ("duplicate rows duplicates identical entries repeated", "duplicates", []),
        ("drop missing rows clean dataset remove missing rows", "drop_missing_rows", []),
        ("drop missing columns clean columns remove missing cols", "drop_missing_cols", []),
        ("show first rows preview head top records sample", "head", []),
        ("show last rows tail bottom end records preview", "tail", []),
        ("pairplot correlation matrix pair plot all features", "pairplot", []),
        ("all columns names list schema features header", "all_cols", [])
    ]

    for col in cols:
        col_clean = col.replace('_', ' ')
        kb.extend([
            (f"value counts distribution unique frequency count for {col_clean}", "val_counts", [col]),
            (f"unique values distinct list for {col_clean}", "unique_vals", [col]),
            (f"histogram count plot distribution frequency histogram for {col_clean}", "hist", [col]),
            (f"box plot boxplot outliers spread range distribution for {col_clean}", "box_single", [col]),
            (f"density kde distribution distplot curve for {col_clean}", "kde", [col]),
            (f"ecdf cumulative distribution plot for {col_clean}", "ecdf", [col])
        ])

    for c1 in cols:
        for c2 in cols:
            if c1 != c2:
                c1_c = c1.replace('_', ' ')
                c2_c = c2.replace('_', ' ')
                kb.extend([
                    (f"scatter plot scatter relationship correlation points {c1_c} vs {c2_c}", "scatter", [c1, c2]),
                    (f"line plot line chart trend time series {c1_c} vs {c2_c}", "line", [c1, c2]),
                    (f"bar plot bar chart counts comparison {c1_c} vs {c2_c}", "bar", [c1, c2]),
                    (f"violin plot probability density distribution {c1_c} vs {c2_c}", "violin", [c1, c2]),
                    (f"box plot boxplot categorized comparison {c1_c} vs {c2_c}", "box_two", [c1, c2]),
                    (f"heatmap correlation matrix heat map {c1_c} and {c2_c}", "heatmap", [c1, c2])
                ])

    queries = [item[0] for item in kb]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english').fit(queries)
    vectors = vectorizer.transform(queries)
    
    return kb, vectorizer, vectors

def query_engine(user_query, kb, vectorizer, vectors):
    q_vec = vectorizer.transform([user_query.lower()])
    scores = cosine_similarity(q_vec, vectors)[0]
    best_idx = np.argmax(scores)
    
    if scores[best_idx] < 0.15:
        return None, 0.0
    return kb[best_idx], scores[best_idx]

def process_command(df, cmd, args):
    fig = None
    text_resp = ""
    data_out = None

    if cmd == "shape":
        text_resp = f"The dataset contains **{df.shape[0]:,}** rows and **{df.shape[1]}** columns."
    elif cmd == "describe":
        text_resp = "Here is the statistical summary of numerical features:"
        data_out = df.describe()
    elif cmd == "object_cols":
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        text_resp = f"Found **{len(cols)}** categorical/text columns:\n`" + "`, `".join(cols) + "`" if cols else "No categorical columns found."
    elif cmd == "numeric_cols":
        cols = df.select_dtypes(include=np.number).columns.tolist()
        text_resp = f"Found **{len(cols)}** numerical columns:\n`" + "`, `".join(cols) + "`" if cols else "No numerical columns found."
    elif cmd == "null_cols":
        cols = df.columns[df.isnull().any()].tolist()
        text_resp = f"Columns containing missing values (**{len(cols)}**):\n`" + "`, `".join(cols) + "`" if cols else "No columns have missing values."
    elif cmd == "count_nulls":
        count = df.columns[df.isnull().any()].shape[0]
        text_resp = f"There are **{count}** columns with missing values."
    elif cmd == "isnull_all":
        text_resp = "Missing values per column:"
        data_out = df.isnull().sum().to_frame("Missing Count")
    elif cmd == "dtypes":
        text_resp = "Dataset schema and data types:"
        data_out = df.dtypes.astype(str).to_frame("Data Type")
    elif cmd == "missing_rows":
        data_out = df[df.isnull().any(axis=1)]
        text_resp = f"Retrieved **{len(data_out)}** rows containing missing values."
    elif cmd == "missing_perc":
        text_resp = "Percentage of missing records per feature:"
        data_out = (df.isnull().mean() * 100).round(2).to_frame("Missing (%)")
    elif cmd == "duplicates":
        data_out = df[df.duplicated()]
        text_resp = f"Found **{len(data_out)}** exact duplicate records."
    elif cmd == "drop_missing_rows":
        st.session_state.df = df.dropna()
        text_resp = f"Dropped rows with null values. New row count: **{len(st.session_state.df):,}**."
    elif cmd == "drop_missing_cols":
        st.session_state.df = df.dropna(axis=1)
        text_resp = f"Dropped columns with null values. Remaining columns: **{st.session_state.df.shape[1]}**."
    elif cmd == "head":
        text_resp = "Previewing top 5 rows:"
        data_out = df.head()
    elif cmd == "tail":
        text_resp = "Previewing bottom 5 rows:"
        data_out = df.tail()
    elif cmd == "all_cols":
        text_resp = "Columns present in dataset:\n`" + "`, `".join(df.columns.tolist()) + "`"
    elif cmd == "val_counts":
        text_resp = f"Top value frequencies for **{args[0]}**:"
        data_out = df[args[0]].value_counts().head(20).to_frame("Frequency")
    elif cmd == "unique_vals":
        vals = df[args[0]].unique()
        text_resp = f"Column **{args[0]}** has **{len(vals)}** unique values."
        data_out = pd.DataFrame(vals[:50], columns=[f"Unique {args[0]} (Top 50)"])
    
    # Plot Generation
    plt.style.use("dark_background")
    f, ax = plt.subplots(figsize=(8, 4.5))
    f.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')

    if cmd == "scatter":
        sns.scatterplot(data=df, x=args[0], y=args[1], ax=ax, color="#3B82F6", alpha=0.7)
        ax.set_title(f"{args[0]} vs {args[1]}", color="#F8FAFC")
        fig = f
        text_resp = f"Generated scatter plot for **{args[0]}** against **{args[1]}**."
    elif cmd == "line":
        sns.lineplot(data=df, x=args[0], y=args[1], ax=ax, color="#10B981")
        ax.set_title(f"Trend: {args[0]} vs {args[1]}", color="#F8FAFC")
        fig = f
        text_resp = f"Generated line plot for **{args[0]}** vs **{args[1]}**."
    elif cmd == "bar":
        sns.barplot(data=df, x=args[0], y=args[1], ax=ax, palette="mako")
        plt.xticks(rotation=45)
        fig = f
        text_resp = f"Generated bar chart comparing **{args[0]}** and **{args[1]}**."
    elif cmd == "violin":
        sns.violinplot(data=df, x=args[0], y=args[1], ax=ax, palette="viridis")
        fig = f
        text_resp = f"Generated violin plot showing distribution of **{args[1]}** across **{args[0]}**."
    elif cmd == "box_two":
        sns.boxplot(data=df, x=args[0], y=args[1], ax=ax, palette="rocket")
        fig = f
        text_resp = f"Generated box plot for **{args[1]}** grouped by **{args[0]}**."
    elif cmd == "box_single":
        sns.boxplot(data=df, y=args[0], ax=ax, color="#8B5CF6")
        fig = f
        text_resp = f"Generated distribution box plot for **{args[0]}**."
    elif cmd == "heatmap":
        corr = df[[args[0], args[1]]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=False)
        fig = f
        text_resp = f"Correlation matrix heatmap between **{args[0]}** and **{args[1]}**."
    elif cmd == "hist":
        sns.histplot(df[args[0]], ax=ax, color="#F59E0B", kde=True)
        fig = f
        text_resp = f"Distribution histogram for **{args[0]}**."
    elif cmd == "kde":
        sns.kdeplot(df[args[0]], ax=ax, color="#EC4899", fill=True)
        fig = f
        text_resp = f"Kernel Density Estimate (KDE) plot for **{args[0]}**."
    elif cmd == "ecdf":
        sns.ecdfplot(df[args[0]], ax=ax, color="#06B6D4")
        fig = f
        text_resp = f"Empirical Cumulative Distribution function plot for **{args[0]}**."
    elif cmd == "pairplot":
        plt.close(f)
        pair_fig = sns.pairplot(df.select_dtypes(include=np.number).dropna())
        fig = pair_fig.fig
        text_resp = "Generated pairwise relationship matrix across numerical features."

    if fig is None:
        plt.close(f)

    return text_resp, data_out, fig

def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'df' not in st.session_state:
        st.session_state.df = None

    with st.sidebar:
        st.title("⚙️ Data Engine")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if st.session_state.df is not None:
            st.divider()
            st.subheader("📌 Dataset Metrics")
            df_curr = st.session_state.df
            col_a, col_b = st.columns(2)
            col_a.metric("Rows", f"{df_curr.shape[0]:,}")
            col_b.metric("Cols", f"{df_curr.shape[1]}")
            st.metric("Missing Cells", f"{df_curr.isnull().sum().sum():,}")
            
            if st.button("Reset Dataset"):
                st.session_state.df = load_data(uploaded_file)
                st.rerun()

    st.title("💬 Davi AI - Data Assistant")
    st.caption("Ask questions about your data, generate plots, or request summaries in natural language.")

    # Greeting fallback when no file is uploaded
    if uploaded_file is None:
        if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "bot",
                "text": "Hello! I am **Davi AI**. Upload a CSV file in the sidebar to get started, or ask me what I can do!",
                "data": None,
                "fig": None
            })

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 <b>Davi AI:</b> {msg["text"]}</div>', unsafe_allow_html=True)

        user_chat = st.chat_input("Say hi or ask Davi AI a question...")
        if user_chat:
            st.session_state.messages.append({"role": "user", "text": user_chat, "data": None, "fig": None})
            
            chat_response = check_conversational_intent(user_chat)
            if not chat_response:
                chat_response = "I am ready to help you analyze your data! Please upload a CSV file in the sidebar so I can generate charts and statistical summaries for you."
            
            st.session_state.messages.append({"role": "bot", "text": chat_response, "data": None, "fig": None})
            st.rerun()

    else:
        if st.session_state.df is None or st.sidebar.button("Re-load Original Data"):
            st.session_state.df = load_data(uploaded_file)
            welcome_msg = f"Successfully parsed **{uploaded_file.name}**. I am ready to analyze your dataset! What would you like to explore?"
            st.session_state.messages = [{
                "role": "bot",
                "text": welcome_msg,
                "data": None,
                "fig": None
            }]

        df = st.session_state.df
        kb, vectorizer, vectors = generate_kb(df)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 <b>Davi AI:</b> {msg["text"]}</div>', unsafe_allow_html=True)
                if msg.get("data") is not None:
                    st.dataframe(msg["data"], use_container_width=True)
                if msg.get("fig") is not None:
                    st.pyplot(msg["fig"])

        user_input = st.chat_input("Ask Davi AI (e.g., 'scatter Age vs Income', 'summary', 'show head')...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "text": user_input, "data": None, "fig": None})

            # Check conversational intents first
            conv_response = check_conversational_intent(user_input)
            if conv_response:
                st.session_state.messages.append({"role": "bot", "text": conv_response, "data": None, "fig": None})
            else:
                # Fall back to dataset search engine
                match_tuple, confidence = query_engine(user_input, kb, vectorizer, vectors)

                if match_tuple and confidence > 0.15:
                    _, cmd, args = match_tuple
                    text_out, data_out, fig_out = process_command(df, cmd, args)
                    
                    st.session_state.messages.append({
                        "role": "bot",
                        "text": text_out,
                        "data": data_out,
                        "fig": fig_out
                    })
                else:
                    fallback_msg = "I couldn't match your query to a dataset feature or plot command. Try asking for `dataset shape`, `missing values`, `histograms`, or `scatter plots`."
                    st.session_state.messages.append({
                        "role": "bot",
                        "text": fallback_msg,
                        "data": None,
                        "fig": None
                    })
            st.rerun()

if __name__ == "__main__":
    main()