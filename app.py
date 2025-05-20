# Streamlit App for M&A Comparable Transactions Finder (cleaned for deployment)
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import requests
from bs4 import BeautifulSoup

# --- Load the reranker model (only once) ---
import time
import joblib

@st.cache_resource
# --- Apply reranker on initial results ---
def rerank_results(df_matches):
    # Prepare input feature for model
    X_features = df_matches[["Similarity Score"]].values
st.set_page_config(page_title="CMT analiza mnożników pod wycene 🔍", layout="wide")

# --- Constants ---
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 100
EXCEL_PATH = "app_data/Database.xlsx"

# --- Custom styling ---
st.markdown("""
    <style>
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1n76uvr {
        border-color: #80c7ff !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("CMT analiza mnożników pod wycene 🔍")

# --- Load and preprocess the embedded Excel ---
import pickle
@st.cache_data
def load_database():
    if os.path.exists("app_data/embedded_database.pkl"):
        with open("app_data/embedded_database.pkl", "rb") as f:
            return pickle.load(f)

    df = pd.read_excel(EXCEL_PATH)
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        'Business Description\n(Target/Issuer)': 'Business Description',
        'Primary Industry\n(Target/Issuer)': 'Primary Industry'
    })
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(subset=[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry'
    ])
    return df

# --- Scrape website text or fallback to archive ---
def scrape_text(domain):
    try:
        res = requests.get(f"https://{domain}", timeout=4)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except:
        pass
    try:
        archive_url = f"http://web.archive.org/web/{domain}"
        res = requests.get(archive_url, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return soup.get_text(separator=' ', strip=True)
    except:
        return ""
    return ""

# --- Batch embedding using OpenAI ---
def get_embeddings(texts, api_key):
    openai.api_key = api_key
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        try:
            response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
            batch_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"OpenAI API error during batch {i // BATCH_SIZE}: {e}")
            raise
        time.sleep(1)
    return embeddings

# --- Embedding logic with composite text ---
def embed_database(df, api_key):
    df["Website Text"] = df["Web page"].fillna("").apply(scrape_text)
    df["Composite"] = df.apply(lambda row: " ".join(filter(None, [
        str(row["Business Description"]),
        str(row["Primary Industry"]),
        str(row["Website Text"])
    ])), axis=1)
    df["embedding"] = get_embeddings(df["Composite"].tolist(), api_key)
    return df

# --- Find top matches ---
def find_top_matches(df, query, api_key, top_n=10):
    query_embedding = get_embeddings([query], api_key)[0]
    emb_matrix = np.vstack(df["embedding"].values)
    emb_matrix_norm = normalize(emb_matrix)
    query_norm = normalize(np.array(query_embedding).reshape(1, -1))
    similarities = cosine_similarity(query_norm, emb_matrix_norm)[0]
    df["Similarity Score"] = similarities
    top = df.sort_values(by="Similarity Score", ascending=False).head(top_n).copy()
    top["Reason for Match"] = "High semantic + content + industry similarity"
    return top[[
        'Target/Issuer Name', 'MI Transaction ID', 'Implied Enterprise Value/ EBITDA (x)',
        'Business Description', 'Primary Industry', 'Web page', 'Similarity Score', 'Reason for Match'
    ]]

# --- User Interface ---
api_key = st.secrets["openai"]["api_key"]
query_input = st.sidebar.text_area("✏️ Paste company profile here:", height=200)

if api_key and query_input:
    try:
        df = load_database()
        with st.spinner("Embedding and scraping in progress..."):
            df_prepared = embed_database(df, api_key)
        initial_matches = find_top_matches(df_prepared, query_input, api_key)
        results = initial_matches

        st.success("Top matches found:")
        st.dataframe(results, use_container_width=True)
        import csv
        import os
        from datetime import datetime

        feedback_file = "match_feedback.csv"

        st.subheader("📣 Provide Feedback on Match Quality")
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")

# Track rejected matches across reruns
try:
    if "rejected_ids" not in st.session_state:
        st.session_state.rejected_ids = []
except Exception as e:
        st.error(f"⚠️ Error: {e}")    

# Filter out rejected matches
display_df = results[~results["MI Transaction ID"].isin(st.session_state.rejected_ids)]

# Only show one match at a time
if not display_df.empty:
    row = display_df.iloc[0]  # get next best

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{row['Target/Issuer Name']}** — {row['Business Description'][:80]}...")
    with col2:
        feedback_key = f"feedback_{row['MI Transaction ID']}"
        if st.button("👍", key=f"{feedback_key}_up"):
            with open(feedback_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), query_input, row["MI Transaction ID"], "thumbs_up"])
            st.success(f"👍 Feedback saved for: {row['Target/Issuer Name']}")

        if st.button("👎", key=f"{feedback_key}_down"):
            with open(feedback_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), query_input, row["MI Transaction ID"], "thumbs_down"])
            st.session_state.rejected_ids.append(row["MI Transaction ID"])
            st.warning(f"👎 Rejected: {row['Target/Issuer Name']} — showing next best match...")
            st.experimental_rerun()
else:
    st.info("🎉 You've reviewed all matches for this query!") 

# Outside the if-else: export and download
try:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results.to_excel(writer, index=False, sheet_name="Top Matches")
            st.download_button("📥 Download Excel", data=output.getvalue(),
                           file_name="Top_Matches.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as e:
    st.error(f"❌ Error: {e}")

if not api_key or not query_input:
    st.info("👉 Enter your OpenAI API key and a company profile to begin.")