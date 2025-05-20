# CMT Streamlit App: Comparable Transactions Finder

This Streamlit app allows users to upload a company profile and find the most similar M&A transactions using OpenAI embeddings and web scraping.

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Data

Place your Excel database at:
```
app_data/Database.xlsx
```

## 🔐 API Key

Store your OpenAI API key in `.streamlit/secrets.toml`:

```toml
[openai]
api_key = "sk-..."
```

## ✅ Features

- Web scraping of target websites
- Text embedding using OpenAI
- Cosine similarity to find top matches