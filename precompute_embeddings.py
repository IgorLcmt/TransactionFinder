import pandas as pd
import openai
import time
import os
from sklearn.preprocessing import normalize
from bs4 import BeautifulSoup
import requests
import pickle

# Constants
EXCEL_PATH = "app_data/Database.xlsx"
OUTPUT_PATH = "app_data/embedded_database.pkl"
EMBEDDING_MODEL = "text-embedding-ada-002"
BATCH_SIZE = 100
API_KEY = os.getenv("OPENAI_API_KEY")

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

def get_embeddings(texts):
    openai.api_key = API_KEY
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        try:
            response = openai.Embedding.create(input=batch, model=EMBEDDING_MODEL)
            batch_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"OpenAI API error during batch {i // BATCH_SIZE}: {e}")
            raise
        time.sleep(1)
    return embeddings

def main():
    if not API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

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
    df["Website Text"] = df["Web page"].fillna("").apply(scrape_text)
    df["Composite"] = df.apply(lambda row: " ".join(filter(None, [
        str(row["Business Description"]),
        str(row["Primary Industry"]),
        str(row["Website Text"])
    ])), axis=1)
    df["embedding"] = get_embeddings(df["Composite"].tolist())

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(df, f)
    print("✅ Embeddings and scraped text saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()