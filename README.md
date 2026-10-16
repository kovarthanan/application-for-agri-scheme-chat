# 🌾 Uzhavan Thunai — Tamil Nadu Farmer Schemes Assistant

**உழவன் துணை** ("Farmer's companion") is a Streamlit app that helps Tamil Nadu farmers find government agriculture schemes and ask questions about them in simple words.

It shows what is available at a glance (scheme counts, highlighted schemes, a searchable list) and has an always-visible chat assistant that answers from official scheme data. Everything runs **locally on your laptop** with [Ollama](https://ollama.com), with no paid API keys.

> Data source: [tn.gov.in – Schemes](https://www.tn.gov.in/schemes.php) (Agriculture – Farmers Welfare and Co-operation departments).

---

## ✨ Features

- **Dashboard**: total farmer schemes, schemes by benefit type (subsidy / grants), departments
- **5 highlighted schemes** with funding details and a one-tap "Ask about this"
- **Search and filter** all schemes by keyword and benefit type
- **Chat assistant** (RAG) that answers only from scheme data, with links to the official pages
- Farmer-friendly theme, works on mobile (chat comes first on phones)
- Industry / MSME schemes that the site lists under Agriculture are filtered out

---

## 🏗️ How it works

```
tn.gov.in ──(Playwright scraper)──► data/tn_gov_schemes.json
                                          │
                     split into chunks + embeddings (nomic-embed-text)
                                          │
                                   FAISS vector index
                                          │
           question ─► similarity search ─► top chunks ─► LLM (qwen2.5) ─► answer
                                          │
                                   Streamlit UI (app.py)
```

| Step | What | File |
|------|------|------|
| 1 | Scrape scheme details with Playwright | `scraper/tn_gov.py` |
| 2 | Build the FAISS vector index from the scraped JSON | inside `app.py` (built automatically on first question) |
| 3 | Answer questions with a local LLM via Ollama | `app.py` |
| — | Optional command-line version of the RAG chat | `agri_rag_app.py` |

---

## 📁 Project structure

```
.
├── app.py                    # Streamlit app (dashboard + chat)
├── agri_rag_app.py           # Optional: terminal chat version
├── scraper/
│   └── tn_gov.py             # Playwright scraper for tn.gov.in
├── data/
│   └── tn_gov_schemes.json   # Scraped schemes (created by the scraper)
├── .streamlit/
│   └── config.toml           # App theme colours
├── requirements.txt
└── README.md
```

---

## 🚀 Getting started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

### 2. Install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m playwright install chromium
```

### 3. Download the AI models

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

### 4. Get the scheme data
`data/tn_gov_schemes.json` is included. To refresh it from the website:

```bash
python scraper/tn_gov.py --list-depts     # see available departments
python scraper/tn_gov.py                  # scrape agriculture-related departments
python scraper/tn_gov.py --dept "Animal"  # add a specific department
```

After re-scraping, delete the `faiss_index_schemes/` folder so the index is rebuilt.

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The **first question** builds the vector index, so it takes longer; later answers are faster.

---

## ⚙️ Configuration

Settings are at the top of `app.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `LLM_MODEL` | `qwen2.5:1.5b` | Answer model. Use `qwen2.5:0.5b` on slow laptops |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `INCLUDE_PDFS` | `False` | Also index the large scheme PDFs (slower) |
| `HIGHLIGHT_KEYWORDS` | paddy, pipes, tractor… | Which schemes appear as highlights |
| `SAMPLE_QUESTIONS` | 4 questions | Quick questions shown in the chat |

Optional: copy `.env.example` to `.env` to enable LangSmith tracing.

---

## 💬 Example questions

- What subsidy is there for water pipes?
- How can I get a loan for a tractor?
- Which schemes help paddy farmers?
- நெல் விதைக்கு மானியம் உண்டா?

---

## ⚠️ Limitations

- Answers are only as complete as the scraped data. Schemes from departments not scraped (for example Animal Husbandry) won't be found until you scrape them.
- A small local model can make mistakes. Always confirm on the official tn.gov.in page linked under each answer.
- Speed depends on your hardware; on CPU-only laptops, answers can take a while.

---

## 🙏 Acknowledgements

- Scheme information © Government of Tamil Nadu ([tn.gov.in](https://www.tn.gov.in))
- Built with [Streamlit](https://streamlit.io), [LangChain](https://www.langchain.com), [FAISS](https://github.com/facebookresearch/faiss), [Ollama](https://ollama.com) and [Playwright](https://playwright.dev)

This is an independent learning project and is not an official Government of Tamil Nadu application.
