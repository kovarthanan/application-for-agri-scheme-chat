import json
import re
import time
from collections import Counter
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # optional: LangSmith keys from .env

# ---------- Settings ----------
APP_NAME = "Uzhavan Thunai"                 # உழவன் துணை = "Farmer's companion"
SCHEMES_FILE = "data/tn_gov_schemes.json"  # output of scraper/tn_gov.py
INDEX_FOLDER = "faiss_index_schemes"
INCLUDE_PDFS = False                       # True = also index the big scheme PDFs (much slower)
LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50

SKIP_BENEFICIARIES = {"manufacturers"}     # MSME schemes listed under Agriculture
SKIP_PDF_WORDS = ("msme",)

# Shown first in "Highlighted schemes" (matched against the title, in this order)
HIGHLIGHT_KEYWORDS = ["quality paddy", "pipes carrying water", "tractor", "soil health", "micro nutrient"]

SAMPLE_QUESTIONS = [
    "What subsidy is there for water pipes?",
    "How can I get a loan for a tractor?",
    "Which schemes help paddy farmers?",
    "Is there free training for farmers?",
]

st.set_page_config(page_title=f"{APP_NAME} · TN Farmer Schemes", page_icon="🌾", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------- Look ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Mukta:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@500;700&display=swap');

    :root {
        --paddy: #2E6B34;      /* deep paddy green */
        --leaf: #6FA54A;       /* young leaf */
        --harvest: #E8B84A;    /* ripe grain */
        --soil: #7A4E2D;       /* red soil */
        --cream: #FBF6EA;      /* dried straw */
        --ink: #2B2A22;
        --muted: #6E6A5C;
        --line: #E6DCC3;
    }

    #MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"] {visibility: hidden;}
    .stApp, [data-testid="stHeader"], [data-testid="stBottom"] > div {background: var(--cream);}
    .block-container {max-width: 1420px; padding-top: 3.2rem;}
    html, body, [class*="css"], .stMarkdown, p, li {font-family: 'Mukta', sans-serif; color: var(--ink);}

    /* ---- hero banner ---- */
    .hero {
        position: relative; overflow: hidden;
        border-radius: 22px; padding: 1.5rem 1.8rem 3.8rem 1.8rem;
        background: linear-gradient(160deg, #FFF3CF 0%, #FBE7A9 45%, #F5D57A 100%);
        border: 1px solid #EBD38E;
    }
    .hero .sun {
        position: absolute; right: 36px; top: 22px; width: 70px; height: 70px; border-radius: 50%;
        background: radial-gradient(circle, #FFD45C 0%, #F4A824 70%);
        box-shadow: 0 0 0 14px rgba(255, 212, 92, .25), 0 0 0 30px rgba(255, 212, 92, .12);
    }
    .hero .greet {font-family: 'Noto Sans Tamil', sans-serif; font-weight: 700; color: var(--soil); font-size: 1.05rem;}
    .hero h1 {
        font-family: 'Merriweather', Georgia, serif; font-weight: 900;
        color: var(--paddy); font-size: 2.1rem; line-height: 1.15; margin: .25rem 0 .5rem 0;
    }
    .hero p {color: #5A4A2A; font-size: 1.05rem; max-width: 520px; margin: 0;}
    .hero .fields {position: absolute; left: 0; right: 0; bottom: -2px; height: 70px;}

    /* ---- stat cards ---- */
    .stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 1.2rem 0 .4rem;}
    .stat {
        background: #FFFDF6; border: 1px solid var(--line); border-radius: 18px;
        padding: 1rem 1.1rem; border-bottom: 5px solid var(--leaf);
    }
    .stat.harvest {border-bottom-color: var(--harvest);}
    .stat.soil {border-bottom-color: var(--soil);}
    .stat.paddy {border-bottom-color: var(--paddy);}
    .stat .num {font-family: 'Merriweather', serif; font-size: 2.1rem; font-weight: 900; color: var(--paddy); line-height: 1;}
    .stat .lbl {color: var(--muted); font-size: .95rem; margin-top: .35rem;}
    .stat .icon {float: right; font-size: 1.6rem;}

    /* ---- section headings ---- */
    .section {font-family: 'Merriweather', serif; font-weight: 700; color: var(--soil); font-size: 1.35rem; margin: 1.6rem 0 .2rem;}
    .section-sub {color: var(--muted); margin-bottom: .8rem;}

    /* ---- highlighted scheme cards ---- */
    .card {
        background: #FFFFFF; border: 1px solid var(--line); border-radius: 18px;
        padding: 1rem 1.1rem; height: 255px; margin-bottom: .5rem;
        box-shadow: 0 2px 0 rgba(122, 78, 45, .06);
        display: flex; flex-direction: column; gap: .45rem;
    }
    .card .tag {
        align-self: flex-start; font-size: .78rem; font-weight: 600; letter-spacing: .02em;
        padding: .12rem .6rem; border-radius: 999px; background: #EAF3E1; color: var(--paddy);
    }
    .card .tag.Loans, .card .tag.Loan {background: #F6E7D8; color: var(--soil);}
    .card .tag.Grants, .card .tag.Incentives {background: #FCF0CC; color: #8A6410;}
    .card > * {flex-shrink: 0;}
    .card .ttl, .card .money, .card .how {display: -webkit-box; -webkit-box-orient: vertical; overflow: hidden; margin: 0;}
    .card .ttl {font-family: 'Mukta', sans-serif; font-weight: 700; font-size: 1.05rem; line-height: 1.3; color: var(--ink); -webkit-line-clamp: 2;}
    .card .money {font-weight: 600; color: var(--paddy); font-size: .92rem; line-height: 1.35; -webkit-line-clamp: 2;}
    .card .how {color: var(--muted); font-size: .85rem; line-height: 1.35; -webkit-line-clamp: 2;}
    .card a {margin-top: auto; color: var(--soil); font-weight: 600; text-decoration: none; font-size: .9rem;}

    /* ---- permanent chat panel (right column) ---- */
    [data-testid="stColumn"]:has(.chat-marker) {
        position: sticky; top: 3.4rem; align-self: flex-start;
        background: #FFFDF6; border: 1px solid var(--line); border-radius: 22px;
        padding: 1rem 1rem .6rem; box-shadow: 0 6px 24px rgba(122, 78, 45, .08);
    }
    .chat-head {display: flex; align-items: center; gap: .7rem; padding-bottom: .6rem;
                border-bottom: 1px dashed var(--line); margin-bottom: .4rem;}
    .chat-head .avatar {width: 44px; height: 44px; border-radius: 50%; background: #EAF3E1;
                        display: grid; place-items: center; font-size: 1.4rem;}
    .chat-head .name {font-family: 'Merriweather', serif; font-weight: 700; color: var(--paddy); font-size: 1.1rem;}
    .chat-head .sub {color: var(--muted); font-size: .85rem; line-height: 1.2;}
    .chat-head .dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--leaf); margin-right: 4px;}
    .welcome {background: #F2F8EC; border: 1px solid #D7E8C6; border-radius: 16px; padding: .8rem 1rem;
              color: var(--ink); font-size: .95rem; margin-bottom: .6rem;}
    .welcome b {color: var(--paddy);}

    /* ---- widgets ---- */
    [data-testid="stChatInput"] {border-radius: 20px; border: 2px solid var(--leaf); background: #FFFFFF;}
    [data-testid="stChatInput"] textarea {font-size: 1.05rem;}
    [data-testid="stChatMessage"] {background: #FFFDF6; border: 1px solid var(--line); border-radius: 16px;}
    section[data-testid="stSidebar"] {background: #F3EBD6; border-right: 1px solid var(--line);}
    .stButton > button {
        border-radius: 14px; border: 1px solid #D8CBA5; background: #FFFDF6; width: 100%;
        color: var(--ink); font-weight: 500;
    }
    .stButton > button:hover {border-color: var(--paddy); color: var(--paddy); background: #F2F8EC;}
    [data-testid="stExpander"] {background: #FFFDF6; border-radius: 14px;}

    @media (max-width: 1000px) {
        [data-testid="stColumn"]:has(.chat-marker) {position: static;}
    }
    @media (max-width: 760px) {
        [data-testid="stColumn"]:has(.chat-marker) {order: -1;}   /* phones: chat first */
        .stats {grid-template-columns: repeat(2, 1fr);}
        .hero h1 {font-size: 1.8rem;}
        .hero .sun {width: 56px; height: 56px; right: 20px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Rows of crops along the bottom of the banner (inline SVG)
FIELDS_SVG = """
<svg class="fields" viewBox="0 0 1000 70" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M0 40 Q 250 10 500 35 T 1000 30 V70 H0Z" fill="#6FA54A"/>
  <path d="M0 52 Q 300 30 600 50 T 1000 45 V70 H0Z" fill="#2E6B34"/>
  <g stroke="#2E6B34" stroke-width="3" stroke-linecap="round">
    """ + "".join(
        f'<path d="M{x} 42 q -4 -12 0 -22 M{x} 42 q 6 -10 8 -18" fill="none"/>'
        for x in range(20, 1000, 38)
    ) + """
  </g>
</svg>
"""


def html(block):
    """Render raw HTML; flattened so Markdown never treats indented lines as code."""
    st.markdown("".join(line.strip() for line in block.splitlines()), unsafe_allow_html=True)


# ---------- Scheme data ----------
def is_farmer_scheme(s):
    fields = s.get("fields", {})
    is_msme = any(w in p.get("path", "").lower() for p in s.get("pdfs", []) for w in SKIP_PDF_WORDS)
    return not is_msme and fields.get("Beneficiaries", "").strip().lower() not in SKIP_BENEFICIARIES


@st.cache_data
def load_schemes():
    path = Path(SCHEMES_FILE)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [s for s in data if is_farmer_scheme(s) and s.get("content", "").strip()]


def short(text, n=150):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


# long co-operative bank names -> short display names
TITLE_SHORTCUTS = {
    "Primary Cooperative Agriculture and Rural Development Bank - ": "Co-op Agri Bank loan: ",
    "District central Cooperative Banks and through its branches - ": "District Co-op Bank loan: ",
    "Large sized multi purpose Coop.Society ": "Co-op Society: ",
}


def display_title(title):
    for long, sh in TITLE_SHORTCUTS.items():
        title = title.replace(long, sh)
    return title


def pick_highlights(schemes, n=5):
    """Keyword favourites first, then the most detailed schemes, one per title."""
    picked = []
    for kw in HIGHLIGHT_KEYWORDS:
        for s in schemes:
            if kw in s["title"].lower() and s not in picked:
                picked.append(s)
                break
    def detail(s):
        f = s.get("fields", {})
        return bool(f.get("Funding Pattern")) * 2 + bool(f.get("How To avail")) + min(len(s["content"]), 1500) / 1500
    for s in sorted(schemes, key=detail, reverse=True):
        if len(picked) >= n:
            break
        if s not in picked:
            picked.append(s)
    return picked[:n]


# ---------- Models (loaded once) ----------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, keep_alive=1800)


@st.cache_resource
def get_chain():
    llm = ChatOllama(
        model=LLM_MODEL, temperature=0,
        num_ctx=2048,       # smaller context window = faster on CPU
        num_predict=300,    # cap answer length
        keep_alive=1800,    # keep the model in memory for 30 min between questions
    )
    prompt = ChatPromptTemplate.from_template(
        """You help Tamil Nadu farmers understand government agriculture schemes.
Answer the question using only the context below.
Mention the scheme name, the subsidy / benefit amount and whom to apply to, when available.
If the answer is not in the context, say "I don't know based on the scheme data."
Use simple words. Keep the answer short. Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
    )
    return prompt | llm | StrOutputParser()


# ---------- Index ----------
def load_index():
    if Path(INDEX_FOLDER).exists():
        return FAISS.load_local(INDEX_FOLDER, get_embeddings(), allow_dangerous_deserialization=True)
    return None


def build_index(schemes):
    docs, pdf_paths = [], set()
    for s in schemes:
        f = s.get("fields", {})
        docs.append(Document(
            page_content=s["content"],
            metadata={
                "type": "scheme", "title": s["title"], "source": s.get("source_url", ""),
                "department": s.get("department", ""), "benefit_type": f.get("Types of Benefits", ""),
            },
        ))
        pdf_paths.update(p["path"] for p in s.get("pdfs", []))

    if INCLUDE_PDFS:
        for pdf in sorted(pdf_paths):
            if Path(pdf).exists():
                for page in PyPDFLoader(pdf).load():
                    page.metadata["type"] = "pdf"
                    docs.append(page)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    for c in chunks:  # keep the scheme name on every piece of a long scheme
        if c.metadata.get("type") == "scheme" and not c.page_content.startswith("Concerned Department"):
            c.page_content = f"Scheme: {c.metadata['title']}\n{c.page_content}"
    if not chunks:
        st.error("No scheme text found to index.")
        return None

    bar = st.progress(0, text="Preparing scheme knowledge (first time only)...")
    store = None
    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            if store is None:
                store = FAISS.from_documents(batch, get_embeddings())
            else:
                store.add_documents(batch)
            done = min(i + BATCH_SIZE, len(chunks))
            bar.progress(done / len(chunks), text=f"Reading schemes {done}/{len(chunks)}")
        store.save_local(INDEX_FOLDER)
    finally:
        bar.empty()
    return store


def get_store(schemes):
    if st.session_state.store is None:
        st.session_state.store = load_index() or build_index(schemes)
    return st.session_state.store


# ---------- Session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = None
if "pending" not in st.session_state:
    st.session_state.pending = None

schemes = load_schemes()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"### 🌾 {APP_NAME}")
    st.caption("உழவன் துணை · Tamil Nadu farmer schemes assistant. Runs fully on your laptop.")

    top_k = st.slider("Schemes to read per answer", 1, 8, 3)

    if st.button("🔄 Rebuild scheme index"):
        with st.spinner("Rebuilding..."):
            st.session_state.store = build_index(schemes)
        if st.session_state.store:
            st.toast("Scheme index updated", icon="✅")

    st.markdown("---")
    st.caption(f"Data: tn.gov.in · {len(schemes)} farmer schemes")

if not schemes:
    st.error(f"No scheme data found at `{SCHEMES_FILE}`. Run `python scraper/tn_gov.py` first.")
    st.stop()

left, right = st.columns([1.55, 1], gap="large")

# =====================  LEFT: what is available (always visible)  =====================
with left:
    html(
        f"""
        <div class="hero">
          <div class="sun"></div>
          <div class="greet">வணக்கம் விவசாயி! 🙏</div>
          <h1>Find the right scheme<br>for your farm</h1>
          <p>Tamil Nadu government schemes for farmers — subsidies, loans, seeds and training —
             in one place. Ask in simple words.</p>
          {FIELDS_SVG}
        </div>
        """
    )

    types = Counter(s.get("fields", {}).get("Types of Benefits", "Other") or "Other" for s in schemes)
    depts = {s.get("department", "") for s in schemes}
    top_types = types.most_common(2) + [("Other", 0)] * 2
    icons = {"Subsidy": "💰", "Loans": "🏦", "Loan": "🏦", "Grants": "🎁", "Incentives": "⭐"}
    html(
        f"""
        <div class="stats">
          <div class="stat paddy"><span class="icon">🌾</span><div class="num">{len(schemes)}</div>
               <div class="lbl">Schemes for farmers</div></div>
          <div class="stat harvest"><span class="icon">{icons.get(top_types[0][0], "📌")}</span>
               <div class="num">{top_types[0][1]}</div><div class="lbl">{top_types[0][0]} schemes</div></div>
          <div class="stat soil"><span class="icon">{icons.get(top_types[1][0], "📌")}</span>
               <div class="num">{top_types[1][1]}</div><div class="lbl">{top_types[1][0]} schemes</div></div>
          <div class="stat"><span class="icon">🏛️</span><div class="num">{len(depts)}</div>
               <div class="lbl">Government departments</div></div>
        </div>
        """
    )

    # ---- 5 highlighted schemes ----
    st.markdown('<div class="section">⭐ Highlighted schemes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Popular support farmers can apply for today. '
                'Tap <b>Ask about this</b> to get details in the chat.</div>', unsafe_allow_html=True)
    highlights = pick_highlights(schemes)
    for row_start in range(0, len(highlights), 3):
        cols = st.columns(3)
        for col, s in zip(cols, highlights[row_start:row_start + 3]):
            f = s.get("fields", {})
            btype = f.get("Types of Benefits", "Scheme")
            money = f.get("Funding Pattern") or f.get("Description", "")
            how = f.get("How To avail", "")
            with col:
                html(
                    f"""
                    <div class="card">
                      <span class="tag {btype}">{btype}</span>
                      <div class="ttl">{short(display_title(s['title']), 70)}</div>
                      <div class="money">💵 {short(money, 90)}</div>
                      <div class="how">{'📝 ' + short(how, 80) if how else ''}</div>
                      <a href="{s.get('source_url', '#')}" target="_blank">View on tn.gov.in →</a>
                    </div>
                    """
                )
                if st.button("💬 Ask about this", key=f"hl_{s['id']}"):
                    st.session_state.pending = f"Tell me about the scheme '{s['title']}'. Who can apply and how?"

    # ---- browse all ----
    st.markdown('<div class="section">📋 All schemes</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    search = c1.text_input("Search schemes", placeholder="🔎 Search: seed, pipe, tractor, loan…",
                           label_visibility="collapsed")
    type_filter = c2.selectbox("Benefit type", ["All types"] + sorted(types), label_visibility="collapsed")
    shown = [
        s for s in schemes
        if (not search or search.lower() in s["content"].lower())
        and (type_filter == "All types" or s.get("fields", {}).get("Types of Benefits", "Other") == type_filter)
    ]
    show_all = bool(search) or type_filter != "All types" or st.toggle(f"Show all {len(shown)} schemes")
    visible = shown if show_all else shown[:6]
    st.caption(f"Showing {len(visible)} of {len(schemes)} schemes")
    for s in visible:
        f = s.get("fields", {})
        with st.expander(f"🌱 {display_title(s['title'])}  ·  {f.get('Types of Benefits', '')}"):
            for key in ("Funding Pattern", "Beneficiaries", "Description", "How To avail", "Concerned Department"):
                if f.get(key):
                    st.markdown(f"**{key}:** {f[key]}")
            b1, b2 = st.columns([1, 1])
            b1.markdown(f"[Open on tn.gov.in]({s.get('source_url', '#')})")
            if b2.button("💬 Ask about this", key=f"all_{s['id']}"):
                st.session_state.pending = f"Tell me about the scheme '{s['title']}'. Who can apply and how?"

# =====================  RIGHT: permanent chat panel  =====================
with right:
    html(
        f"""
        <div class="chat-marker"></div>
        <div class="chat-head">
          <div class="avatar">🌾</div>
          <div><div class="name">Ask {APP_NAME}</div>
               <div class="sub"><span class="dot"></span>Answers from {len(schemes)} TN farmer schemes</div></div>
        </div>
        """
    )
    box = st.container(height=470, border=False)
    question = st.chat_input("Type your question… (English or தமிழ்)")
    if st.session_state.pending and not question:
        question, st.session_state.pending = st.session_state.pending, None

    with box:
        if not st.session_state.messages and not question:
            html(
                """
                <div class="welcome">👋 <b>Vanakkam!</b> Ask me about any farmer scheme — subsidy,
                loan, seeds, irrigation or training. Try one of these:</div>
                """
            )
            for q in SAMPLE_QUESTIONS:
                if st.button(q, key=f"q_{q}"):
                    st.session_state.pending = q
                    st.rerun()

        # history
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="🌾" if m["role"] == "assistant" else "🧑‍🌾"):
                st.markdown(m["content"])
                if m.get("sources"):
                    with st.expander("Schemes used"):
                        for src in m["sources"]:
                            st.markdown(f"- {src}")

        # new question
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑‍🌾"):
                st.markdown(question)

            with st.chat_message("assistant", avatar="🌾"):
                start = time.time()
                status = st.status("Looking through the schemes...", expanded=False)
                try:
                    store = get_store(schemes)
                    if store is None:
                        st.stop()
                    docs = store.similarity_search(question, k=top_k)
                    context = "\n\n".join(d.page_content for d in docs)
                    status.update(label=f"Found {len(docs)} matching parts. Thinking...")

                    def stream_answer():
                        first = True
                        for token in get_chain().stream({"context": context, "question": question}):
                            if first:
                                status.update(label="Writing answer...")
                                first = False
                            yield token
                        status.update(label=f"Done in {time.time() - start:.0f}s", state="complete",
                                      expanded=False)

                    answer = st.write_stream(stream_answer())
                except Exception as e:
                    status.update(label="Could not reach the AI model", state="error")
                    st.session_state.messages.pop()
                    st.error(
                        "Ollama is not running or the models are missing. Start it with `ollama serve` "
                        f"and run `ollama pull {LLM_MODEL}` and `ollama pull {EMBED_MODEL}`.\n\nDetails: {e}"
                    )
                    st.stop()

                sources = []
                for d in docs:
                    if d.metadata.get("type") == "scheme":
                        line = f"[{display_title(d.metadata['title'])}]({d.metadata['source']})"
                    else:
                        line = f"{Path(d.metadata.get('source', 'document')).name}, page {d.metadata.get('page', 0) + 1}"
                    if line not in sources:
                        sources.append(line)
                with st.expander("Schemes used"):
                    for src in sources:
                        st.markdown(f"- {src}")

            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    if st.session_state.messages and st.button("🧹 New chat", key="new_chat"):
        st.session_state.messages = []
        st.rerun()