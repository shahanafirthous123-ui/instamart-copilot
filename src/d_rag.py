import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import mock_llm" works
import mock_llm

POLICY_DIR = Path("data/policies")

def chunk_words(words, size=40, overlap=10):
    step, out, i = size - overlap, [], 0
    while True:
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words):          # this chunk reaches the last word -> stop
            break
        i += step
    return out

def make_chunks(size=40, overlap=10):
    chunks = []
    for f in sorted(POLICY_DIR.glob("*.txt")):
        for text in chunk_words(f.read_text(encoding="utf-8").split(), size, overlap):
            chunks.append({"source": f.name, "text": text})
    return chunks
def build_index(overlap=10):
    chunks = make_chunks(40, overlap)
    vec = TfidfVectorizer()
    mat = vec.fit_transform([c["text"] for c in chunks])      # fit on chunk texts only
    return {"chunks": chunks, "vec": vec, "mat": mat}

INDEX = build_index(overlap=10)

def retrieve(query, k=3, index=None):
    index = index or INDEX
    sims = cosine_similarity(index["vec"].transform([query]), index["mat"]).ravel()
    order = np.argsort(-sims, kind="stable")[:k]
    return [{"source": index["chunks"][i]["source"],
             "score": float(sims[i]),
             "text": index["chunks"][i]["text"]} for i in order]
SYSTEM_RAG = ("You are a POLICY ASSISTANT. Answer only from the supplied context. "
              "If the answer is not in the context, say you could not find the information.")

def answer(question, index=None):
    hits = retrieve(question, k=3, index=index)
    ctx = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_RAG},
                {"role": "user", "content": f"CONTEXT:\n{ctx}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(index, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    failures = []
    for r in ev.itertuples():
        ans, sources = answer(r.question, index)
        if r.expected_doc == "NONE":
            ok = ans.startswith("I could not find")
            fallback += ok
            if not ok:
                failures.append((r.question, "should have refused", ans))
        else:
            d, f = r.expected_doc in sources, r.expected_fact in ans
            doc_hit += d; fact_hit += f
            if not (d and f):
                failures.append((r.question, f"doc_ok={d} fact_ok={f}", ans))
    print(f"\n[{label}] expected doc in top-3: {doc_hit}/12 | fact in answer: {fact_hit}/12 "
          f"| fallback on unanswerable: {fallback}/2")
    for q, why, ans in failures:
        print("  FAIL:", q, "|", why, "|", ans)

if __name__ == "__main__":
    from collections import Counter
    print("chunks:", len(INDEX["chunks"]), dict(Counter(c["source"] for c in INDEX["chunks"])))
    print("top 3 sources for 'refund policy late delivery':",
          [h["source"] for h in retrieve("refund policy late delivery")])
    evaluate(INDEX, "40 words, overlap 10")
    evaluate(build_index(overlap=0), "40 words, overlap 0")