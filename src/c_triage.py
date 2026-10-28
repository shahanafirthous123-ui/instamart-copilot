
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# let "import mock_llm" find the file in the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATS = ["billing", "delivery", "technical", "fraud"]

# keep_default_na=False keeps empty ticket texts as "" instead of NaN
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# ---------------- C2: redaction (order matters: email, phone, order id) ----------------
PATTERNS = [("EMAIL",    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE",    re.compile(r"\b[6-9]\d{9}\b"),          "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"),               "[ORDER_ID]")]

def redact(text, counts=None):
    for name, pat, placeholder in PATTERNS:
        text, n = pat.subn(placeholder, text)
        if counts is not None:
            counts[name] += n
    return text

def redaction_report():
    counts = Counter()
    redacted = [redact(t, counts) for t in tickets.text]
    print("substitutions:", dict(counts), "total:", sum(counts.values()))   # 55 / 91 / 132 = 278
    left = sum(len(pat.findall(t)) for t in redacted for _, pat, _ in PATTERNS)
    print("PII left after re-applying the patterns:", left)                  # 0

# ---------------- C3: prompts ----------------
ZERO_SHOT = (
    "You are a support ticket router. CLASSIFY the customer ticket into exactly one category: "
    "billing, delivery, technical or fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: Someone hacked my account and used my wallet balance. -> {"category": "fraud", "reason": "unauthorized use of account"}
Ticket: The app crashes every time I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: The rider marked [ORDER_ID] as delivered but I never received it. -> {"category": "delivery", "reason": "not received"}"""

assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT

# ---------------- C4: retries with exponential backoff and jitter ----------------
STATS = Counter()

def call_with_retry(messages, ticket_id):
    for attempt in range(MAX_RETRIES + 1):            # first try + up to 3 retries
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            STATS["transient_errors"] += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            print(f"retry ticket={ticket_id} retry_no={attempt} wait={wait:.4f}s")
            time.sleep(wait)

# ---------------- C5: validate every answer ----------------
def validate(reply):
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):          # not JSON -> keyword route
        found = set(re.findall(r"[a-z]+", (reply or "").lower())) & set(CATS)
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATS:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""                # valid JSON but unusable (e.g. "shipping")

# ---------------- C6: pipeline ----------------
def run_pipeline(system_prompt, token_budget=None):
    mock_llm.reset()                                   # fresh counters for every full run
    STATS.clear()
    total = 0
    rows = []
    for t in tickets.itertuples(index=False):          # file order
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id,
               "true_category": t.true_category, "status": "", "category": "", "tokens_used": 0}
        if not t.text.strip():                                            # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:                       # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and total >= token_budget:          # budget check
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(t.text)}]      # redacted text and nothing else
            for m in messages:                                            # proof no PII reaches the model
                for _, pat, _ in PATTERNS:
                    assert not pat.search(m["content"]), "PII reached the prompt"
            resp = call_with_retry(messages, t.ticket_id)
            row["tokens_used"] = resp.usage.total_tokens
            total += row["tokens_used"]
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), total

def summarise(df, total, name):
    ok = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = int((ok.category == ok.true_category).sum())
    print(f"\n[{name}] statuses: {df.status.value_counts().to_dict()}")
    print(f"classified={len(ok)}  accuracy={correct / len(ok):.1%} ({correct}/{len(ok)})  "
          f"total_tokens={total}  tokens_per_classified={total / len(ok):.1f}  "
          f"transient_errors={STATS['transient_errors']}")
    return ok

# ---------------- main ----------------
if __name__ == "__main__":
    print("tickets:", len(tickets))                                                      # 164
    print("empty:", int((tickets.text.str.strip() == "").sum()))                         # 4
    print("long :", int(tickets.text.map(est_tokens).gt(MAX_INPUT_TOKENS).sum()),
          "| long ticket estimate:", tickets.text.map(est_tokens).max())                 # 4 | 734

    redaction_report()

    # C7: zero-shot vs few-shot
    zs, zs_total = run_pipeline(ZERO_SHOT)
    ok_zs = summarise(zs, zs_total, "zero-shot")
    print(pd.crosstab(ok_zs.true_category, ok_zs.category, rownames=["true"], colnames=["predicted"]))

    fs, fs_total = run_pipeline(FEW_SHOT)
    summarise(fs, fs_total, "few-shot")

    # C8a: budget demo (lower to 5000 if nothing is exhausted, and report the number you used)
    BUDGET_DEMO = 6000
    demo, demo_total = run_pipeline(ZERO_SHOT, token_budget=BUDGET_DEMO)
    demo[["ticket_id", "customer_id", "status", "category", "tokens_used"]].to_csv(
        "outputs/triage_budget_demo.csv", index=False)
    print(f"\nbudget demo ({BUDGET_DEMO}): budget_exhausted =",
          int((demo.status == "budget_exhausted").sum()), "| tokens spent:", demo_total)

    # C8b: final run with routing
    seg = pd.read_csv("outputs/customer_segments.csv")
    premium = dict(zip(seg.customer_id, seg.is_premium))    # no row -> treated as not premium
    seg_name = dict(zip(seg.customer_id, seg.segment_name))

    HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

    def route(r):
        if r.status in HUMAN:
            return "human_review", "NORMAL"
        if r.category == "fraud":
            return "security_team", "HIGH"
        if premium.get(r.customer_id, False):
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    final, final_total = run_pipeline(FEW_SHOT, token_budget=100000)
    final[["route", "priority"]] = [route(r) for r in final.itertuples()]
    final["segment_name"] = final.customer_id.map(seg_name).fillna("")
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name",
            "priority", "route", "tokens_used"]
    final[cols].to_csv("outputs/triage_results.csv", index=False)
    print("\nfinal rows:", len(final))                           # 164
    print(final.route.value_counts().to_dict())