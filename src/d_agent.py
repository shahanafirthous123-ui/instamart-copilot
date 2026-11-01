import json
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mock_llm
from d_rag import retrieve          # reuse the retrieval from D2

DB = "data/instamart.db"
LEDGER = Path("outputs/refund_ledger.jsonl")
RUNS = Path("outputs/agent_runs.jsonl")
STATE = Path("state")

# ---------------- the five tools ----------------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes "
                      "FROM orders WHERE order_id = ?", (order_id,)).fetchone()   # parameterised query
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2],
            "delivery_minutes": row[3]}          # NULL in SQL becomes None in Python

def search_policy(query):
    return {"results": [{"source": r["source"], "score": round(r["score"], 4), "text": r["text"]}
                        for r in retrieve(query, k=3)]}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}

def _ledger(tool, order_id, refund_inr):
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool, "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy,
            "calculate_refund": calculate_refund, "issue_refund": issue_refund,
            "escalate_to_human": escalate_to_human}

def _spec(name, desc, props):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id in the database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents for relevant passages.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from order amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000).",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]

class SimulatedCrash(Exception):
    pass

SYSTEM_AGENT = ("You are a refund AGENT. Use the available tools to decide refund requests "
                "and finish with a one-line answer of the form STATUS: explanation.")

def save_checkpoint(st):
    STATE.mkdir(exist_ok=True)
    out = dict(st, step=len(st["steps"]))
    (STATE / f"{st['run_id']}.json").write_text(json.dumps(out, indent=1))

def finish(st, status, final):
    refund = 0.0
    for s in st["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            refund = s["observation"]["refund_inr"]
    return {"run_id": st["run_id"], "status": status, "final_answer": final,
            "steps": st["steps"], "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "refund_inr": refund}

def fallback(st, status, reason):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    final = (f"{reason}. Handing over to a human agent. Steps completed so far: {done}. "
             f"Request: {st['user_message']}")
    return finish(st, status, final)

def _loop(st, crash_after_tools=None):
    lim, start = st["limits"], time.monotonic()
    while st["llm_calls"] < lim["max_iterations"]:
        if time.monotonic() - start >= lim["time_limit_seconds"]:          # (c) timeout, top of iteration
            return fallback(st, "FALLBACK_TIMEOUT", "Time limit reached")
        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > lim["token_limit"]:                         # (b) before running the tool
            return fallback(st, "FALLBACK_TOKEN_LIMIT", "Token limit exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                             # final answer
            final = msg.content
            return finish(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        try:
            obs = TOOL_MAP[tc.name](**tc.arguments)                        # look up the name, call it
        except Exception as e:
            obs = {"error": str(e)}
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(obs)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["tools_executed"].append(tc.name)
        save_checkpoint(st)                                                # D7: after every observation
        if crash_after_tools and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(st["run_id"])
    return fallback(st, "FALLBACK_MAX_ITERATIONS", "Maximum iterations reached without a final answer")  # (a)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, crash_after_tools=None, run_id=None):
    st = {"run_id": run_id or f"{scenario_id}-{uuid.uuid4().hex[:6]}",
          "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": SYSTEM_AGENT},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "steps": [], "tools_executed": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                     "time_limit_seconds": time_limit_seconds}}
    return _loop(st, crash_after_tools)

def resume_agent(run_id):
    st = json.loads((STATE / f"{run_id}.json").read_text())
    st.pop("step", None)
    return _loop(st)          # continues with restored messages; old tools are NOT re-run

if __name__ == "__main__":
    LEDGER.parent.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
    LEDGER.write_text(""); RUNS.write_text("")                       # fresh outputs for a clean rerun
    for p in STATE.glob("*.json"):
        p.unlink()

    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("scenario | status | refund | calls | tokens | expected | ok")
    with RUNS.open("a") as f:
        for r in sc.itertuples():
            res = run_agent(r.message, r.scenario_id)
            f.write(json.dumps({"scenario_id": r.scenario_id, "status": res["status"],
                                "final_answer": res["final_answer"], "llm_calls": res["llm_calls"],
                                "tokens_used": res["tokens_used"], "steps": res["steps"]}) + "\n")
            exp = r.expected_outcome
            ok = (res["status"].startswith("FALLBACK") if exp == "FALLBACK" else res["status"] == exp) \
                 and abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
            print(r.scenario_id, res["status"], res["refund_inr"], res["llm_calls"],
                  res["tokens_used"], exp, "OK" if ok else "MISMATCH")

    s1 = sc[sc.scenario_id == "S1"].message.iloc[0]
    d1 = run_agent(s1, "S1", token_limit=200, run_id="S1-tokenlimit-demo")
    print("\nS1 token_limit=200       ->", d1["status"])
    d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    print("S1 time_limit_seconds=0  ->", d2["status"])

    # crash and resume
    lines_before = len(LEDGER.read_text().splitlines())
    try:
        run_agent(s1, "S1", run_id="S1-crash-demo", crash_after_tools=2)
    except SimulatedCrash as e:
        print("\nsimulated crash in run", e)
    res = resume_agent("S1-crash-demo")
    tools = json.loads((STATE / "S1-crash-demo.json").read_text())["tools_executed"]
    new_lines = len(LEDGER.read_text().splitlines()) - lines_before
    print("tools executed across crash + resume:", tools)
    print("final:", res["status"], res["refund_inr"], "| new ledger lines:", new_lines)
    assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) <= 0.01 and new_lines == 1
    print("crash-and-resume check passed")