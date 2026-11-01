# Instamart Customer Intelligence & Support Copilot

## Overview
This repository contains a Customer Intelligence & Support Copilot built on synthetic, fictional
Instamart data (all money in INR). It has four connected parts:
- **Part A** cleans the raw exports and stores them in a SQLite database.
- **Part B** segments customers (RFM + K-means) and predicts who will lapse.
- **Part C** triages support tickets with privacy, retry, validation and budget guardrails, and routes
  them using the Part B segments.
- **Part D** is a policy RAG assistant and a refund agent that queries the Part A database.

All language-model calls use the supplied offline `mock_llm.py`. No API key, network or paid service is used.
`generate_data.py` and `mock_llm.py` are unmodified.

## Reproduce
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python generate_data.py`
4. `python src/a_audit.py`
5. `python src/a_clean.py`
6. `python src/a_validate.py`
7. `python src/a_sql.py`
8. `python src/a_eda.py`
9. `python src/b_rfm_cluster.py`
10. `python src/b_lapse.py`
11. `python src/c_triage.py`
12. `python src/d_rag.py`
13. `python src/d_agent.py`

Run every command from the repository root. Python 3.11+.

## Part A

### Cleaning rules (with rows or values changed)
**Customers**

ages set to missing (outside 10-100): 6
ages to fill with city median: 67
exact duplicates removed: 119
superseded pending records removed: 656
amounts filled from product price: 112
negative delivery_minutes set to missing: 105
completed orders needing a delivery time: 774

After cleaning: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184;
female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.


             missing   pct

age               61  5.08
gender            30  2.50
delivery_minutes  2541 19.42

## Orders 

| 1 | order_ts converted to datetime | all rows |
| 2 | Exact duplicate rows removed | ‹119› |
| 3 | Per order_id keep latest order_ts (drops superseded pending records) | ‹656› |
| 4 | Unparseable order_amount made missing, then filled with round(price x quantity x (1 - discount/100), 2) | 112 |
| 5a | Negative delivery_minutes set to missing | 105 |
| 5b | Missing delivery_minutes of completed orders filled with category median (completed only) | 774 |
| 5c | Cancelled orders left missing, because a cancelled order was never delivered | 1,199 |

After cleaning (reference): 12,309 rows; 11,110 completed; 1,199 cancelled; no pending;
0 missing order_amount; completed revenue INR 2,998,668.80.

===== AUDIT FIGURES =====
raw order rows                           13084
exact duplicate rows                     119
pending rows                             656
distinct order_id                        12309
non-numeric order_amount                 120
negative delivery_minutes                106
blank delivery_minutes                   2541
blank age                                61
age outside 10-100                       6
distinct raw city spellings              24
distinct non-blank gender spellings      10
blank gender                             30

### Storage and validation
Clean tables are in `data/clean/*.csv` and `data/instamart.db` (tables customers, products, orders;
order_ts stored as text `YYYY-MM-DD HH:MM:SS`). `src/a_validate.py` asserts 13 checkpoints and prints
`ALL CHECKPOINTS PASSED`.

### SQL queries (`sql/queries.sql`)

Q1 customers: 30
Q2 rows: 30 sum: 2998668.8
        city     category    revenue
1  Bengaluru  Electronics  INR 233909.45  total INR 2,998,668.80, largest cell Bengaluru / Electronics INR 233,909.45 
Q3 products: 11
Q4 customers: 25 with no completed order
Q5 customers: 550 avg: INR 2552.06
    customer_id       city  total_spend  rnk  drnk  rn
0         C0082  Bengaluru     12856.20    1     1   1
192       C0930    Chennai     10151.40    1     1   1
379       C0291      Delhi     13742.60    1     1   1
584       C0743  Hyderabad     12730.05    1     1   1
771       C0693     Mumbai     14766.55    1     1   1
954       C0426       Pune     10377.10    1     1   1
Q6 rows where RANK != ROW_NUMBER: 8
Q7 rows: 183 last cumulative: 2998668.8

Top customer per city (Q6): Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60),
Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10).

RANK differs from ROW_NUMBER only on ties: RANK gives tied customers the same rank, while ROW_NUMBER
gives every row a unique number, so tied rows get different ROW_NUMBERs.

### pandas cross-check and exploratory analysis
- Pivot table (margins=True) grand total = ‹2,998,668.80›, equal to the SQL Q2 total.
- Per city (completed order_amount): ‹table of sum / mean / count›
  (reference: Bengaluru 523,643.40 / 276.47 / 1,894; Mumbai 479,058.50 / 297.00 / 1,613).
- Distribution of completed order_amount: mean 269.91, median 135.00, skewness ≈ 4.61, Q1 64.80,
  Q3 297.00, upper fence 645.30 with 897 outliers above and 0 below the lower fence.
- Correlation matrix: ‹4x4 table›; largest absolute value is order_amount vs discount_pct ≈ -0.095.


Since Given data set is raw. The audit gives the “before” picture. Your cleaning must change those numbers in a documented way, and the grader compares your figures against the references. Next is to clean the data of customer and order . and store the cleaned data.

## Part B

customers: 1175
recency  median/min/max: 19.0 0 180
frequency mean/max: 9.46 33
monetary  mean/max: 2552.06 14766.55

 K  inertia  silhouette
 2   1820.8       0.465
 3   1268.2       0.445
 4    743.1       0.526
 5    617.4       0.480
 6    503.2       0.482
 7    408.5       0.416
 8    366.5       0.412

K4 is best for campeign . since the inertia is the total squared distance from each point to its cluster centre.Adding clusters always puts centres closer to points, so K=8 is always lower than K=2, and K equal to the number of customers would give 0. Because inertia can only fall as K grows, the number alone can’t pick K, so you look for where the improvement flattens (the elbow) and combine it with silhouette and business sense.

             recency  frequency  monetary  size
segment_id                                    
0             28.86       3.77   1057.62   436
1             24.12       6.77   7129.70   138
2              8.77      19.40   3410.02   419
3            109.04       2.20    685.99   182

segment_name
At-risk fading buyers        182
Loyal frequent buyers        419
Premium big spenders         138
Regular occasional buyers    436
dtype: int64
PCA explained variance: [0.634 0.218] total: 0.852

cluster numbers are only labels with no ordering

cohort: 1156 | lapsed: 443 (38.32%)
no missing values in any feature column
train/test: 924 232
baseline accuracy: 0.616 F1: 0.0


Logistic regression (test)
confusion matrix [[TN FP],[FN TP]]:
[[111  32]
 [ 31  58]]
precision: 0.644 recall: 0.652 F1: 0.648

Random forest (test)
confusion matrix [[TN FP],[FN TP]]:
[[113  30]
 [ 31  58]]
precision: 0.659 recall: 0.652 F1: 0.655
balanced class counts: {1: 570, 0: 570}

LR on over-sampled training data (test)
confusion matrix [[TN FP],[FN TP]]:
[[90 53]
 [13 76]]
precision: 0.589 recall: 0.854 F1: 0.697

LR original, un-balanced (test)
confusion matrix [[TN FP],[FN TP]]:
[[111  32]
 [ 31  58]]
precision: 0.644 recall: 0.652 F1: 0.648

recall moved the most. The model now catches far more of the customers who really will lapse, but it also flags more loyal customers by mistake, so precision doesn’t improve. For the retention team this means a bigger contact list with more wasted offers. That is acceptable when offers are cheap and losing a customer is expensive.


cohort: 1156 | lapsed: 443 (38.32%)
no missing values in any feature column
train/test: 924 232
baseline accuracy: 0.616 F1: 0.0
scaler mean of frequency (train only): 8.115
full-cohort mean of frequency        : 8.029
LR best: {'clf__C': 100} CV F1: 0.651
RF best: {'clf__max_depth': 8, 'clf__min_samples_leaf': 10} CV F1: 0.681

Logistic regression (test)
confusion matrix [[TN FP],[FN TP]]:
[[111  32]
 [ 31  58]]
precision: 0.644 recall: 0.652 F1: 0.648

Random forest (test)
confusion matrix [[TN FP],[FN TP]]:
[[113  30]
 [ 31  58]]
precision: 0.659 recall: 0.652 F1: 0.655
balanced class counts: {1: 570, 0: 570}

LR on over-sampled training data (test)
confusion matrix [[TN FP],[FN TP]]:
[[90 53]
 [13 76]]
precision: 0.589 recall: 0.854 F1: 0.697

LR original, un-balanced (test)
confusion matrix [[TN FP],[FN TP]]:
[[111  32]
 [ 31  58]]
precision: 0.644 recall: 0.652 F1: 0.648

Top 5 random-forest features:
 num__frequency       0.280
num__monetary        0.271
num__recency         0.115
num__avg_delivery    0.061
num__cancel_rate     0.052
dtype: float64

LR coefficients (recency, frequency, cancel_rate):
 num__recency        0.039
num__frequency     -1.697
num__cancel_rate   -0.124
dtype: float64
reloaded model reproduces test predictions exactly
premium customers in test set: 40
in the list: 10 | truly lapsed: 2

the Premium flag comes from whole-window RFM that includes September, the same window that defines the lapse label, so it is a post-hoc label and this retention list is illustrative, not a leakage-free evaluation.

## Part C
### Input guardrails and redaction
| Item | Value (reference) |
|---|---|
| tickets / empty / long | 164 / 4 / 4 (long tickets estimate 734 tokens, limit 400) |
| emails / phones / order ids redacted | 55 / 91 / 132 |
| total substitutions | 278 |
| PII left after re-applying the patterns | 0 |

Patterns applied in order: email, phone, order id. An assertion checks every prompt before each model
call. The user message is exactly the redacted ticket text.

### Prompts
Zero-shot: contains CLASSIFY, the four categories, JSON-only format, and no `EXAMPLES:`.
Few-shot: same plus an `EXAMPLES:` line with four labelled examples, including fraud.

### Retries
Maximum 3 retries per ticket; wait = 0.05 x 2^n x uniform(0.9, 1.15) seconds. Transient errors per
full run: ‹19›, all recovered. Observed waits: ‹min–max, 0.045–0.0575›. `mock_llm.reset()` is called
before every full run.

### Validation
Valid JSON with an allowed category gives `classified_json`. Non-JSON replies use
`re.findall(r"[a-z]+", ...)` and are accepted only if exactly one category word is present
(`classified_keyword`). Anything else, including JSON with category outside the list, is
`flagged_invalid_output` and goes to human review.
Status counts over the 156 eligible tickets: 135 / 12 / 9.

[zero-shot statuses]: {'classified_json': 135, 'classified_keyword': 12, 'flagged_invalid_output': 9, 'skipped_empty': 4, 'flagged_long': 4}
classified=147  accuracy=85.7% (126/147)  total_tokens=10492  tokens_per_classified=71.4  transient_errors=19
predicted  billing  delivery  fraud  technical
true                                          
billing         38         0      0          0
delivery         0        36      0          0
fraud           11         0     23          0
technical       10         0      0         29


[few-shot] statuses: {'classified_json': 135, 'classified_keyword': 12, 'flagged_invalid_output': 9, 'skipped_empty': 4, 'flagged_long': 4}
classified=147  accuracy=100.0% (147/147)  total_tokens=25000  tokens_per_classified=170.1  transient_errors=19

budget demo (6000): budget_exhausted = 66 | tokens spent: 6052

Routing for `outputs/triage_results.csv` (few-shot, budget 100000):
| Condition (in order) | Route | Priority |
|---|---|---|
| skipped_empty, flagged_long, flagged_invalid_output, budget_exhausted | human_review | NORMAL |
| category = fraud | security_team | HIGH |
| customer is_premium | premium_desk | HIGH |
| otherwise | support_queue | NORMAL |

## Part D
### Chunking and retrieval
40-word chunks, overlap 10 (step 30): 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2).
For "refund policy late delivery" the top three chunks all come from refund_policy.txt.
TF measures how often a word appears in a chunk; IDF down-weights words that appear in many chunks.
Limitation: TF-IDF matches exact words only, so "money back" would not match "refund".

chunks: 15 {'delivery_policy.txt': 3, 'escalation_policy.txt': 3, 'payment_policy.txt': 3, 'privacy_policy.txt': 2, 'refund_policy.txt': 4}
top 3 sources for 'refund policy late delivery': ['refund_policy.txt', 'refund_policy.txt', 'refund_policy.txt']

[40 words, overlap 10] expected doc in top-3: 12/12 | fact in answer: 12/12 | fallback on unanswerable: 2/2

[40 words, overlap 0] expected doc in top-3: 12/12 | fact in answer: 10/12 | fallback on unanswerable: 2/2
  FAIL: What delivery time makes a completed order eligible for a full refund? | doc_ok=True fact_ok=False | receives a full refund of the order amount.
  FAIL: How soon is money returned when a payment fails but is deducted? | doc_ok=True fact_ok=False | If a payment fails but money is deducted, the amount is returned automatically
 
1000 must be approved by a human support lead before money is returned.

### Seven scenarios
scenario | status | refund | calls | tokens | expected | ok
S1 REFUND 128.25 5 992 REFUND OK
S2 REFUND 135.0 5 992 REFUND OK
S3 ESCALATE 1999.0 5 979 ESCALATE OK
S4 DENIED 0.0 4 681 DENIED OK
S5 DENIED 0.0 3 394 DENIED OK
S6 NOT_FOUND 0.0 2 110 NOT_FOUND OK
S7 FALLBACK_MAX_ITERATIONS 0.0 6 3114 FALLBACK OK


### Stop-safely demonstrations
| Demo | Status |
|---|---|
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT |
| S7 with default token_limit=6000 | FALLBACK_MAX_ITERATIONS (not stopped early) |

### Crash and resume (S1)
Crash raised after the second tool execution. Tools executed across both parts, in order:
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund

Four executions, none repeated; final status REFUND, INR 128.25. Checkpoints are in `state/`.
The ledger contains S1 twice only because the normal run and the crash demo both process S1; inside the
crash run exactly one new ledger line was written. On resume the clock restarts and the model-call
count continues from the checkpoint.

ReAct: the model decides one step at a time after seeing each result.
Plan-and-execute: the model writes the whole list of steps once, your code runs them, and the model is called again only if something fails.

## Limitations and Next Steps

- All data is synthetic; results do not describe real Instamart behaviour.
- The Premium flag is a post-hoc whole-window label that includes September, so the retention list is
  illustrative and not a leakage-free evaluation.
- Lapse model uses one cutoff and one split; no time-based validation or calibration.
- TF-IDF retrieval is word-based and misses paraphrases; the mock model is rule-based.
- Redaction uses fixed regular expressions and would miss names and addresses.
- Next steps: time-based validation, embedding retrieval, a real model behind the same guardrails,
  human approval workflow for escalations.