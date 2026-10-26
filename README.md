# Instamart Customer Intelligence & Support Copilot

## Overview

## Reproduce
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python generate_data.py`

## Part A

ages set to missing (outside 10-100): 6
ages to fill with city median: 67
exact duplicates removed: 119
superseded pending records removed: 656
amounts filled from product price: 112
negative delivery_minutes set to missing: 105
completed orders needing a delivery time: 774

audit Data: 
===== customers =====
shape: (1200, 5)
customer_id        str
city               str
signup_date        str
age            float64
gender             str
dtype: object
             missing   pct
customer_id        0  0.00
city               0  0.00
signup_date        0  0.00
age               61  5.08
gender            30  2.50

===== products =====
shape: (40, 4)
product_id        str
product_name      str
category          str
price_inr       int64
dtype: object
              missing  pct
product_id          0  0.0
product_name        0  0.0
category            0  0.0
price_inr           0  0.0

===== orders =====
shape: (13084, 9)
order_id                str
customer_id             str
product_id              str
quantity              int64
order_ts                str
status                  str
discount_pct          int64
order_amount            str
delivery_minutes    float64
dtype: object
                  missing    pct
order_id                0   0.00
customer_id             0   0.00
product_id              0   0.00
quantity                0   0.00
order_ts                0   0.00
status                  0   0.00
discount_pct            0   0.00
order_amount            0   0.00
delivery_minutes     2541  19.42

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
blank gender       

cleaned data:

Q1 customers: 30
Q2 rows: 30 sum: 2998668.8
        city     category    revenue
1  Bengaluru  Electronics  233909.45
Q3 products: 11
Q4 customers: 25
Q5 customers: 550 avg: 2552.06
    customer_id       city  total_spend  rnk  drnk  rn
0         C0082  Bengaluru     12856.20    1     1   1
192       C0930    Chennai     10151.40    1     1   1
379       C0291      Delhi     13742.60    1     1   1
584       C0743  Hyderabad     12730.05    1     1   1
771       C0693     Mumbai     14766.55    1     1   1
954       C0426       Pune     10377.10    1     1   1
Q6 rows where RANK != ROW_NUMBER: 8
Q7 rows: 183 last cumulative: 2998668.8

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

## Part C
## Part D
## Limitations and Next Steps