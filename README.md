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
## Part C
## Part D
## Limitations and Next Steps