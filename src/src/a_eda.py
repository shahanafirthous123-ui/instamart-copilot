import sqlite3
import numpy as np
import pandas as pd

con = sqlite3.connect("data/instamart.db")
df = pd.read_sql_query("""
    SELECT o.*, c.city, p.category FROM orders o
    JOIN customers c USING(customer_id) JOIN products p USING(product_id)
    WHERE o.status = 'completed'""", con)

# (a) pivot vs SQL Q2
pv = pd.pivot_table(df, index="city", columns="category", values="order_amount",
                    aggfunc="sum", margins=True)
print(pv.round(2))
print("grand total:", round(pv.loc["All", "All"], 2))               # must equal the Q2 total

city = df.groupby("city").agg(total=("order_amount", "sum"),
                              mean=("order_amount", "mean"),
                              n=("order_amount", "count")).round(2)
print(city)          # Bengaluru 523643.40 / 276.47 / 1894 ; Mumbai 479058.50 / 297.00 / 1613

# (b) distribution of order_amount
x = df.order_amount
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("mean", round(x.mean(), 2), "median", x.median(), "skew", round(x.skew(), 2))   # 269.91, 135.00, ~4.61
print("Q1", q1, "Q3", q3, "fences", lo, hi)                                           # 64.80, 297.00, upper 645.30
print("outliers above:", int((x > hi).sum()), "below:", int((x < lo).sum()))          # 897, 0

cm = df[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(cm.round(3))
pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack()
print("largest |corr|:", pairs.abs().idxmax(), round(pairs[pairs.abs().idxmax()], 3))  # amount vs discount ~ -0.095