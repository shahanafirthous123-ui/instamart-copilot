import re
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql").read()
print("file length:", len(text))

# Split at every comment line that contains Q1..Q7, even if indented or written as "--Q1:" etc.
parts = re.split(r"^[ \t]*--[ \t]*Q(\d)\b[^\n]*$", text, flags=re.M | re.I)
q = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
print("queries found:", sorted(q))
if not q:
    raise SystemExit("No '-- Q1' style markers found in sql/queries.sql. Check the file contents.")

r = {n: pd.read_sql_query(sql, con) for n, sql in q.items()}

print("Q1 customers:", len(r[1]))                                          # 30
print("Q2 rows:", len(r[2]), "sum:", round(r[2].revenue.sum(), 2))         # 30, 2998668.8
print(r[2].sort_values("revenue", ascending=False).head(1))                # Bengaluru / Electronics 233909.45
print("Q3 products:", len(r[3]))                                           # 11
print("Q4 customers:", len(r[4]))                                          # 25
print("Q5 customers:", len(r[5]), "avg:", round(r[5].avg_total.iloc[0], 2))  # 550, 2552.06
print(r[6][r[6].rn == 1].sort_values("city"))                              # top customer per city
print("Q6 rows where RANK != ROW_NUMBER:", int((r[6].rnk != r[6].rn).sum()))  # 8
print("Q7 rows:", len(r[7]), "last cumulative:", round(r[7].cumulative_revenue.iloc[-1], 2))  # 183, 2998668.8