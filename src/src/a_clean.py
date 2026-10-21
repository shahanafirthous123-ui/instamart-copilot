import sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw/", "data/clean/"

def load(name):
    return pd.read_csv(RAW + name, keep_default_na=False, na_values=[""])

c, p, o = load("customers.csv"), load("products.csv"), load("orders.csv")

# ---------- CUSTOMERS ----------
c["city"] = c.city.str.strip().str.title()          # (a)

def map_gender(x):                                   # (b)
    if pd.isna(x):
        return "unknown"
    x = x.strip().lower()
    return "male" if x in ("m", "male") else "female" if x in ("f", "female") else "unknown"
c["gender"] = c.gender.map(map_gender)

bad_age = (c.age < 10) | (c.age > 100)               # (c)
print("ages set to missing (outside 10-100):", int(bad_age.sum()))   # 6
c.loc[bad_age, "age"] = np.nan
print("ages to fill with city median:", int(c.age.isna().sum()))     # 67 = 61 + 6
c["age"] = c.age.fillna(c.groupby("city")["age"].transform("median")).round().astype(int)

c["signup_date"] = pd.to_datetime(c.signup_date)     # (d)

# ---------- ORDERS ----------
o["order_ts"] = pd.to_datetime(o.order_ts)                                    # rule 1

n = len(o); o = o.drop_duplicates()                                           # rule 2
print("exact duplicates removed:", n - len(o))                                # 119

n = len(o)
o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last")        # rule 3
print("superseded pending records removed:", n - len(o))                      # 656
o = o.sort_values("order_id").reset_index(drop=True)

o["order_amount"] = pd.to_numeric(o.order_amount, errors="coerce")            # rule 4
price = o.product_id.map(p.set_index("product_id").price_inr)
recomputed = (price * o.quantity * (1 - o.discount_pct / 100)).round(2)
print("amounts filled from product price:", int(o.order_amount.isna().sum()))
o["order_amount"] = o.order_amount.fillna(recomputed)

neg = o.delivery_minutes < 0                                                  # rule 5
print("negative delivery_minutes set to missing:", int(neg.sum()))
o.loc[neg, "delivery_minutes"] = np.nan

category = o.product_id.map(p.set_index("product_id").category)
completed = o.status == "completed"
print("completed orders needing a delivery time:", int((completed & o.delivery_minutes.isna()).sum()))
med = o.loc[completed, "delivery_minutes"].groupby(category[completed]).transform("median")
o["delivery_minutes"] = o.delivery_minutes.fillna(med)    # med only has completed rows,
                                                          # so cancelled rows stay missing
# ---------- STORE ----------
c_out = c.copy(); c_out["signup_date"] = c_out.signup_date.dt.strftime("%Y-%m-%d")
o_out = o.copy(); o_out["order_ts"] = o_out.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")

con = sqlite3.connect("data/instamart.db")
for name, df in [("customers", c_out), ("products", p), ("orders", o_out)]:
    df.to_csv(f"{CLEAN}{name}.csv", index=False)
    df.to_sql(name, con, if_exists="replace", index=False)
con.close()
print("clean tables written")