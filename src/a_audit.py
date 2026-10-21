import pandas as pd

RAW = "data/raw/"

def load(name):
    # Only an EMPTY cell counts as missing. "N/A" and "error" stay as text so we can count them.
    return pd.read_csv(RAW + name, keep_default_na=False, na_values=[""])

customers, products, orders = load("customers.csv"), load("products.csv"), load("orders.csv")

# --- general profile of each table ---
for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n===== {name} =====")
    print("shape:", df.shape)
    print(df.dtypes)
    miss = df.isna().sum()
    print(pd.DataFrame({"missing": miss, "pct": (miss / len(df) * 100).round(2)}))

# --- the specific audit figures ---
amount_num = pd.to_numeric(orders.order_amount, errors="coerce")
audit = {
    "raw order rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders.status == "pending").sum()),
    "distinct order_id": orders.order_id.nunique(),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((orders.delivery_minutes < 0).sum()),
    "blank delivery_minutes": int(orders.delivery_minutes.isna().sum()),
    "blank age": int(customers.age.isna().sum()),
    "age outside 10-100": int(((customers.age < 10) | (customers.age > 100)).sum()),
    "distinct raw city spellings": customers.city.nunique(),
    "distinct non-blank gender spellings": customers.gender.nunique(),
    "blank gender": int(customers.gender.isna().sum()),
}
print("\n===== AUDIT FIGURES =====")
for k, v in audit.items():
    print(f"{k:40s} {v}")