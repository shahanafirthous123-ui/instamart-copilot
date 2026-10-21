import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)

comp = o[o.status == "completed"]
assert len(o) == 12309, "order rows"
assert o.order_id.is_unique, "one row per order_id"
assert (o.status == "completed").sum() == 11110, "completed count"
assert (o.status == "cancelled").sum() == 1199, "cancelled count"
assert not (o.status == "pending").any(), "no pending rows"
assert o.order_amount.isna().sum() == 0, "missing amounts"
assert comp.delivery_minutes.isna().sum() == 0, "missing delivery (completed)"
assert o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199, "cancelled delivery stays missing"
assert round(comp.order_amount.sum(), 2) == 2998668.80, "completed revenue"
assert c.city.nunique() == 6, "six cities"
assert c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                           "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
assert c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
assert c.age.isna().sum() == 0 and c.age.between(18, 60).all(), "ages"
print("ALL CHECKPOINTS PASSED")