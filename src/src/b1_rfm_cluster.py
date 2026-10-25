import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------- B1: RFM ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = orders[orders.status == "completed"]
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = (comp.groupby("customer_id")
           .agg(last_ts=("order_ts", "max"),
                frequency=("order_id", "nunique"),
                monetary=("order_amount", "sum"))
           .reset_index())
rfm["recency"] = (snapshot - rfm.last_ts).dt.days

print("customers:", len(rfm))
print("recency  median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max())
print("monetary  mean/max:", round(rfm.monetary.mean(), 2), round(rfm.monetary.max(), 2))

# ---------- B2: K-means ----------
FEATURES = ["recency", "frequency", "monetary"]
X_scaled = StandardScaler().fit_transform(rfm[FEATURES])

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X_scaled, km.labels_), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# ---------- B3: profile, name, PCA ----------
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X_scaled)
rfm["segment_id"] = km.labels_

prof = rfm.groupby("segment_id")[FEATURES].mean()
prof["size"] = rfm.groupby("segment_id").size()
print(prof.round(2))

# Name segments by their behaviour, not by their cluster number
names, left = {}, set(prof.index)
premium = prof.loc[list(left), "monetary"].idxmax()
names[premium] = "Premium big spenders"; left.discard(premium)
loyal = prof.loc[list(left), "frequency"].idxmax()
names[loyal] = "Loyal frequent buyers"; left.discard(loyal)
at_risk = prof.loc[list(left), "recency"].idxmax()
names[at_risk] = "At-risk fading buyers"; left.discard(at_risk)
for s in left:
    names[s] = "Regular occasional buyers"

rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium
print(rfm.groupby("segment_name").size())

pca = PCA(n_components=2).fit(X_scaled)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3),
      "total:", round(pca.explained_variance_ratio_.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id",
     "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv")

