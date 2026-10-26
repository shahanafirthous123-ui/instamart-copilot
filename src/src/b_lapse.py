import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- load ----------
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id", how="left")

CUT = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

# ---------- B4: features from BEFORE the cutoff only ----------
pre = orders[orders.order_ts < CUT]                    # the only orders features may use
pre_c = pre[pre.status == "completed"]

g = pre_c.groupby("customer_id")
feat = g.agg(last_ts=("order_ts", "max"),
             frequency=("order_id", "nunique"),
             monetary=("order_amount", "sum"),
             avg_discount=("discount_pct", "mean"),
             avg_delivery=("delivery_minutes", "mean"))
feat["recency"] = (CUT - feat.last_ts).dt.days
feat["fav_category"] = g.category.agg(lambda s: s.mode().iloc[0])
feat["cancel_rate"] = pre.groupby("customer_id").status.apply(lambda s: (s == "cancelled").mean())
feat = feat.reset_index().merge(cust[["customer_id", "city", "age", "gender", "signup_date"]],
                                on="customer_id", how="left")
feat["tenure_days"] = (CUT - feat.signup_date).dt.days
feat = feat.set_index("customer_id")

# ---------- label: from September only ----------
sept = orders[(orders.status == "completed") & (orders.order_ts >= CUT) & (orders.order_ts < END)]
feat["lapsed"] = (~feat.index.isin(sept.customer_id)).astype(int)

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]

print("cohort:", len(X), "| lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%)")   # 1156 | 443 (38.32%)
assert X.isna().sum().sum() == 0, "feature matrix has missing values"
print("no missing values in any feature column")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(X_train), len(X_test))                                  # 924 / 232

# baseline: always predict "not lapsed" (0)
base = np.zeros(len(y_test), dtype=int)
print("baseline accuracy:", round(accuracy_score(y_test, base), 3),
      "F1:", f1_score(y_test, base, zero_division=0))   

      # ---------- B5: preprocessing (fitted on training rows only) ----------
def make_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

probe = make_pre().fit(X_train)                       # fitted on TRAIN rows only
learned = probe.named_transformers_["num"].mean_[NUM.index("frequency")]
print("scaler mean of frequency (train only):", round(learned, 3))
print("full-cohort mean of frequency        :", round(X.frequency.mean(), 3))

# ---------- B6: tune and evaluate ----------
lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
gs_lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
gs_lr.fit(X_train, y_train)

rf_pipe = Pipeline([("pre", make_pre()),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
gs_rf = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8],
                               "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1")
gs_rf.fit(X_train, y_train)

print("LR best:", gs_lr.best_params_, "CV F1:", round(gs_lr.best_score_, 3))
print("RF best:", gs_rf.best_params_, "CV F1:", round(gs_rf.best_score_, 3))

def report(name, model, Xt, yt):
    pred = model.predict(Xt)
    print(f"\n{name}\nconfusion matrix [[TN FP],[FN TP]]:\n{confusion_matrix(yt, pred)}")
    print("precision:", round(precision_score(yt, pred, zero_division=0), 3),
          "recall:", round(recall_score(yt, pred, zero_division=0), 3),
          "F1:", round(f1_score(yt, pred, zero_division=0), 3))

report("Logistic regression (test)", gs_lr.best_estimator_, X_test, y_test)
report("Random forest (test)", gs_rf.best_estimator_, X_test, y_test)

# ---------- B7: over-sample the TRAINING set only ----------
train = X_train.assign(lapsed=y_train)
n_extra = (y_train == 0).sum() - (y_train == 1).sum()
extra = train[train.lapsed == 1].sample(n=n_extra, replace=True, random_state=42)
bal = pd.concat([train, extra])
print("balanced class counts:", bal.lapsed.value_counts().to_dict())       # 570 / 570

lr_bal = clone(gs_lr.best_estimator_).fit(bal[NUM + CAT], bal.lapsed)
report("LR on over-sampled training data (test)", lr_bal, X_test, y_test)
report("LR original, un-balanced (test)", gs_lr.best_estimator_, X_test, y_test)

# ---------- B8a: interpretation ----------
rf_best = gs_rf.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names)
print("\nTop 5 random-forest features:\n", imp.sort_values(ascending=False).head(5).round(3))

lr_best = gs_lr.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0],
                 index=lr_best.named_steps["pre"].get_feature_names_out())
print("\nLR coefficients (recency, frequency, cancel_rate):\n",
      coef[["num__recency", "num__frequency", "num__cancel_rate"]].round(3))

      # ---------- B8b: persist and reload ----------
best = gs_rf.best_estimator_          # change to gs_lr.best_estimator_ if you selected logistic regression
joblib.dump(best, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
assert (loaded.predict(X_test) == best.predict(X_test)).all()
print("reloaded model reproduces test predictions exactly")

# ---------- B8c: retention list (Premium customers in the test set) ----------
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg.is_premium, "customer_id"])

res = pd.DataFrame({"customer_id": X_test.index,
                    "lapse_probability": best.predict_proba(X_test)[:, 1].round(4),
                    "actually_lapsed": y_test.values})
res = res[res.customer_id.isin(premium_ids)].sort_values("lapse_probability", ascending=False)
top = res.head(10)
top.to_csv("outputs/retention_list.csv", index=False)
print("premium customers in test set:", len(res))
print("in the list:", len(top), "| truly lapsed:", int(top.actually_lapsed.sum()))