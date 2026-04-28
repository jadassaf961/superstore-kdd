"""
Machine Learning utilities for the ML Lab page.

Three blocks:
  1. Supervised REGRESSION — predict Total Profit per order
       Models: Linear, Random Forest, Gradient Boosting
       Eval: 5-fold CV (R², MAE, RMSE) + held-out test
  2. Supervised CLASSIFICATION — predict whether an order will be unprofitable
       Models: Logistic Regression, Random Forest, Gradient Boosting
       Eval: 5-fold CV (ROC-AUC, F1) + confusion matrix
  3. Unsupervised CLUSTERING — RFM customer segmentation via K-Means
       Eval: silhouette score, elbow method
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    silhouette_score,
)


# ══════════════════════════════════════════════════════════════════════════
# Common feature builder
# ══════════════════════════════════════════════════════════════════════════
ML_NUMERIC = ["Quantity", "Total Revenue"]
ML_CATEG   = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])


# ══════════════════════════════════════════════════════════════════════════
# 1. REGRESSION — predict Profit Margin per order
#
# Why margin instead of raw profit?
# Raw profit dollars vary by 4 orders of magnitude (a $20 paper order vs.
# a $20,000 copier order) and the dominant driver — discount % — is NOT
# in the dataset. So raw-profit R² is poor and not defensible.
#
# Margin (profit / revenue) is bounded, well-behaved, and answers the more
# useful business question: "What margin should we expect on this order?"
# ══════════════════════════════════════════════════════════════════════════
def regression_features(df: pd.DataFrame):
    """Predict Profit Margin (= Profit / Revenue) per order."""
    feats = df.copy()
    # Compute margin if not already present
    if "Profit Margin" not in feats.columns:
        feats["Profit Margin"] = np.where(
            feats["Total Revenue"].abs() > 1e-9,
            feats["Total Profit"] / feats["Total Revenue"],
            0.0,
        )
    # Drop rows with extreme/invalid margin so the target is sane
    feats = feats[feats["Profit Margin"].between(-2.0, 1.0)].reset_index(drop=True)

    num = ["Quantity", "Total Revenue"]
    if "Shipping Days" in feats.columns:
        num.append("Shipping Days")
    cat = [c for c in ML_CATEG if c in feats.columns]
    X = feats[num + cat].copy()
    y = feats["Profit Margin"]
    return X, y, num, cat


REGRESSORS = {
    "Linear Regression":   lambda **kw: LinearRegression(),
    "Random Forest":       lambda n_estimators=100, max_depth=None, **kw:
        RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                              random_state=42, n_jobs=-1),
    "Gradient Boosting":   lambda n_estimators=100, max_depth=3, learning_rate=0.1, **kw:
        GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  learning_rate=learning_rate, random_state=42),
}


def train_regressor(model_name: str, df: pd.DataFrame, hyperparams: dict = None,
                    cv_folds: int = 5, test_size: float = 0.2):
    hyperparams = hyperparams or {}
    X, y, num, cat = regression_features(df)
    pre = build_preprocessor(num, cat)
    estimator = REGRESSORS[model_name](**hyperparams)
    pipe = Pipeline([("pre", pre), ("est", estimator)])

    # Cross-validation on full data
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    cv_mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1))

    # Held-out test
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    metrics = {
        "cv_r2_mean":   float(cv_r2.mean()),
        "cv_r2_std":    float(cv_r2.std()),
        "cv_mae_mean":  float(cv_mae.mean()),
        "cv_rmse_mean": float(cv_rmse.mean()),
        "test_r2":      float(r2_score(yte, pred)),
        "test_mae":     float(mean_absolute_error(yte, pred)),
        "test_rmse":    float(np.sqrt(mean_squared_error(yte, pred))),
        "y_test":       yte.values,
        "y_pred":       pred,
        "feature_names": _get_feature_names(pipe, num, cat),
    }
    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════════
# 2. CLASSIFICATION — predict whether order is unprofitable
# ══════════════════════════════════════════════════════════════════════════
def classification_features(df: pd.DataFrame):
    feats = df.copy()
    feats["is_unprofitable"] = (feats["Total Profit"] < 0).astype(int)
    num = ["Quantity", "Total Revenue"]
    if "Shipping Days" in feats.columns:
        num.append("Shipping Days")
    cat = [c for c in ML_CATEG if c in feats.columns]
    X = feats[num + cat].copy()
    y = feats["is_unprofitable"]
    return X, y, num, cat


CLASSIFIERS = {
    "Logistic Regression": lambda **kw: LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":       lambda n_estimators=100, max_depth=None, **kw:
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda n_estimators=100, max_depth=3, learning_rate=0.1, **kw:
        GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   learning_rate=learning_rate, random_state=42),
}


def train_classifier(model_name: str, df: pd.DataFrame, hyperparams: dict = None,
                     cv_folds: int = 5, test_size: float = 0.2):
    hyperparams = hyperparams or {}
    X, y, num, cat = classification_features(df)
    pre = build_preprocessor(num, cat)
    estimator = CLASSIFIERS[model_name](**hyperparams)
    pipe = Pipeline([("pre", pre), ("est", estimator)])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1", n_jobs=-1)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                           stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "cv_auc_mean":  float(cv_auc.mean()),
        "cv_auc_std":   float(cv_auc.std()),
        "cv_f1_mean":   float(cv_f1.mean()),
        "test_auc":     float(roc_auc_score(yte, proba)),
        "test_f1":      float(f1_score(yte, pred)),
        "test_acc":     float(accuracy_score(yte, pred)),
        "confusion":    confusion_matrix(yte, pred),
        "y_test":       yte.values,
        "y_proba":      proba,
        "y_pred":       pred,
        "positive_rate": float(y.mean()),
        "feature_names": _get_feature_names(pipe, num, cat),
    }
    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════════
# 3. UNSUPERVISED — RFM customer segmentation via K-Means
# ══════════════════════════════════════════════════════════════════════════
def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features per customer.
    Recency  = days since last order (smaller = more recent)
    Frequency = number of distinct orders
    Monetary  = total revenue
    """
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if snapshot_date is None:
        snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency  =("Order Date",   lambda s: (snapshot_date - s.max()).days),
        Frequency=("Order ID",     "nunique"),
        Monetary =("Total Revenue", "sum"),
    ).reset_index()

    # Bring back name for display
    name_map = df.drop_duplicates("Customer ID").set_index("Customer ID")["Customer Name"]
    rfm["Customer Name"] = rfm["Customer ID"].map(name_map)
    return rfm


def kmeans_rfm(rfm: pd.DataFrame, k: int = 4, log_transform: bool = True):
    """Run K-Means on RFM features. Returns (rfm_with_cluster, model, scaler, silhouette)."""
    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if log_transform:
        feats["Frequency"] = np.log1p(feats["Frequency"])
        feats["Monetary"]  = np.log1p(feats["Monetary"].clip(lower=0))

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else np.nan

    out = rfm.copy()
    out["Cluster"] = labels

    # Label clusters by Monetary value, descending → tier names
    tier_order = (out.groupby("Cluster")["Monetary"].mean()
                     .sort_values(ascending=False).index.tolist())
    tier_names = ["Champions", "Loyal", "Potential", "At-Risk", "Hibernating",
                   "Lost", "Lost-2", "Lost-3"][:k]
    label_map = {c: tier_names[i] for i, c in enumerate(tier_order)}
    out["Segment"] = out["Cluster"].map(label_map)

    return out, km, scaler, float(sil)


def kmeans_elbow(rfm: pd.DataFrame, k_range=(2, 9), log_transform: bool = True):
    """Compute inertia and silhouette across k for the elbow / silhouette plot."""
    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if log_transform:
        feats["Frequency"] = np.log1p(feats["Frequency"])
        feats["Monetary"]  = np.log1p(feats["Monetary"].clip(lower=0))
    X = StandardScaler().fit_transform(feats)
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _get_feature_names(pipe: Pipeline, num: list, cat: list) -> list:
    """Recover post-encoding feature names for feature importance display."""
    pre = pipe.named_steps["pre"]
    out = list(num)
    if cat:
        ohe = pre.named_transformers_["cat"]
        try:
            cat_names = ohe.get_feature_names_out(cat)
        except Exception:
            cat_names = [f"{c}_{v}" for c in cat
                          for v in ohe.categories_[cat.index(c)]]
        out.extend(cat_names)
    return out


def feature_importance(pipe: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importance / coefficients from the trained pipeline."""
    est = pipe.named_steps["est"]
    feature_names = _get_feature_names(pipe,
                                        pipe.named_steps["pre"].transformers_[0][2],
                                        pipe.named_steps["pre"].transformers_[1][2])
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.atleast_2d(est.coef_)[0]
        imp = np.abs(coef)
    else:
        return pd.DataFrame()
    df = (pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    return df
