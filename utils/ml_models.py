"""
Machine Learning utilities for the ML Lab page.

Four blocks:
  1. Supervised MULTI-CLASS CLASSIFICATION — predict a profit/revenue tier per order
       Models: Logistic Regression, Random Forest, Gradient Boosting, SVC
       Eval: 5-fold stratified CV (macro F1, accuracy) + confusion matrix
  2. Supervised BINARY CLASSIFICATION — predict whether an order meets a binary condition
       Models: Logistic Regression, Random Forest, Gradient Boosting, SVC
       Eval: 5-fold stratified CV (ROC-AUC, F1) + confusion matrix
  3. Unsupervised CLUSTERING — RFM customer segmentation via K-Means
       Eval: silhouette score, elbow method
  4. Single-row prediction helper — run a trained pipeline on user-supplied inputs
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, StratifiedShuffleSplit,
    cross_val_score, train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    silhouette_score,
)


# ══════════════════════════════════════════════════════════════════════════
# Common feature builder
# ══════════════════════════════════════════════════════════════════════════
ML_NUMERIC = ["Quantity", "Total Revenue"]
ML_CATEG   = ["Ship Mode", "Segment", "Region", "Category", "Sub-Category"]

SVM_MAX_ROWS = 2_000   # hard cap — SVR/SVC is O(n²), cloud instances crash above this

# ── Target label registries ───────────────────────────────────────────────
MULTICLASS_TARGETS = {
    "Profit-Margin Tier  (Loss / Low / High)":    "margin_tier",
    "Order-Value Tier   (Small / Medium / Large)": "revenue_tier",
}

CLASSIFICATION_TARGETS = {
    "Unprofitable Order  (Total Profit < 0)":         "is_unprofitable",
    "High-Revenue Order  (Revenue > dataset median)": "is_high_revenue",
}


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])


def _engineer_features(feats: pd.DataFrame, num: list) -> list:
    """Add Rev_per_unit + seasonal signals in-place; return updated num list."""
    feats["Rev_per_unit"] = (
        feats["Total Revenue"] / feats["Quantity"].clip(lower=1)
    )
    num = num + ["Rev_per_unit"]

    if "Order Date" in feats.columns:
        dates = pd.to_datetime(feats["Order Date"], errors="coerce")
        feats["Order_Month"]   = dates.dt.month.fillna(0).astype(int)
        feats["Order_Quarter"] = dates.dt.quarter.fillna(0).astype(int)
        num = num + ["Order_Month", "Order_Quarter"]
    return num


# ══════════════════════════════════════════════════════════════════════════
# Shared classifier registry  (used by BOTH supervised tabs)
# ══════════════════════════════════════════════════════════════════════════
CLASSIFIERS = {
    "Logistic Regression": lambda **kw:
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest":       lambda n_estimators=100, max_depth=None, **kw:
        RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting":   lambda n_estimators=100, max_depth=3, learning_rate=0.1, **kw:
        GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   learning_rate=learning_rate, random_state=42),
    "SVC (RBF kernel)":    lambda C=1.0, **kw:
        SVC(kernel="rbf", C=C, probability=True, random_state=42, class_weight="balanced"),
}


def _maybe_subsample_svm(model_name, X, y, cv_folds):
    """If model is SVC and dataset is large, subsample + shrink folds."""
    if model_name != "SVC (RBF kernel)" or len(X) <= SVM_MAX_ROWS:
        return X, y, cv_folds, -1          # cv_jobs=-1 = parallel

    sss = StratifiedShuffleSplit(n_splits=1, train_size=SVM_MAX_ROWS, random_state=42)
    idx, _ = next(sss.split(X, y))
    return (X.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True),
            3, 1)                           # 3-fold, sequential


# ══════════════════════════════════════════════════════════════════════════
# 1. MULTI-CLASS CLASSIFICATION — predict a tier label per order
# ══════════════════════════════════════════════════════════════════════════
def multiclass_features(df: pd.DataFrame, target: str = "margin_tier"):
    """
    Build X / y for multi-class classification.

    target options:
      "margin_tier"  — Loss / Low Margin / High Margin  (based on Profit ÷ Revenue)
      "revenue_tier" — Small / Medium / Large order      (based on Total Revenue tertiles)
    """
    feats = df.copy()

    if target == "margin_tier":
        # Compute margin
        feats["_margin"] = np.where(
            feats["Total Revenue"].abs() > 1e-9,
            feats["Total Profit"] / feats["Total Revenue"],
            0.0,
        )
        # Trim extreme margins so bins are clean
        feats = feats[feats["_margin"].between(-2.0, 1.0)].reset_index(drop=True)
        feats["_target"] = pd.cut(
            feats["_margin"],
            bins=[-np.inf, 0.0, 0.15, np.inf],
            labels=["Loss", "Low Margin", "High Margin"],
        ).astype(str)
        target_col = "Profit-Margin Tier"
        classes    = ["Loss", "Low Margin", "High Margin"]

    elif target == "revenue_tier":
        q33, q67 = feats["Total Revenue"].quantile([1/3, 2/3])
        feats["_target"] = pd.cut(
            feats["Total Revenue"],
            bins=[-np.inf, q33, q67, np.inf],
            labels=["Small", "Medium", "Large"],
        ).astype(str)
        target_col = "Order-Value Tier"
        classes    = ["Small", "Medium", "Large"]

    else:
        raise ValueError(f"Unknown multiclass target: {target}")

    num = ["Quantity", "Total Revenue"]
    if "Shipping Days" in feats.columns:
        num.append("Shipping Days")
    num = _engineer_features(feats, num)

    cat = [c for c in ML_CATEG if c in feats.columns]
    X   = feats[num + cat].copy()
    y   = feats["_target"]
    return X, y, num, cat, target_col, classes


def train_multiclass(model_name: str, df: pd.DataFrame,
                     target: str = "margin_tier",
                     hyperparams: dict = None,
                     cv_folds: int = 5, test_size: float = 0.2):
    hyperparams = hyperparams or {}
    X, y, num, cat, target_col, classes = multiclass_features(df, target=target)
    X, y, cv_folds, cv_jobs = _maybe_subsample_svm(model_name, X, y, cv_folds)

    pre       = build_preprocessor(num, cat)
    estimator = CLASSIFIERS[model_name](**hyperparams)
    pipe      = Pipeline([("pre", pre), ("est", estimator)])

    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro",  n_jobs=cv_jobs)
    cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy",  n_jobs=cv_jobs)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                           stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    pred  = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)          # shape (n, n_classes)

    # Macro ROC-AUC (one-vs-rest) on test set
    le = LabelEncoder().fit(classes)
    y_bin = le.transform(yte)
    try:
        test_auc = float(roc_auc_score(
            y_bin, proba, multi_class="ovr", average="macro",
            labels=list(range(len(classes))),
        ))
    except Exception:
        test_auc = float("nan")

    metrics = {
        "cv_f1_mean":    float(cv_f1.mean()),
        "cv_f1_std":     float(cv_f1.std()),
        "cv_acc_mean":   float(cv_acc.mean()),
        "test_f1":       float(f1_score(yte, pred, average="macro", zero_division=0)),
        "test_acc":      float(accuracy_score(yte, pred)),
        "test_auc":      test_auc,
        "confusion":     confusion_matrix(yte, pred, labels=classes),
        "classes":       classes,
        "y_test":        yte.values,
        "y_pred":        pred,
        "y_proba":       proba,
        "class_dist":    {c: float((y == c).mean()) for c in classes},
        "feature_names": _get_feature_names(pipe, num, cat),
        # ── For prediction panel ─────────────────────────────────────────
        "num_cols":      num,
        "cat_cols":      cat,
        "cat_values":    {c: sorted(df[c].dropna().unique().tolist()) for c in cat},
        "num_defaults":  {c: float(df[c].median()) if c in df.columns else 1.0
                          for c in num},
        "target_col":    target_col,
        "target_key":    target,
    }
    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════════
# 2. BINARY CLASSIFICATION — predict a binary label per order
# ══════════════════════════════════════════════════════════════════════════
def classification_features(df: pd.DataFrame, target: str = "is_unprofitable"):
    """
    Build X / y for binary classification.

    target options:
      "is_unprofitable" — Total Profit < 0
      "is_high_revenue" — Total Revenue > dataset median
    """
    feats = df.copy()

    if target == "is_unprofitable":
        feats["_target"] = (feats["Total Profit"] < 0).astype(int)
        target_col = "Unprofitable Order"
    elif target == "is_high_revenue":
        med = feats["Total Revenue"].median()
        feats["_target"] = (feats["Total Revenue"] > med).astype(int)
        target_col = "High-Revenue Order"
    else:
        raise ValueError(f"Unknown classification target: {target}")

    num = ["Quantity", "Total Revenue"]
    if "Shipping Days" in feats.columns:
        num.append("Shipping Days")
    num = _engineer_features(feats, num)

    cat = [c for c in ML_CATEG if c in feats.columns]
    X   = feats[num + cat].copy()
    y   = feats["_target"]
    return X, y, num, cat, target_col


def train_classifier(model_name: str, df: pd.DataFrame,
                     target: str = "is_unprofitable",
                     hyperparams: dict = None,
                     cv_folds: int = 5, test_size: float = 0.2):
    hyperparams = hyperparams or {}
    X, y, num, cat, target_col = classification_features(df, target=target)
    X, y, cv_folds, cv_jobs = _maybe_subsample_svm(model_name, X, y, cv_folds)

    pre       = build_preprocessor(num, cat)
    estimator = CLASSIFIERS[model_name](**hyperparams)
    pipe      = Pipeline([("pre", pre), ("est", estimator)])

    skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=cv_jobs)
    cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1",      n_jobs=cv_jobs)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                           stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    metrics = {
        "cv_auc_mean":   float(cv_auc.mean()),
        "cv_auc_std":    float(cv_auc.std()),
        "cv_f1_mean":    float(cv_f1.mean()),
        "test_auc":      float(roc_auc_score(yte, proba)),
        "test_f1":       float(f1_score(yte, pred)),
        "test_acc":      float(accuracy_score(yte, pred)),
        "confusion":     confusion_matrix(yte, pred),
        "y_test":        yte.values,
        "y_proba":       proba,
        "y_pred":        pred,
        "positive_rate": float(y.mean()),
        "feature_names": _get_feature_names(pipe, num, cat),
        # ── For prediction panel ─────────────────────────────────────────
        "num_cols":      num,
        "cat_cols":      cat,
        "cat_values":    {c: sorted(df[c].dropna().unique().tolist()) for c in cat},
        "num_defaults":  {c: float(df[c].median()) if c in df.columns else 1.0
                          for c in num},
        "target_col":    target_col,
        "target_key":    target,
    }
    return pipe, metrics


# ══════════════════════════════════════════════════════════════════════════
# 3. UNSUPERVISED — RFM customer segmentation via K-Means
# ══════════════════════════════════════════════════════════════════════════
def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    df = df.copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if snapshot_date is None:
        snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency  =("Order Date",    lambda s: (snapshot_date - s.max()).days),
        Frequency=("Order ID",      "nunique"),
        Monetary =("Total Revenue", "sum"),
    ).reset_index()

    name_map = df.drop_duplicates("Customer ID").set_index("Customer ID")["Customer Name"]
    rfm["Customer Name"] = rfm["Customer ID"].map(name_map)
    return rfm


def kmeans_rfm(rfm: pd.DataFrame, k: int = 4, log_transform: bool = True):
    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if log_transform:
        feats["Frequency"] = np.log1p(feats["Frequency"])
        feats["Monetary"]  = np.log1p(feats["Monetary"].clip(lower=0))

    scaler = StandardScaler()
    X      = scaler.fit_transform(feats)

    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels) if k > 1 else np.nan

    out = rfm.copy()
    out["Cluster"] = labels

    tier_order = (out.groupby("Cluster")["Monetary"].mean()
                     .sort_values(ascending=False).index.tolist())
    tier_names = ["Champions", "Loyal", "Potential", "At-Risk",
                  "Hibernating", "Lost", "Lost-2", "Lost-3"][:k]
    label_map  = {c: tier_names[i] for i, c in enumerate(tier_order)}
    out["Segment"] = out["Cluster"].map(label_map)

    return out, km, scaler, float(sil)


def kmeans_elbow(rfm: pd.DataFrame, k_range=(2, 9), log_transform: bool = True):
    feats = rfm[["Recency", "Frequency", "Monetary"]].copy()
    if log_transform:
        feats["Frequency"] = np.log1p(feats["Frequency"])
        feats["Monetary"]  = np.log1p(feats["Monetary"].clip(lower=0))
    X = StandardScaler().fit_transform(feats)
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k":          k,
            "inertia":    float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 4. Single-row prediction helpers
# ══════════════════════════════════════════════════════════════════════════
def predict_single(pipe: Pipeline, input_dict: dict,
                   num_cols: list, cat_cols: list) -> np.ndarray:
    """Run a trained pipeline on a single observation, returning the predicted label."""
    row = pd.DataFrame([{k: input_dict[k] for k in num_cols + cat_cols}])
    return pipe.predict(row)


def predict_single_proba(pipe: Pipeline, input_dict: dict,
                          num_cols: list, cat_cols: list) -> np.ndarray:
    """Same as predict_single but returns class probabilities (shape: 1 × n_classes)."""
    row = pd.DataFrame([{k: input_dict[k] for k in num_cols + cat_cols}])
    return pipe.predict_proba(row)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _get_feature_names(pipe: Pipeline, num: list, cat: list) -> list:
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
    est          = pipe.named_steps["est"]
    feature_names = _get_feature_names(
        pipe,
        pipe.named_steps["pre"].transformers_[0][2],
        pipe.named_steps["pre"].transformers_[1][2],
    )
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        coef = np.atleast_2d(est.coef_)
        imp  = np.abs(coef).mean(axis=0)   # average over classes for multiclass
    else:
        return pd.DataFrame()
    df = (pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    return df
