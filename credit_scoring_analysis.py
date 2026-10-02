import numpy as np
import pandas as pd
import gc
import os
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import json
import shap

# Project Credit Scoring Policy Fairness Analysis
# Evaluates credit allocation fairness across economic segments

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration and Helpers
BASE_DIR = r"data"


def p(name: str) -> str:
    return os.path.join(BASE_DIR, f"{name}.csv")


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


def sanitize_feature_names(cols):
    # Remove invalid JSON characters
    bad = r'[\[\]\{\}",:]'
    cleaned = []
    for c in cols:
        s = str(c)
        s = re.sub(bad, "_", s)
        s = re.sub(r"\s+", "_", s)
        s = s.strip("_")
        cleaned.append(s)

    # Ensure unique feature names
    seen = {}
    unique = []
    for s in cleaned:
        if s not in seen:
            seen[s] = 0
            unique.append(s)
        else:
            seen[s] += 1
            unique.append(f"{s}__{seen[s]}")
    return unique


def one_hot_encoder(df, nan_as_category=True):
    # One hot encode categorical columns
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Data Preprocessing
def application_train_test(num_rows=None, nan_as_category=False):
    # Process application train and test
    df = pd.read_csv(p("application_train"), nrows=num_rows)
    test_df = pd.read_csv(p("application_test"), nrows=num_rows)
    df = pd.concat([df, test_df], axis=0, ignore_index=True)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Binary encode categorical features
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # One hot encode categorical features
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # Replace 365243 with NaN in DAYS_EMPLOYED
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Create proportional features
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    del test_df
    gc.collect()
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    # Process bureau and bureau balance
    bureau = pd.read_csv(p("bureau"), nrows=num_rows)
    bb = pd.read_csv(p("bureau_balance"), nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Aggregate bureau balance
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Aggregate numeric and categorical features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Aggregate active credits
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Aggregate closed credits
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


def previous_applications(num_rows=None, nan_as_category=True):
    # Process previous applications
    prev = pd.read_csv(p("previous_application"), nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # Replace 365243 with NaN
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Calculate application to credit ratio
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Aggregate approved applications
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Aggregate refused applications
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    # Process POS CASH balance
    pos = pd.read_csv(p("POS_CASH_balance"), nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(num_rows=None, nan_as_category=True):
    # Process installments payments
    ins = pd.read_csv(p("installments_payments"), nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    # Calculate payment percentage and difference
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Calculate days past due and days before due
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(num_rows=None, nan_as_category=True):
    # Process credit card balance
    cc = pd.read_csv(p("credit_card_balance"), nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    # Drop SK_ID_PREV
    if 'SK_ID_PREV' in cc.columns:
        cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    # Convert categorical columns to numeric
    for c in cat_cols:
        if c in cc.columns:
            cc[c] = pd.to_numeric(cc[c], errors='coerce')

    # Select numeric columns
    id_cols = {'SK_ID_CURR'}
    num_cols = [c for c in cc.columns if c not in id_cols and c not in cat_cols]

    # Define aggregations
    aggs = {}
    for c in num_cols:
        aggs[c] = ['min', 'max', 'mean', 'sum', 'var']
    for c in cat_cols:
        if c in cc.columns:
            aggs[c] = ['mean']

    cc_agg = cc.groupby('SK_ID_CURR').agg(aggs)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


# Model Evaluation and Visualization
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
        :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig('outputs/lgbm_importances01.png')


def _ensure_outputs_dir():
    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def _plot_roc(y_true, y_score, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} | AUC={auc:.4f}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return auc


def _ks_curve(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False).reset_index(drop=True)
    df["cum_pos"] = df["y"].cumsum()
    df["cum_neg"] = (1 - df["y"]).cumsum()
    total_pos = df["y"].sum()
    total_neg = (1 - df["y"]).sum()
    df["tpr"] = df["cum_pos"] / (total_pos if total_pos > 0 else 1.0)
    df["fpr"] = df["cum_neg"] / (total_neg if total_neg > 0 else 1.0)
    df["ks"] = (df["tpr"] - df["fpr"]).abs()
    ks_max_idx = int(df["ks"].values.argmax())
    ks_value = float(df.loc[ks_max_idx, "ks"])
    ks_thr = float(df.loc[ks_max_idx, "s"])
    return df, ks_value, ks_thr


def _plot_ks(y_true, y_score, out_path, title):
    df, ks_value, ks_thr = _ks_curve(y_true, y_score)
    plt.figure()
    plt.plot(df["tpr"].values, label="TPR")
    plt.plot(df["fpr"].values, label="FPR")
    plt.plot(df["ks"].values, label="|TPR-FPR|")
    plt.axvline(x=df.index[df["s"] >= ks_thr].max() if (df["s"] >= ks_thr).any() else 0, linestyle="--")
    plt.xlabel("Sorted samples")
    plt.ylabel("Rate")
    plt.title(f"{title} | KS={ks_value:.4f} @ thr={ks_thr:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return ks_value, ks_thr


def _plot_calibration(y_true, y_score, out_path, title, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{title} | Brier={brier:.4f}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return brier


def _psi_from_reference(reference, compare, n_bins=10, eps=1e-6):
    ref = np.asarray(reference, dtype=float)
    cmp = np.asarray(compare, dtype=float)
    qs = np.quantile(ref[~np.isnan(ref)], q=np.linspace(0, 1, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    ref_bins = np.digitize(ref, qs[1:-1], right=True)
    cmp_bins = np.digitize(cmp, qs[1:-1], right=True)
    ref_counts = np.bincount(ref_bins, minlength=n_bins).astype(float)
    cmp_counts = np.bincount(cmp_bins, minlength=n_bins).astype(float)
    ref_pct = ref_counts / max(ref_counts.sum(), 1.0)
    cmp_pct = cmp_counts / max(cmp_counts.sum(), 1.0)
    ref_pct = np.clip(ref_pct, eps, 1.0)
    cmp_pct = np.clip(cmp_pct, eps, 1.0)
    psi = float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))
    return psi, ref_pct, cmp_pct


def _plot_psi(reference, compare, out_path, title, n_bins=10):
    psi, ref_pct, cmp_pct = _psi_from_reference(reference, compare, n_bins=n_bins)
    x = np.arange(n_bins)
    width = 0.4
    plt.figure()
    plt.bar(x - width / 2, ref_pct, width=width, label="Expected")
    plt.bar(x + width / 2, cmp_pct, width=width, label="Actual")
    plt.xlabel("PSI bin index")
    plt.ylabel("Proportion")
    plt.title(f"{title} | PSI={psi:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return psi


def _cost_curve(y_true, y_score, cost_fp=1.0, cost_fn=10.0, n_grid=201):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.linspace(0.0, 1.0, n_grid)
    costs = np.zeros_like(thresholds)
    stats = []
    for i, thr in enumerate(thresholds):
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        total_cost = cost_fp * fp + cost_fn * fn
        costs[i] = total_cost
        stats.append((thr, tn, fp, fn, tp, total_cost))
    best_i = int(costs.argmin())
    best = stats[best_i]
    return thresholds, costs, best


def _plot_cost_and_confusion(y_true, y_score, out_cost_path, out_cm_path, title, cost_fp=1.0, cost_fn=10.0):
    thresholds, costs, best = _cost_curve(y_true, y_score, cost_fp=cost_fp, cost_fn=cost_fn)
    best_thr, tn, fp, fn, tp, best_cost = best

    plt.figure()
    plt.plot(thresholds, costs)
    plt.axvline(best_thr, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title(f"{title} | Cost curve")
    plt.tight_layout()
    plt.savefig(out_cost_path)
    plt.close()

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    plt.figure()
    plt.imshow(cm, cmap="Blues", vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{int(v)}", ha="center", va="center", color="black")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{title} | Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_cm_path)
    plt.close()

    return float(best_thr), float(best_cost), int(tn), int(fp), int(fn), int(tp)


def _run_requested_analyses(y_true, oof_preds, sub_preds, outputs_dir, model_name="LightGBM", cost_fp=1.0,
                            cost_fn=10.0):
    tag = _now_tag()

    roc_path = os.path.join(outputs_dir, f"lgb_roc_{tag}.png")
    ks_path = os.path.join(outputs_dir, f"lgb_ks_{tag}.png")
    cal_path = os.path.join(outputs_dir, f"lgb_calibration_{tag}.png")
    psi_train_test_path = os.path.join(outputs_dir, f"lgb_psi_train_test_{tag}.png")
    cost_path = os.path.join(outputs_dir, f"lgb_cost_curve_{tag}.png")
    cm_path = os.path.join(outputs_dir, f"lgb_confusion_{tag}.png")

    auc = _plot_roc(y_true, oof_preds, roc_path, f"{model_name} ROC")
    ks_value, ks_thr = _plot_ks(y_true, oof_preds, ks_path, f"{model_name} KS")
    brier = _plot_calibration(y_true, oof_preds, cal_path, f"{model_name} Calibration")
    psi_train_test = _plot_psi(oof_preds, sub_preds, psi_train_test_path, f"{model_name} Score PSI")

    best_thr, best_cost, tn, fp, fn, tp = _plot_cost_and_confusion(
        y_true, oof_preds,
        cost_path, cm_path,
        f"{model_name}",
        cost_fp=cost_fp, cost_fn=cost_fn
    )

    summary = pd.DataFrame([{
        "model": model_name,
        "auc_oof": auc,
        "ks_oof": ks_value,
        "ks_thr_oof": ks_thr,
        "brier_oof": brier,
        "psi_oof_vs_test": psi_train_test,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "best_thr_cost": best_thr,
        "best_cost": best_cost,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }])

    summary_path = os.path.join(outputs_dir, f"model_results_{tag}.csv")
    summary.to_csv(summary_path, index=False)

    return summary_path


def _read_best_thr_from_summary(summary_path: str, fallback: float = 0.085) -> float:
    try:
        s = pd.read_csv(summary_path)
        if "best_thr_cost" in s.columns and len(s) > 0:
            v = float(s.loc[0, "best_thr_cost"])
            if np.isfinite(v):
                return v
    except Exception:
        pass
    return float(fallback)


# Part 1 SHAP Analysis
def _shap_analysis_lgbm_tree(models, X_train_df, feature_names, outputs_dir,
                             max_samples=4000, random_state=1001, max_display=40):
    n = len(X_train_df)
    if n == 0:
        return None

    # Sample data to reduce runtime
    rng = np.random.RandomState(random_state)
    if n > max_samples:
        sample_idx = rng.choice(np.arange(n), size=max_samples, replace=False)
        Xs = X_train_df.iloc[sample_idx].copy()
    else:
        Xs = X_train_df.copy()

    # Convert data to numeric
    Xs = Xs.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    shap_sum = None

    # Average SHAP values across models
    for m in models:
        explainer = shap.TreeExplainer(m.booster_, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(Xs, check_additivity=False)

        # Extract positive class SHAP values
        if isinstance(sv, list) and len(sv) > 1:
            sv = sv[1]
        sv = np.asarray(sv)

        if shap_sum is None:
            shap_sum = sv
        else:
            shap_sum += sv

    shap_avg = shap_sum / max(len(models), 1)

    tag = _now_tag()
    beeswarm_path = os.path.join(outputs_dir, f"shap_beeswarm_{tag}.png")
    bar_path = os.path.join(outputs_dir, f"shap_bar_{tag}.png")
    imp_csv_path = os.path.join(outputs_dir, f"shap_mean_abs_{tag}.csv")

    shap.summary_plot(shap_avg, Xs, feature_names=feature_names, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_avg, Xs, feature_names=feature_names, show=False, plot_type="bar", max_display=max_display)
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(shap_avg), axis=0)
    shap_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(imp_csv_path, index=False)

    top_feats = shap_imp["feature"].head(3).tolist()
    for f in top_feats:
        try:
            dep_path = os.path.join(outputs_dir, f"shap_dependence_{f}_{tag}.png")
            shap.dependence_plot(f, shap_avg, Xs, show=False, interaction_index=None)
            plt.tight_layout()
            plt.savefig(dep_path, bbox_inches="tight")
            plt.close()
        except Exception:
            try:
                plt.close()
            except Exception:
                pass

    return {
        "beeswarm_png": beeswarm_path,
        "bar_png": bar_path,
        "mean_abs_csv": imp_csv_path
    }


# Part 2 Segmentation and Fairness Analysis
def _safe_income_per_person(df: pd.DataFrame, preferred_col: str = "INCOME_PER_PERSON") -> pd.Series:
    if preferred_col in df.columns:
        x = pd.to_numeric(df[preferred_col], errors="coerce")
    else:
        if ("AMT_INCOME_TOTAL" in df.columns) and ("CNT_FAM_MEMBERS" in df.columns):
            denom = pd.to_numeric(df["CNT_FAM_MEMBERS"], errors="coerce").replace(0, np.nan)
            x = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") / denom
        elif "AMT_INCOME_TOTAL" in df.columns:
            x = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce")
        else:
            x = pd.Series(np.nan, index=df.index)

    x = x.replace([np.inf, -np.inf], np.nan)
    med = float(np.nanmedian(x.values)) if np.isfinite(np.nanmedian(x.values)) else 0.0
    return x.fillna(med)


def _segment_test_three_groups_by_income(test_df: pd.DataFrame,
                                         income_col: str = "INCOME_PER_PERSON",
                                         low_mult: float = 0.60,
                                         high_mult: float = 2.00):
    income = _safe_income_per_person(test_df, preferred_col=income_col)

    med = float(np.nanmedian(income.values)) if np.isfinite(np.nanmedian(income.values)) else 0.0
    low_thr = float(low_mult) * float(med)
    high_thr = float(high_mult) * float(med)

    groups = pd.Series("MID", index=test_df.index, dtype=object)
    groups.loc[income < low_thr] = "LOW"
    groups.loc[income > high_thr] = "HIGH"

    return groups, income, med, low_thr, high_thr


def _expected_error_types_from_pd(pd_scores: np.ndarray, cutoff: float):
    s = np.asarray(pd_scores, dtype=float)
    s = np.clip(s, 0.0, 1.0)

    reject = (s >= float(cutoff))
    accept = ~reject

    exp_bad = float(np.sum(s))
    exp_good = float(np.sum(1.0 - s))

    # Calculate expected values
    exp_good_reject = float(np.sum((1.0 - s) * reject))
    exp_bad_accept = float(np.sum(s * accept))

    good_rej_rate = exp_good_reject / exp_good if exp_good > 0 else np.nan
    bad_acc_rate = exp_bad_accept / exp_bad if exp_bad > 0 else np.nan

    reject_rate = float(np.mean(reject)) if len(reject) > 0 else np.nan

    return {
        "n": int(len(s)),
        "reject_rate": reject_rate,
        "exp_good": exp_good,
        "exp_bad": exp_bad,
        "exp_good_reject": exp_good_reject,
        "exp_bad_accept": exp_bad_accept,
        "exp_good_rejection_rate": float(good_rej_rate) if np.isfinite(good_rej_rate) else np.nan,
        "exp_bad_acceptance_rate": float(bad_acc_rate) if np.isfinite(bad_acc_rate) else np.nan,
    }


def _stage2_test_only_policy_analysis(test_df: pd.DataFrame, sub_preds: np.ndarray, outputs_dir: str,
                                      cutoff: float, income_col: str = "INCOME_PER_PERSON", n_each: int = 4874):
    tag = _now_tag()

    groups, income, med_income, low_thr, high_thr = _segment_test_three_groups_by_income(
        test_df, income_col=income_col, low_mult=0.60, high_mult=2.00
    )

    # Plot income distribution
    income_dist_path = os.path.join(outputs_dir, f"stage2_test_income_per_person_dist_{tag}.png")
    plt.figure()
    vals = income.values
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=60, density=True, alpha=0.8)
    plt.axvline(low_thr, linestyle="--")
    plt.axvline(high_thr, linestyle="--")
    plt.xlabel(income_col)
    plt.ylabel("Density")
    plt.title(f"Test distribution {income_col}")
    plt.tight_layout()
    plt.savefig(income_dist_path)
    plt.close()

    s = np.asarray(sub_preds, dtype=float)
    low_mask = (groups.values == "LOW")
    mid_mask = (groups.values == "MID")
    high_mask = (groups.values == "HIGH")

    low_scores = s[low_mask]
    mid_scores = s[mid_mask]
    high_scores = s[high_mask]

    # Plot PD distribution and rates
    pd_hist_path = os.path.join(outputs_dir, f"stage2_policy_pd_hist_three_groups_test_{tag}.png")
    plt.figure()
    bins = np.linspace(0, 1, 41)
    plt.hist(high_scores, bins=bins, density=True, alpha=0.6, label="HIGH income")
    plt.hist(mid_scores, bins=bins, density=True, alpha=0.6, label="MID income")
    plt.hist(low_scores, bins=bins, density=True, alpha=0.6, label="LOW income")
    plt.axvline(float(cutoff), linestyle="--")
    plt.xlabel("PD score")
    plt.ylabel("Density")
    plt.title("PD distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pd_hist_path)
    plt.close()

    reject_bar_path = os.path.join(outputs_dir, f"stage2_policy_reject_rate_three_groups_test_{tag}.png")
    rr_high = float(np.mean(high_scores >= cutoff)) if len(high_scores) else np.nan
    rr_mid = float(np.mean(mid_scores >= cutoff)) if len(mid_scores) else np.nan
    rr_low = float(np.mean(low_scores >= cutoff)) if len(low_scores) else np.nan
    plt.figure()
    plt.bar(["HIGH income", "MID income", "LOW income"], [rr_high, rr_mid, rr_low])
    plt.ylabel("Reject rate")
    plt.title("Reject rate by income group")
    plt.tight_layout()
    plt.savefig(reject_bar_path)
    plt.close()

    err_bar_path = os.path.join(outputs_dir, f"stage2_policy_error_types_three_groups_test_{tag}.png")
    m_high = _expected_error_types_from_pd(high_scores, cutoff)
    m_mid = _expected_error_types_from_pd(mid_scores, cutoff)
    m_low = _expected_error_types_from_pd(low_scores, cutoff)

    plt.figure()
    x = np.arange(3)
    w = 0.35
    plt.bar(x - w / 2,
            [m_high["exp_good_rejection_rate"], m_mid["exp_good_rejection_rate"], m_low["exp_good_rejection_rate"]],
            width=w, label="Expected Good rejection rate")
    plt.bar(x + w / 2,
            [m_high["exp_bad_acceptance_rate"], m_mid["exp_bad_acceptance_rate"], m_low["exp_bad_acceptance_rate"]],
            width=w, label="Expected Bad acceptance rate")
    plt.xticks(x, ["HIGH income", "MID income", "LOW income"])
    plt.ylabel("Rate")
    plt.title("Expected error type rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(err_bar_path)
    plt.close()

    # Create summary table image
    table_path = os.path.join(outputs_dir, f"stage2_policy_summary_table_three_groups_test_{tag}.png")
    table_df = pd.DataFrame([
        {"Group": "HIGH", "n": m_high["n"], "Reject_rate": m_high["reject_rate"],
         "Exp_GoodRej_rate": m_high["exp_good_rejection_rate"], "Exp_BadAcc_rate": m_high["exp_bad_acceptance_rate"]},
        {"Group": "MID", "n": m_mid["n"], "Reject_rate": m_mid["reject_rate"],
         "Exp_GoodRej_rate": m_mid["exp_good_rejection_rate"], "Exp_BadAcc_rate": m_mid["exp_bad_acceptance_rate"]},
        {"Group": "LOW", "n": m_low["n"], "Reject_rate": m_low["reject_rate"],
         "Exp_GoodRej_rate": m_low["exp_good_rejection_rate"], "Exp_BadAcc_rate": m_low["exp_bad_acceptance_rate"]},
        {"Group": "LOW-HIGH gap", "n": "",
         "Reject_rate": (m_low["reject_rate"] - m_high["reject_rate"]) if (
                     np.isfinite(m_low["reject_rate"]) and np.isfinite(m_high["reject_rate"])) else np.nan,
         "Exp_GoodRej_rate": (m_low["exp_good_rejection_rate"] - m_high["exp_good_rejection_rate"]) if (
                     np.isfinite(m_low["exp_good_rejection_rate"]) and np.isfinite(
                 m_high["exp_good_rejection_rate"])) else np.nan,
         "Exp_BadAcc_rate": (m_low["exp_bad_acceptance_rate"] - m_high["exp_bad_acceptance_rate"]) if (
                     np.isfinite(m_low["exp_bad_acceptance_rate"]) and np.isfinite(
                 m_high["exp_bad_acceptance_rate"])) else np.nan},
    ])

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    ax.set_title("Policy summary table")
    display_df = table_df.copy()
    for c in ["Reject_rate", "Exp_GoodRej_rate", "Exp_BadAcc_rate"]:
        display_df[c] = display_df[c].apply(lambda v: "" if v == "" else (f"{v:.6f}" if pd.notnull(v) else ""))
    tbl = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(table_path)
    plt.close(fig)

    # Save results to JSON
    json_path = os.path.join(outputs_dir, f"stage2_policy_summary_three_groups_test_{tag}.json")
    payload = {
        "cutoff": float(cutoff),
        "income_col": income_col,
        "median_income": float(med_income),
        "low_mult": 0.60,
        "high_mult": 2.00,
        "low_thr": float(low_thr),
        "high_thr": float(high_thr),
        "artifacts": {
            "income_dist_png": income_dist_path,
            "pd_hist_png": pd_hist_path,
            "reject_rate_png": reject_bar_path,
            "error_types_png": err_bar_path,
            "summary_table_png": table_path
        },
        "metrics": {
            "HIGH": m_high,
            "MID": m_mid,
            "LOW": m_low
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return groups, income, {
        "income_dist_png": income_dist_path,
        "pd_hist_png": pd_hist_path,
        "reject_rate_png": reject_bar_path,
        "error_types_png": err_bar_path,
        "summary_table_png": table_path,
        "summary_json": json_path
    }


# Part 3 Scenario Analysis
def _predict_ensemble(models, X_df):
    preds = np.zeros(len(X_df), dtype=float)
    n = max(len(models), 1)
    for m in models:
        it = m.best_iteration_ if hasattr(m, "best_iteration_") else None
        if it is None:
            preds += m.predict_proba(X_df)[:, 1] / n
        else:
            preds += m.predict_proba(X_df, num_iteration=it)[:, 1] / n
    return preds


def _apply_credit_multiplier(df: pd.DataFrame, mult: float):
    d = df.copy()
    if "AMT_CREDIT" in d.columns:
        d["AMT_CREDIT"] = pd.to_numeric(d["AMT_CREDIT"], errors="coerce") * float(mult)

    if ("AMT_INCOME_TOTAL" in d.columns) and ("AMT_CREDIT" in d.columns) and ("INCOME_CREDIT_PERC" in d.columns):
        denom = pd.to_numeric(d["AMT_CREDIT"], errors="coerce").replace(0, np.nan)
        d["INCOME_CREDIT_PERC"] = pd.to_numeric(d["AMT_INCOME_TOTAL"], errors="coerce") / denom

    if ("AMT_ANNUITY" in d.columns) and ("AMT_CREDIT" in d.columns) and ("PAYMENT_RATE" in d.columns):
        denom = pd.to_numeric(d["AMT_CREDIT"], errors="coerce").replace(0, np.nan)
        d["PAYMENT_RATE"] = pd.to_numeric(d["AMT_ANNUITY"], errors="coerce") / denom

    d = d.replace([np.inf, -np.inf], np.nan)
    return d


def _stage3_scenario_analysis(models, test_df: pd.DataFrame, feats: list, groups: pd.Series,
                              outputs_dir: str, cutoff: float, sub_preds: np.ndarray):
    tag = _now_tag()

    results = {}
    base_scores = np.asarray(sub_preds, dtype=float)

    for grp in ["LOW", "MID", "HIGH"]:
        mask = (groups.values == grp)
        df_g = test_df.loc[mask].copy()
        if len(df_g) == 0:
            continue

        scenario_rows = []
        for name, mult in [("down", 0.8), ("normal", 1.0), ("up", 1.2)]:
            if name == "normal":
                # Use baseline predictions for normal scenario
                pd_scores = base_scores[mask]
            else:
                df_s = _apply_credit_multiplier(df_g, mult)
                # Convert scenario data to numeric
                Xs = df_s[feats].apply(pd.to_numeric, errors="coerce")
                pd_scores = _predict_ensemble(models, Xs)

            m = _expected_error_types_from_pd(pd_scores, cutoff)

            scenario_rows.append({
                "Group": grp,
                "Scenario": name,
                "Credit_multiplier": float(mult),
                "n": m["n"],
                "Reject_rate": m["reject_rate"],
                "Exp_GoodRejection_rate": m["exp_good_rejection_rate"],
                "Exp_BadAcceptance_rate": m["exp_bad_acceptance_rate"],
                "Exp_GoodReject_count": m["exp_good_reject"],
                "Exp_BadAccept_count": m["exp_bad_accept"],
            })

        df_out = pd.DataFrame(scenario_rows)
        csv_path = os.path.join(outputs_dir, f"stage3_scenario_{grp.lower()}_{tag}.csv")
        df_out.to_csv(csv_path, index=False)

        table_path = os.path.join(outputs_dir, f"stage3_scenario_{grp.lower()}_table_{tag}.png")
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.axis("off")
        ax.set_title(f"Scenario analysis Group={grp}")

        disp = df_out.copy()
        for c in ["Reject_rate", "Exp_GoodRejection_rate", "Exp_BadAcceptance_rate"]:
            disp[c] = disp[c].apply(lambda v: f"{v:.6f}" if pd.notnull(v) else "")
        for c in ["Exp_GoodReject_count", "Exp_BadAccept_count"]:
            disp[c] = disp[c].apply(lambda v: f"{v:.2f}" if pd.notnull(v) else "")

        tbl = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.4)
        plt.tight_layout()
        plt.savefig(table_path)
        plt.close(fig)

        json_path = os.path.join(outputs_dir, f"stage3_scenario_{grp.lower()}_{tag}.json")
        payload = {
            "cutoff": float(cutoff),
            "group": grp,
            "artifacts": {"csv": csv_path, "table_png": table_path},
            "rows": df_out.to_dict(orient="records")
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        results[grp] = {"csv": csv_path, "table_png": table_path, "json": json_path}

    return results


# Part 4 LightGBM Model Training
def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # Split train and test data
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()

    del df
    gc.collect()

    def sanitize_cols(columns):
        # Sanitize column names
        bad = r'[\[\]\{\}",:]'
        cleaned = []
        for c in columns:
            s = str(c)
            s = s.replace("\ufeff", "").strip()
            s = re.sub(bad, "_", s)
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"__+", "_", s)
            s = s.strip("_")
            cleaned.append(s)

        # Make names unique
        seen = {}
        out = []
        for s in cleaned:
            if s not in seen:
                seen[s] = 0
                out.append(s)
            else:
                seen[s] += 1
                out.append(f"{s}__{seen[s]}")
        return out

    orig_cols = train_df.columns.tolist()
    new_cols = sanitize_cols(orig_cols)
    col_map = dict(zip(orig_cols, new_cols))
    train_df.rename(columns=col_map, inplace=True)
    test_df.rename(columns=col_map, inplace=True)

    # Initialize cross validation
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    # Define features
    drop_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    feats = [f for f in train_df.columns if f not in drop_cols]

    # Convert all features to numeric
    train_df[feats] = train_df[feats].apply(pd.to_numeric, errors='coerce')
    test_df[feats] = test_df[feats].apply(pd.to_numeric, errors='coerce')

    # Initialize storage arrays
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    fold_results = []
    import lightgbm as lgb

    # Store models for scenario analysis
    models = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x = train_df[feats].iloc[train_idx]
        train_y = train_df['TARGET'].iloc[train_idx]
        valid_x = train_df[feats].iloc[valid_idx]
        valid_y = train_df['TARGET'].iloc[valid_idx]

        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            verbose=-1
        )

        clf.fit(
            train_x, train_y,
            eval_set=[(valid_x, valid_y)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=200)
            ]
        )

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])

        fold_results.append({
            "fold": n_fold + 1,
            "auc": fold_auc,
            "best_iteration": int(clf.best_iteration_) if clf.best_iteration_ is not None else np.nan,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx))
        })

        fold_importance_df = pd.DataFrame({
            "feature": feats,
            "importance": clf.feature_importances_,
            "fold": n_fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        models.append(clf)

        del train_x, train_y, valid_x, valid_y
        gc.collect()

    # Save submission and plot importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)

    outputs_dir = _ensure_outputs_dir()
    display_importances(feature_importance_df)

    fold_results_df = pd.DataFrame(fold_results)
    tag = _now_tag()
    fold_results_path = os.path.join(outputs_dir, f"fold_results_{tag}.csv")
    fold_results_df.to_csv(fold_results_path, index=False)

    # Run evaluation metrics
    summary_path = _run_requested_analyses(
        y_true=train_df['TARGET'].values.astype(int),
        oof_preds=oof_preds,
        sub_preds=sub_preds,
        outputs_dir=outputs_dir,
        model_name="LightGBM",
        cost_fp=1.0,
        cost_fn=10.0
    )

    # Run SHAP analysis
    _shap_analysis_lgbm_tree(
        models=models,
        X_train_df=train_df[feats].fillna(0.0),
        feature_names=feats,
        outputs_dir=outputs_dir,
        max_samples=4000,
        random_state=1001,
        max_display=40
    )

    cutoff = _read_best_thr_from_summary(summary_path, fallback=0.085)

    # Run policy fairness analysis
    groups_test, income_test, stage2_artifacts = _stage2_test_only_policy_analysis(
        test_df=test_df,
        sub_preds=sub_preds,
        outputs_dir=outputs_dir,
        cutoff=cutoff,
        income_col="INCOME_PER_PERSON",
        n_each=4874
    )

    # Run scenario analysis
    _stage3_scenario_analysis(
        models=models,
        test_df=test_df,
        feats=feats,
        groups=groups_test,
        outputs_dir=outputs_dir,
        cutoff=cutoff,
        sub_preds=sub_preds
    )

    return feature_importance_df, fold_results_df, summary_path


# Main Execution
def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run LightGBM with kfold"):
        feat_importance, fold_results_df, summary_path = kfold_lightgbm(df, num_folds=10, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()