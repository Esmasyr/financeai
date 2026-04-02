"""
FinanceAI


"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Opsiyonel kütüphaneler ────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ── Loglama ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("financeai")

# ── Sabitler ──────────────────────────────────────────────────────────────────
RANDOM_SEED         = 42
CHURN_INACTIVE_DAYS = 90
MAX_TRAIN_ROWS      = 500_000
BATCH_SIZE          = 100_000
CV_FOLDS            = 5

# ── Yollar ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "data"
DB_PATH    = DATA_PATH / "financeai.db"


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCILAR
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _timer(label: str):
    """Kod bloğunun süresini loglar."""
    t0 = time.time()
    yield
    log.info("  ⏱  %s — %.1fs", label, time.time() - t0)


def _section(title: str) -> None:
    log.info("")
    log.info("─" * 64)
    log.info("  %s", title)
    log.info("─" * 64)


def _clean_currency(series: pd.Series) -> pd.Series:
    """'$1,234.56' → 1234.56. Dönüştürülemeyen değer → 0."""
    return (
        pd.to_numeric(
            series.astype(str)
                  .str.replace("$", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .str.strip(),
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)          # negatif limit mantıklı değil
    )


def _log_series(name: str, s: pd.Series) -> None:
    log.info(
        "  %-32s  mean=%.4f  std=%.4f  max=%.4f  >0.5=%d",
        name, s.mean(), s.std(), s.max(), (s > 0.5).sum(),
    )


def _mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def _validate_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    """Zorunlu sütunların varlığını doğrular, eksikse ValueError fırlatır."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Eksik sütunlar: {missing}")


def _git_hash() -> str:
    """Mevcut git commit hash'ini döner (varsa)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=BASE_DIR,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _df_fingerprint(df: pd.DataFrame) -> str:
    """DataFrame içeriğinin kısa hash'i — reproducibility kontrolü için."""
    raw = str(df.shape) + str(df.columns.tolist()) + str(df.iloc[0].tolist() if len(df) > 0 else "")
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════════════════════
# 1. İŞLEM VERİSİ
# ══════════════════════════════════════════════════════════════════════════════

def load_fraud_ids() -> set[int]:
    """
    train_fraud_labels.json → fraud işlem id seti.
    Desteklenen formatlar:
        {"target": {"123": "yes", "456": "1"}}
        {"123": 1, "456": 0}
        [123, 456]   (id listesi)
    """
    label_path = DATA_PATH / "train_fraud_labels.json"
    if not label_path.exists():
        log.warning("  train_fraud_labels.json bulunamadı — tüm etiketler 0.")
        return set()

    with open(label_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Format 1: liste
    if isinstance(raw, list):
        fraud_ids = {int(x) for x in raw}
    else:
        labels = raw.get("target", raw)
        fraud_ids = {
            int(k)
            for k, v in labels.items()
            if str(v).strip().lower() in {"yes", "1", "true"}
        }

    if not fraud_ids:
        log.warning(
            "  train_fraud_labels.json yüklendi ama pozitif etiket yok. "
            "Format: {\"target\": {\"id\": \"yes\"}} veya [id1, id2, ...]"
        )
    else:
        log.info("  Fraud etiket: %d", len(fraud_ids))

    return fraud_ids


def _process_tx_chunk(chunk: pd.DataFrame, fraud_ids: set[int]) -> pd.DataFrame:
    """Tek chunk'ı işler: temizle → özellik üret → etiketle."""

    # ── Tutar ─────────────────────────────────────────────────────────────────
    if "amount" in chunk.columns:
        chunk["amount"] = _clean_currency(chunk["amount"])
    elif "abs_amount" in chunk.columns:
        chunk["amount"] = pd.to_numeric(chunk["abs_amount"], errors="coerce").fillna(0)
    else:
        chunk["amount"] = 0.0

    chunk["errors"] = chunk.get("errors", pd.Series(["No Error"] * len(chunk))).fillna("No Error")

    # ── Zaman ─────────────────────────────────────────────────────────────────
    chunk["tarih"] = pd.to_datetime(chunk.get("date", pd.NaT), errors="coerce")
    chunk["saat"]  = chunk["tarih"].dt.hour.fillna(12).astype(int)
    chunk["gun"]   = chunk["tarih"].dt.dayofweek.fillna(0).astype(int)
    chunk["ay"]    = chunk["tarih"].dt.month.fillna(1).astype(int)

    # ── Temel bayraklar ───────────────────────────────────────────────────────
    chunk["gece_islemi"]  = ((chunk["saat"] >= 22) | (chunk["saat"] <= 6)).astype(int)
    chunk["hafta_sonu"]   = (chunk["gun"] >= 5).astype(int)
    chunk["hata_var"]     = (chunk["errors"] != "No Error").astype(int)
    chunk["online_islem"] = (chunk.get("use_chip", pd.Series([""] * len(chunk))) == "Online Transaction").astype(int)
    chunk["buyuk_islem"]  = (chunk["amount"] > 1000).astype(int)
    chunk["negatif"]      = (chunk["amount"] < 0).astype(int)
    chunk["abs_amount"]   = chunk["amount"].abs()

    # ── Z-score anomali bayrağı ───────────────────────────────────────────────
    mu  = chunk["amount"].mean()
    std = chunk["amount"].std() + 1e-9
    chunk["zscore_flag"] = (((chunk["amount"] - mu) / std).abs() > 3).astype(int)

    # ── Etiket ────────────────────────────────────────────────────────────────
    chunk["fraud_label"] = (
        chunk["id"].isin(fraud_ids).astype(int)
        if "id" in chunk.columns else 0
    )

    keep = [
        "client_id", "amount", "abs_amount", "saat", "gun", "ay",
        "gece_islemi", "hafta_sonu", "hata_var", "online_islem",
        "buyuk_islem", "negatif", "zscore_flag", "fraud_label", "tarih",
    ]
    return chunk[[c for c in keep if c in chunk.columns]]


def load_transactions_with_labels() -> pd.DataFrame:
    _section("1. İşlem verisi yükleniyor")

    fraud_ids    = load_fraud_ids()
    parquet_path = DATA_PATH / "transactions_clean.parquet"
    csv_path     = DATA_PATH / "transactions_data.csv"
    chunks: list[pd.DataFrame] = []

    with _timer("veri yükleme"):
        if parquet_path.exists():
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(batch_size=MAX_TRAIN_ROWS):
                chunks.append(_process_tx_chunk(batch.to_pandas(), fraud_ids))
            log.info("  Kaynak: parquet")
        elif csv_path.exists():
            for chunk in pd.read_csv(csv_path, chunksize=MAX_TRAIN_ROWS, low_memory=False):
                chunks.append(_process_tx_chunk(chunk, fraud_ids))
            log.info("  Kaynak: CSV")
        else:
            raise FileNotFoundError(
                f"İşlem verisi bulunamadı.\n"
                f"  Aranan: {parquet_path}\n"
                f"  Aranan: {csv_path}"
            )

    df = pd.concat(chunks, ignore_index=True)

    log.info("  Toplam işlem   : %d", len(df))
    log.info("  Fraud işlem    : %d  (%.3f%%)", df["fraud_label"].sum(), df["fraud_label"].mean() * 100)
    log.info("  Unique müşteri : %d", df["client_id"].nunique())
    log.info("  Bellek         : %.1f MB", _mem_mb(df))
    log.info("  Parmak izi     : %s", _df_fingerprint(df))

    if df["fraud_label"].sum() == 0:
        raise AssertionError(
            "Hiç fraud etiketi yok! "
            "train_fraud_labels.json formatını ve işlem id'lerini kontrol et."
        )

    return df


def add_client_context(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Her işleme müşteri bazlı istatistikler ekler (velocity + RFM temeli)."""
    log.info("  Müşteri bağlamı hesaplanıyor...")

    ctx = (
        df_tx.groupby("client_id")
             .agg(
                 musteri_ort_tutar    =("amount",      "mean"),
                 musteri_std_tutar    =("amount",      "std"),
                 musteri_islem_sayisi =("amount",      "count"),
                 musteri_gece_oran    =("gece_islemi", "mean"),
                 musteri_hata_oran    =("hata_var",    "mean"),
                 musteri_zscore_oran  =("zscore_flag", "mean"),
             )
             .reset_index()
    )
    ctx["musteri_std_tutar"] = ctx["musteri_std_tutar"].fillna(0)

    df_tx = df_tx.merge(ctx, on="client_id", how="left")
    df_tx["tutar_sapma"] = (
        (df_tx["amount"] - df_tx["musteri_ort_tutar"])
        / (df_tx["musteri_std_tutar"] + 1)
    ).clip(-10, 10)

    # Velocity: son işlem ile önceki arasındaki süre (dakika)
    df_tx = df_tx.sort_values(["client_id", "tarih"])
    df_tx["velocity_dk"] = (
        df_tx.groupby("client_id")["tarih"]
             .diff()
             .dt.total_seconds()
             .div(60)
             .fillna(9999)
             .clip(0, 9999)
    )

    return df_tx.fillna(0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════════════════════

TX_FEATURE_COLS = [
    "amount", "abs_amount", "saat", "gun", "ay",
    "gece_islemi", "hafta_sonu", "hata_var", "online_islem",
    "buyuk_islem", "negatif", "zscore_flag", "velocity_dk",
    "musteri_ort_tutar", "musteri_std_tutar", "musteri_islem_sayisi",
    "musteri_gece_oran", "musteri_hata_oran", "musteri_zscore_oran", "tutar_sapma",
]


def _cross_validate(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """
    Stratified K-Fold CV — tek train/test split'e göre çok daha güvenilir.
    Döner: mean/std AUC, F1, Precision, Recall
    """
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs, f1s, precs, recs, aps = [], [], [], [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)

        aucs.append(roc_auc_score(yte, prob))
        f1s.append(f1_score(yte, pred, zero_division=0))
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
        aps.append(average_precision_score(yte, prob))

        log.info(
            "    Fold %d/%d — AUC=%.4f  F1=%.4f  AP=%.4f",
            fold, CV_FOLDS, aucs[-1], f1s[-1], aps[-1],
        )

    result = {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std":  round(float(np.std(aucs)),  4),
        "f1_mean":  round(float(np.mean(f1s)),  4),
        "ap_mean":  round(float(np.mean(aps)),  4),
        "prec_mean":round(float(np.mean(precs)), 4),
        "rec_mean": round(float(np.mean(recs)),  4),
    }
    log.info(
        "  %s CV özeti — AUC %.4f ± %.4f  F1 %.4f  AP %.4f",
        label, result["auc_mean"], result["auc_std"], result["f1_mean"], result["ap_mean"],
    )
    return result


def _build_models(scale: float) -> dict:
    """Eğitilecek model konfigürasyonlarını döner."""
    models: dict[str, object] = {}

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="aucpr",   # PR-AUC — dengesiz veri için daha anlamlı
            verbosity=0,
            early_stopping_rounds=30,
        )

    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return models


def train_and_compare_models(
    df_tx: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, list[str], str]:
    _section("2. Model eğitimi")

    available = []
    if XGB_AVAILABLE:  available.append("XGBoost")
    if LGB_AVAILABLE:  available.append("LightGBM")
    available.append("RandomForest")
    log.info("  Eğitilecek modeller: %s", ", ".join(available))

    fcols = [c for c in TX_FEATURE_COLS if c in df_tx.columns]
    X     = df_tx[fcols].fillna(0)
    y     = df_tx["fraud_label"]

    log.info("  Özellik sayısı : %d", len(fcols))
    log.info("  İşlem sayısı   : %d", len(X))
    log.info("  Fraud oranı    : %.4f%%", y.mean() * 100)

    # Örnekleme
    if len(X) > MAX_TRAIN_ROWS:
        log.info("  %dK sınırına örnekleniyor...", MAX_TRAIN_ROWS // 1000)
        idx  = X.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y

    scale = (y_s == 0).sum() / max((y_s == 1).sum(), 1)
    log.info("  Sınıf ağırlığı (pos_weight): %.1f", scale)

    # SMOTE — dengesizlik çok fazlaysa
    if SMOTE_AVAILABLE and scale > 20:
        log.info("  SMOTE uygulanıyor (scale=%.0f)...", scale)
        try:
            sm   = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, int((y_s == 1).sum()) - 1))
            X_s, y_s = sm.fit_resample(X_s, y_s)
            scale    = 1.0
            log.info("  SMOTE sonrası: %d örnek, fraud oranı %.2f%%",
                     len(X_s), y_s.mean() * 100)
        except Exception as exc:
            log.warning("  SMOTE başarısız: %s — devam ediliyor.", exc)

    Xtr, Xte, ytr, yte = train_test_split(
        X_s, y_s, test_size=0.2, random_state=RANDOM_SEED, stratify=y_s
    )

    modeller  = _build_models(scale)
    sonuclar: dict[str, dict] = {}
    cv_sonuclar: dict[str, dict] = {}

    for isim, model in modeller.items():
        log.info("  [%s] eğitiliyor...", isim)
        with _timer(f"{isim} eğitim"):
            # XGBoost early stopping için eval_set gerekir
            if isim == "XGBoost" and XGB_AVAILABLE:
                model.fit(
                    Xtr, ytr,
                    eval_set=[(Xte, yte)],
                    verbose=False,
                )
            else:
                model.fit(Xtr, ytr)

        prob = model.predict_proba(Xte)[:, 1]
        pred = model.predict(Xte)
        sonuclar[isim] = {
            "model": model,
            "auc":   roc_auc_score(yte, prob),
            "f1":    f1_score(yte, pred, zero_division=0),
            "prec":  precision_score(yte, pred, zero_division=0),
            "rec":   recall_score(yte, pred, zero_division=0),
            "ap":    average_precision_score(yte, prob),
        }
        log.info(
            "  %-12s  AUC=%.4f  F1=%.4f  AP=%.4f  Prec=%.4f  Rec=%.4f",
            isim,
            sonuclar[isim]["auc"],
            sonuclar[isim]["f1"],
            sonuclar[isim]["ap"],
            sonuclar[isim]["prec"],
            sonuclar[isim]["rec"],
        )

        # Sadece en umut vaat eden modele CV uygula (zaman tasarrufu)
        if isim != "RandomForest":
            log.info("  [%s] %d-Fold CV başlıyor...", isim, CV_FOLDS)
            cv_model = type(model)(**{
                k: v for k, v in model.get_params().items()
                if k != "early_stopping_rounds"
            })
            cv_sonuclar[isim] = _cross_validate(cv_model, X_s, y_s, isim)

    # ── Kazanan seçimi ────────────────────────────────────────────────────────
    log.info("")
    log.info("  %-12s %8s %8s %8s %10s %8s", "Model", "AUC", "AP", "F1", "Precision", "Recall")
    log.info("  %s", "─" * 62)
    for isim, s in sonuclar.items():
        log.info("  %-12s %8.4f %8.4f %8.4f %10.4f %8.4f",
                 isim, s["auc"], s["ap"], s["f1"], s["prec"], s["rec"])

    # AP (Average Precision) fraud için AUC'dan daha anlamlı ölçüt
    en_iyi_isim = max(sonuclar, key=lambda k: sonuclar[k]["ap"])
    en_iyi      = sonuclar[en_iyi_isim]
    log.info("  Kazanan: %s  (AP=%.4f  AUC=%.4f)", en_iyi_isim, en_iyi["ap"], en_iyi["auc"])

    # ── Tüm veri için skor ────────────────────────────────────────────────────
    log.info("  %d işlem için skor üretiliyor...", len(X))
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), BATCH_SIZE):
        probs[i: i + BATCH_SIZE] = (
            en_iyi["model"]
            .predict_proba(X.iloc[i: i + BATCH_SIZE])[:, 1]
            .astype(np.float32)
        )
    df_tx = df_tx.copy()
    df_tx["fraud_prob_tx"] = probs

    _log_series("fraud_prob_tx", pd.Series(probs))
    if probs.max() < 0.05:
        log.warning(
            "  fraud_prob_tx max=%.4f çok düşük — etiket kalitesini kontrol et.",
            probs.max(),
        )

    # ── Feature importance ────────────────────────────────────────────────────
    imp_data: dict[str, list] = {}
    for isim, s in sonuclar.items():
        m = s["model"]
        if hasattr(m, "feature_importances_"):
            imp = (
                pd.DataFrame({"feature": fcols, "importance": m.feature_importances_})
                  .sort_values("importance", ascending=False)
            )
            imp_data[isim] = imp.head(10).to_dict("records")
            log.info("  %s — Top 5:", isim)
            for _, r in imp.head(5).iterrows():
                log.info("    %-32s %.4f", r["feature"], r["importance"])

    # ── SHAP global özeti ─────────────────────────────────────────────────────
    shap_summary: dict = {}
    if SHAP_AVAILABLE and en_iyi_isim in ("XGBoost", "LightGBM"):
        try:
            log.info("  SHAP global özeti hesaplanıyor...")
            sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
            X_shap     = X.iloc[sample_idx]
            explainer  = shap.TreeExplainer(en_iyi["model"])
            sv         = explainer.shap_values(X_shap)
            mean_abs   = np.abs(sv).mean(axis=0)
            shap_imp   = (
                pd.DataFrame({"feature": fcols, "shap": mean_abs})
                  .sort_values("shap", ascending=False)
            )
            shap_summary = shap_imp.head(10).to_dict("records")
            log.info("  SHAP Top 3: %s",
                     ", ".join(f"{r['feature']}({r['shap']:.3f})" for r in shap_summary[:3]))
        except Exception as exc:
            log.warning("  SHAP global hesaplanamadı: %s", exc)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    joblib.dump(en_iyi["model"], MODEL_PATH / "best_model.pkl")
    joblib.dump(en_iyi["model"], MODEL_PATH / "xgboost_fraud.pkl")   # geriye uyumluluk
    for isim, s in sonuclar.items():
        fname = isim.lower().replace(" ", "_") + "_fraud.pkl"
        joblib.dump(s["model"], MODEL_PATH / fname)
    joblib.dump(fcols, MODEL_PATH / "feature_cols.pkl")

    metrics = {
        "en_iyi_model":      en_iyi_isim,
        "pipeline_version":  "7.0",
        "git_hash":          _git_hash(),
        "egitim_tarihi":     datetime.now().isoformat(),
        "toplam_ornek":      int(len(X_s)),
        "fraud_orani":       round(float(y.mean()), 6),
        "pos_weight":        round(float(scale), 2),
        "smote_kullanildi":  SMOTE_AVAILABLE and scale > 20,
        "karsilastirma": {
            k: {
                "auc":       round(v["auc"],  4),
                "ap":        round(v["ap"],   4),
                "f1":        round(v["f1"],   4),
                "precision": round(v["prec"], 4),
                "recall":    round(v["rec"],  4),
            }
            for k, v in sonuclar.items()
        },
        "cross_validation":  cv_sonuclar,
        "auc_roc":           round(en_iyi["auc"],  4),
        "average_precision": round(en_iyi["ap"],   4),
        "f1_skoru":          round(en_iyi["f1"],   4),
        "precision":         round(en_iyi["prec"], 4),
        "recall":            round(en_iyi["rec"],  4),
        "feature_importance": imp_data,
        "shap_global":       shap_summary,
    }
    with open(MODEL_PATH / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    log.info("  Modeller ve metrikler kaydedildi.")
    return df_tx, metrics, fcols, en_iyi_isim


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÜŞTERİ BAZINA TOPLAMA
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_client(df_tx: pd.DataFrame) -> pd.DataFrame:
    _section("3. Müşteri bazına toplama")

    needed = [
        "client_id", "fraud_prob_tx", "amount", "gece_islemi",
        "hata_var", "online_islem", "hafta_sonu", "buyuk_islem",
        "negatif", "zscore_flag", "velocity_dk", "tarih",
    ]
    df_s = df_tx[[c for c in needed if c in df_tx.columns]]

    with _timer("groupby"):
        client = (
            df_s.groupby("client_id")
                .agg(
                    fraud_skoru_xgb    =("fraud_prob_tx", lambda x: x.mean() * 100),
                    fraud_max_tx       =("fraud_prob_tx", "max"),
                    fraud_tx_sayisi    =("fraud_prob_tx", lambda x: (x > 0.5).sum()),
                    tx_islem_sayisi    =("amount",        "count"),
                    tx_toplam_tutar    =("amount",        "sum"),
                    tx_ortalama_tutar  =("amount",        "mean"),
                    tx_std_tutar       =("amount",        "std"),
                    tx_max_tutar       =("amount",        "max"),
                    tx_gece_oran       =("gece_islemi",   "mean"),
                    tx_hata_oran       =("hata_var",      "mean"),
                    tx_online_oran     =("online_islem",  "mean"),
                    tx_hafta_sonu_oran =("hafta_sonu",    "mean"),
                    tx_buyuk_oran      =("buyuk_islem",   "mean"),
                    tx_negatif_oran    =("negatif",       "mean"),
                    tx_zscore_oran     =("zscore_flag",   "mean"),
                    tx_velocity_ort    =("velocity_dk",   "mean"),
                    son_islem_tarihi   =("tarih",         "max"),
                )
                .reset_index()
        )

    client = client.fillna(0)

    # RFM skoru
    if "son_islem_tarihi" in client.columns:
        bugun = pd.Timestamp.now()
        client["son_islem_tarihi"] = pd.to_datetime(client["son_islem_tarihi"], errors="coerce")
        client["rfm_recency"]  = (bugun - client["son_islem_tarihi"]).dt.days.fillna(9999).clip(0, 9999)
        client["rfm_frequency"]= client["tx_islem_sayisi"]
        client["rfm_monetary"] = client["tx_toplam_tutar"]

    client = _add_time_window_features(df_tx, client)

    log.info("  Müşteri sayısı : %d", len(client))
    log.info("  Sütun sayısı   : %d", len(client.columns))
    return client


def _add_time_window_features(df_tx: pd.DataFrame, df_client: pd.DataFrame) -> pd.DataFrame:
    """7 / 30 / 90 günlük pencere özellikleri."""
    if "tarih" not in df_tx.columns:
        log.warning("  'tarih' yok — zaman pencereleri atlandı.")
        return df_client

    df_tx   = df_tx.copy()
    df_tx["tarih"] = pd.to_datetime(df_tx["tarih"], errors="coerce")
    df_tx   = df_tx.dropna(subset=["tarih"])
    bugun   = df_tx["tarih"].max()

    def _window(days: int, prefix: str) -> pd.DataFrame:
        sub = df_tx[df_tx["tarih"] >= bugun - pd.Timedelta(days=days)]
        if sub.empty:
            log.warning("  %s penceresi boş.", prefix)
            return pd.DataFrame(columns=["client_id"])
        return (
            sub.groupby("client_id")
               .agg(**{
                   f"{prefix}_islem_sayisi": ("amount",      "count"),
                   f"{prefix}_toplam_tutar": ("amount",      "sum"),
                   f"{prefix}_gece_oran":    ("gece_islemi", "mean"),
                   f"{prefix}_hata_oran":    ("hata_var",    "mean"),
               })
               .reset_index()
        )

    w7  = _window(7,  "son7")
    w30 = _window(30, "son30")
    w90 = _window(90, "son90")

    for w, cols in [
        (w7,  ["client_id", "son7_islem_sayisi",  "son7_toplam_tutar"]),
        (w30, None),
        (w90, ["client_id", "son90_islem_sayisi", "son90_toplam_tutar",
                "son90_gece_oran", "son90_hata_oran"]),
    ]:
        merge_cols = [c for c in (cols or w.columns.tolist()) if c in w.columns]
        if len(merge_cols) > 1:
            df_client = df_client.merge(w[merge_cols], on="client_id", how="left")

    df_client = df_client.fillna(0)

    eps = 1e-6
    df_client["tutar_degisim_7_30"] = (
        df_client.get("son7_toplam_tutar", 0)
        / (df_client.get("son30_toplam_tutar", 0) / 4.3 + eps)
    ).clip(0, 10)

    df_client["islem_degisim_7_30"] = (
        df_client.get("son7_islem_sayisi", 0)
        / (df_client.get("son30_islem_sayisi", 0) / 4.3 + eps)
    ).clip(0, 10)

    df_client["gece_ani_artis"] = (
        df_client.get("son7_gece_oran", pd.Series(0, index=df_client.index))
        - df_client.get("son90_gece_oran", pd.Series(0, index=df_client.index))
    ).clip(-1, 1)

    df_client["hata_ani_artis"] = (
        df_client.get("son7_hata_oran", pd.Series(0, index=df_client.index))
        - df_client.get("son90_hata_oran", pd.Series(0, index=df_client.index))
    ).clip(-1, 1)

    log.info("  Zaman penceresi özellikleri eklendi.")
    return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 4. KART & KULLANICI VERİSİ
# ══════════════════════════════════════════════════════════════════════════════

def build_user_features() -> pd.DataFrame:
    path = DATA_PATH / "users_data.csv"
    if not path.exists():
        log.warning("  users_data.csv bulunamadı.")
        return pd.DataFrame(columns=["client_id"])

    df = pd.read_csv(path, low_memory=False)
    for col in ["per_capita_income", "yearly_income", "total_debt"]:
        if col in df.columns:
            df[col] = _clean_currency(df[col])

    df = df.rename(columns={"id": "client_id"})
    keep = ["client_id", "current_age", "credit_score",
            "per_capita_income", "yearly_income", "total_debt"]
    feat = df[[c for c in keep if c in df.columns]].copy()

    if {"total_debt", "yearly_income"}.issubset(feat.columns):
        feat["borc_gelir_orani"] = (
            feat["total_debt"] / (feat["yearly_income"] + 1)
        ).clip(0, 50)

    if "credit_score" in feat.columns:
        feat["dusuk_kredi"] = (feat["credit_score"] < 600).astype(int)

    return feat.fillna(0)


def build_card_features() -> pd.DataFrame:
    path = DATA_PATH / "cards_data.csv"
    if not path.exists():
        log.warning("  cards_data.csv bulunamadı.")
        return pd.DataFrame(columns=["client_id"])

    df = pd.read_csv(path, low_memory=False)
    df["credit_limit"] = _clean_currency(df.get("credit_limit", pd.Series(dtype=str)))

    if "card_on_dark_web" in df.columns:
        df["dark_web_flag"] = (
            df["card_on_dark_web"].astype(str).str.strip().str.lower() == "yes"
        ).astype(int)
    else:
        log.warning("  'card_on_dark_web' sütunu yok — 0 atanıyor.")
        df["dark_web_flag"] = 0

    feat = (
        df.groupby("client_id")
          .agg(
              kart_adedi    =("id",           "count"),
              toplam_limit  =("credit_limit", "sum"),
              ort_limit     =("credit_limit", "mean"),
              dark_web_oran =("dark_web_flag","mean"),
              dark_web_kart =("dark_web_flag","max"),
              chip_oran     =("has_chip",     lambda x: (x == "YES").mean()
                              if "has_chip" in df.columns else 0),
          )
          .reset_index()
    )
    log.info(
        "  dark_web_kart=1: %d müşteri (toplam %d)",
        (feat["dark_web_kart"] == 1).sum(), len(feat),
    )
    return feat.fillna(0)


def enrich_with_db_features(df_client: pd.DataFrame) -> pd.DataFrame:
    _section("4. DB zenginleştirme")

    if not DB_PATH.exists():
        log.warning("  financeai.db yok — DB adımı atlandı.")
        return df_client

    conn    = sqlite3.connect(DB_PATH)
    eklenen = 0

    TABLOLAR = {
        "client_ozet": [
            "client_id", "avg_transaction", "islem_basina_hata",
            "gece_oran", "online_oran", "risk_skoru",
            "kart_adedi", "toplam_limit", "ort_limit",
        ],
        "client_risk": ["client_id", "risk_skoru"],
    }

    for tablo, cols in TABLOLAR.items():
        if "risk_skoru" in df_client.columns and tablo == "client_risk":
            continue
        try:
            mevcut_cols = pd.read_sql(
                f"SELECT name FROM pragma_table_info('{tablo}')", conn
            )["name"].tolist()
            secilecek = [c for c in cols if c in mevcut_cols]
            if not secilecek or "client_id" not in secilecek:
                log.warning("  %s: uygun sütun bulunamadı — atlandı.", tablo)
                continue
            df_ek     = pd.read_sql(f"SELECT {', '.join(secilecek)} FROM {tablo}", conn)
            onceki    = len(df_client.columns)
            df_client = df_client.merge(df_ek, on="client_id", how="left")
            eklenen  += len(df_client.columns) - onceki
            log.info("  %s → %d özellik eklendi.", tablo, len(df_client.columns) - onceki - 0)
        except Exception as exc:
            log.warning("  %s okunamadı: %s", tablo, exc)

    conn.close()
    df_client = df_client.fillna(0)
    log.info("  Toplam eklenen özellik: %d", eklenen)
    return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 5. ISOLATION FOREST
# ══════════════════════════════════════════════════════════════════════════════

CLIENT_FEATURE_COLS = [
    "fraud_skoru_xgb", "fraud_max_tx", "fraud_tx_sayisi",
    "tx_islem_sayisi", "tx_toplam_tutar", "tx_ortalama_tutar",
    "tx_std_tutar", "tx_max_tutar", "tx_gece_oran", "tx_hata_oran",
    "tx_online_oran", "tx_hafta_sonu_oran", "tx_buyuk_oran", "tx_negatif_oran",
    "tx_zscore_oran", "tx_velocity_ort",
    "dark_web_kart", "kart_adedi", "toplam_limit", "dark_web_oran",
    "son30_islem_sayisi", "son30_toplam_tutar", "son30_gece_oran", "son30_hata_oran",
    "son7_islem_sayisi",  "son7_toplam_tutar",
    "tutar_degisim_7_30", "islem_degisim_7_30",
    "gece_ani_artis",     "hata_ani_artis",
    "rfm_recency",        "rfm_frequency",      "rfm_monetary",
    "risk_skoru",
    "avg_transaction", "islem_basina_hata", "gece_oran", "online_oran",
    "ort_limit", "borc_gelir_orani", "credit_score", "dusuk_kredi",
]


def train_isolation_forest(df_client: pd.DataFrame) -> pd.DataFrame:
    _section("5. Isolation Forest")

    fcols  = [c for c in CLIENT_FEATURE_COLS if c in df_client.columns]
    X      = df_client[fcols].fillna(0)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        max_features=min(0.8, len(fcols) / len(fcols)),  # tüm özellikler kullanılsın
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(Xs)

    df_client["anomali_skoru"] = iso.decision_function(Xs)
    df_client["iso_tahmin"]    = (iso.predict(Xs) == -1).astype(int)

    joblib.dump(iso,    MODEL_PATH / "isolation_forest.pkl")
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")
    joblib.dump(fcols,  MODEL_PATH / "iso_feature_cols.pkl")

    n_anomali = int(df_client["iso_tahmin"].sum())
    log.info("  Anomali: %d  (%.1f%%)", n_anomali, n_anomali / len(df_client) * 100)
    return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHURN
# ══════════════════════════════════════════════════════════════════════════════

def build_churn_labels(df_client: pd.DataFrame) -> pd.DataFrame:
    _section("6. Churn etiketleri")

    if "son_islem_tarihi" not in df_client.columns:
        log.warning("  son_islem_tarihi yok — churn_label=0.")
        df_client["churn_label"] = 0
        df_client["churn_riski"] = 0.5
        return df_client

    bugun = pd.Timestamp.now()
    df_client["son_islem_tarihi"] = pd.to_datetime(df_client["son_islem_tarihi"], errors="coerce")
    df_client["gun_fark"] = (
        (bugun - df_client["son_islem_tarihi"]).dt.days.fillna(9999)
    ).clip(0, 9999)

    df_client["churn_label"] = (df_client["gun_fark"] >= CHURN_INACTIVE_DAYS).astype(int)
    df_client["churn_riski"] = (df_client["gun_fark"].clip(0, 365) / 365).round(4)

    n0 = int((df_client["churn_label"] == 0).sum())
    n1 = int((df_client["churn_label"] == 1).sum())
    log.info("  Eşik: %d gün | Aktif=%d  Churn=%d  (%.1f%%)",
             CHURN_INACTIVE_DAYS, n0, n1, n1 / max(n0 + n1, 1) * 100)
    return df_client


def train_churn_model(df_client: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    _section("6b. Churn modeli")

    fcols = [c for c in CLIENT_FEATURE_COLS if c in df_client.columns]
    X     = df_client[fcols].fillna(0)
    y     = df_client["churn_label"]

    if y.nunique() < 2:
        log.warning("  Tek sınıf — churn modeli eğitilemiyor.")
        df_client["churn_skoru"]   = 0.0
        df_client["churn_tahmini"] = "Dusuk Risk"
        return df_client, {"churn_auc": 0.0, "churn_ap": 0.0}

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scale = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale, random_state=RANDOM_SEED,
            n_jobs=-1, verbosity=0,
        )
    elif LGB_AVAILABLE:
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale, random_state=RANDOM_SEED,
            n_jobs=-1, verbose=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )

    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    auc  = roc_auc_score(yte, prob)
    ap   = average_precision_score(yte, prob)

    df_client["churn_skoru"] = (model.predict_proba(X)[:, 1] * 100).round(2)
    df_client["churn_tahmini"] = pd.cut(
        df_client["churn_skoru"],
        bins=[-np.inf, 30, 60, np.inf],
        labels=["Dusuk Risk", "Orta Risk", "Yuksek Risk"],
    ).astype(str)

    joblib.dump(model, MODEL_PATH / "churn_model.pkl")
    log.info("  Churn AUC=%.4f  AP=%.4f", auc, ap)
    return df_client, {"churn_auc": round(auc, 4), "churn_ap": round(ap, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP AÇIKLAMALARI
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_explanations(df_client: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """
    Her müşteri için top-3 SHAP nedeni hesaplar.
    Yeni sütunlar: shap_reason_1/2/3, shap_val_1/2/3
    """
    if not SHAP_AVAILABLE:
        log.warning("  SHAP kurulu değil — pip install shap")
        return df_client

    model_path = MODEL_PATH / "best_model.pkl"
    if not model_path.exists():
        log.warning("  best_model.pkl bulunamadı — SHAP atlandı.")
        return df_client

    try:
        model = joblib.load(model_path)
        X     = df_client[[c for c in fcols if c in df_client.columns]].fillna(0)

        log.info("  SHAP açıklamaları hesaplanıyor (%d müşteri)...", len(X))
        explainer = shap.TreeExplainer(model)

        r1_list, r2_list, r3_list = [], [], []
        v1_list, v2_list, v3_list = [], [], []

        for start in range(0, len(X), BATCH_SIZE):
            batch      = X.iloc[start: start + BATCH_SIZE]
            sv         = explainer.shap_values(batch)
            cols_list  = X.columns.tolist()

            for row in sv:
                idx = np.argsort(np.abs(row))[::-1]
                r1_list.append(cols_list[idx[0]] if len(idx) > 0 else "")
                r2_list.append(cols_list[idx[1]] if len(idx) > 1 else "")
                r3_list.append(cols_list[idx[2]] if len(idx) > 2 else "")
                v1_list.append(round(float(row[idx[0]]), 4) if len(idx) > 0 else 0.0)
                v2_list.append(round(float(row[idx[1]]), 4) if len(idx) > 1 else 0.0)
                v3_list.append(round(float(row[idx[2]]), 4) if len(idx) > 2 else 0.0)

            log.info("  SHAP: %d/%d", min(start + BATCH_SIZE, len(X)), len(X))

        df_client = df_client.copy()
        df_client["shap_reason_1"] = r1_list
        df_client["shap_reason_2"] = r2_list
        df_client["shap_reason_3"] = r3_list
        df_client["shap_val_1"]    = v1_list
        df_client["shap_val_2"]    = v2_list
        df_client["shap_val_3"]    = v3_list

        log.info("  SHAP tamamlandı.")

    except Exception as exc:
        log.error("  SHAP hatası: %s", exc)

    return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 8. FİNAL FRAUD SKORU
# ══════════════════════════════════════════════════════════════════════════════

def compute_final_fraud_score(df_client: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost/LGB skoru + Isolation Forest + kural tabanlı skor
    normalize ağırlıklarla birleştirilir.

    Ağırlık mantığı:
      - ML modeli varyansı yüksekse → daha güvenilir → daha fazla ağırlık
      - Isolation Forest aralığı geniş → daha güvenilir
      - Kural tabanlı kısım her zaman en az %10 ağırlık alır
      - Toplam = 1 (normalize)
    """
    _section("7. Final fraud skoru")

    df_client = df_client.copy()
    s = np.zeros(len(df_client))

    raw_w: dict[str, float] = {}

    if "fraud_skoru_xgb" in df_client.columns:
        var = df_client["fraud_skoru_xgb"].var()
        raw_w["ml"] = 0.60 if var > 10 else (0.50 if var > 3 else 0.40)
    else:
        raw_w["ml"] = 0.0

    iso_n = pd.Series(np.zeros(len(df_client)), index=df_client.index)
    if "anomali_skoru" in df_client.columns:
        rng = df_client["anomali_skoru"].max() - df_client["anomali_skoru"].min()
        if rng > 1e-6:
            iso_n     = ((-df_client["anomali_skoru"] - df_client["anomali_skoru"].min()) / rng * 100)
            raw_w["iso"] = 0.25 if rng > 0.5 else 0.15
        else:
            raw_w["iso"] = 0.0
    else:
        raw_w["iso"] = 0.0

    raw_w["kural"] = max(0.10, 1.0 - raw_w["ml"] - raw_w["iso"])

    total = sum(raw_w.values())
    w     = {k: v / total for k, v in raw_w.items()}

    log.info("  Ağırlıklar — ML:%.0f%%  ISO:%.0f%%  Kural:%.0f%%",
             w["ml"] * 100, w["iso"] * 100, w["kural"] * 100)

    if w["ml"] > 0 and "fraud_skoru_xgb" in df_client.columns:
        s += df_client["fraud_skoru_xgb"].values * w["ml"]
    if w["iso"] > 0:
        s += iso_n.values * w["iso"]

    def _safe_get(col: str) -> pd.Series:
        return df_client.get(col, pd.Series(0, index=df_client.index))

    kural_s  = np.zeros(len(df_client))
    kural_s += _safe_get("dark_web_oran").values    * 0.45
    kural_s += _safe_get("tx_hata_oran").values     * 0.20
    kural_s += _safe_get("islem_basina_hata").clip(0, 1).values * 0.15
    kural_s += _safe_get("hata_ani_artis").clip(0, 1).values    * 0.10
    kural_s += _safe_get("tx_zscore_oran").values   * 0.10
    s += kural_s * (w["kural"] * 100)

    df_client["fraud_skoru"] = np.clip(s, 0, 100).round(2)

    q75 = float(df_client["fraud_skoru"].quantile(0.75))
    q90 = float(df_client["fraud_skoru"].quantile(0.90))

    log.info("  Eşikler — Şüpheli:%.1f  Yüksek:%.1f", q75, q90)
    log.info("  Skor — mean:%.2f  std:%.2f  max:%.2f",
             df_client["fraud_skoru"].mean(),
             df_client["fraud_skoru"].std(),
             df_client["fraud_skoru"].max())

    df_client["fraud_tahmini"] = pd.cut(
        df_client["fraud_skoru"],
        bins=[-np.inf, q75, q90, np.inf],
        labels=["Normal", "Supheli", "Yuksek Risk"],
    ).astype(str)

    return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 9. KAYIT
# ══════════════════════════════════════════════════════════════════════════════

SAVE_COLS = [
    "client_id", "fraud_skoru", "fraud_tahmini",
    "fraud_skoru_xgb", "anomali_skoru", "iso_tahmin",
    "churn_skoru", "churn_tahmini",
    "tx_gece_oran", "tx_hata_oran", "tx_zscore_oran", "tx_velocity_ort",
    "dark_web_oran", "dark_web_kart",
    "tx_islem_sayisi", "borc_gelir_orani", "dusuk_kredi",
    "fraud_max_tx", "fraud_tx_sayisi", "risk_skoru",
    "avg_transaction", "islem_basina_hata",
    "son30_islem_sayisi", "son30_toplam_tutar",
    "son7_islem_sayisi",  "son7_toplam_tutar",
    "tutar_degisim_7_30", "islem_degisim_7_30",
    "gece_ani_artis",     "hata_ani_artis",
    "rfm_recency",        "rfm_frequency",      "rfm_monetary",
    "shap_reason_1",      "shap_reason_2",       "shap_reason_3",
    "shap_val_1",         "shap_val_2",           "shap_val_3",
]


def save_results(df_client: pd.DataFrame) -> dict:
    _section("9. Kayıt")

    save_cols = [c for c in SAVE_COLS if c in df_client.columns]
    conn      = sqlite3.connect(DB_PATH)

    df_client[save_cols].to_sql("client_ml", conn, if_exists="replace", index=False)
    log.info("  client_ml: %d kayıt, %d sütun", len(df_client), len(save_cols))

    ozet = {
        "toplam":           int(len(df_client)),
        "normal":           int((df_client["fraud_tahmini"] == "Normal").sum()),
        "supheli":          int((df_client["fraud_tahmini"] == "Supheli").sum()),
        "yuksek_risk":      int((df_client["fraud_tahmini"] == "Yuksek Risk").sum()),
        "churn_yuksek":     int((df_client.get("churn_tahmini", pd.Series()) == "Yuksek Risk").sum()),
        "ort_fraud_skoru":  round(float(df_client["fraud_skoru"].mean()), 2),
        "max_fraud_skoru":  round(float(df_client["fraud_skoru"].max()), 2),
        "shap_aktif":       int("shap_reason_1" in df_client.columns),
        "hesaplama_tarihi": datetime.now().isoformat(),
        "pipeline_version": "7.0",
    }
    pd.DataFrame([ozet]).to_sql("ml_ozet", conn, if_exists="replace", index=False)
    conn.close()

    log.info("  ml_ozet kaydedildi.")
    return ozet


# ══════════════════════════════════════════════════════════════════════════════
# 10. ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_model() -> tuple[pd.DataFrame, dict]:
    log.info("=" * 64)
    log.info("  FinanceAI — ML Pipeline v7.0")
    log.info("  XGBoost  : %s", XGB_AVAILABLE)
    log.info("  LightGBM : %s", LGB_AVAILABLE)
    log.info("  SHAP     : %s", SHAP_AVAILABLE)
    log.info("  SMOTE    : %s", SMOTE_AVAILABLE)
    log.info("=" * 64)
    t0 = datetime.now()

    # 1-2. Veri + model
    df_tx = load_transactions_with_labels()
    df_tx = add_client_context(df_tx)
    df_tx, metrics, fcols, en_iyi_model = train_and_compare_models(df_tx)

    # 3. Müşteri bazına topla
    df_client = aggregate_to_client(df_tx)
    del df_tx
    gc.collect()
    log.info("  İşlem verisi RAM'den temizlendi.")

    # 4. Zenginleştirme
    _section("4. Profil zenginleştirme")
    user      = build_user_features()
    card      = build_card_features()
    df_client = df_client.merge(user, on="client_id", how="left")
    df_client = df_client.merge(card, on="client_id", how="left", suffixes=("", "_card"))
    df_client = df_client.fillna(0)
    df_client = enrich_with_db_features(df_client)

    # 5. Anomali
    df_client = train_isolation_forest(df_client)

    # 6. Churn
    df_client = build_churn_labels(df_client)
    df_client, cm = train_churn_model(df_client)
    metrics.update(cm)

    # 7. SHAP (client düzeyinde)
    _section("8. SHAP açıklamaları")
    df_client = compute_shap_explanations(df_client, fcols)

    # 8. Final skor
    df_client = compute_final_fraud_score(df_client)

    # 9. Kaydet
    ozet = save_results(df_client)

    sure = int((datetime.now() - t0).total_seconds())
    log.info("")
    log.info("=" * 64)
    log.info("  SONUÇ — v7.0")
    log.info("=" * 64)
    log.info("  Kazanan Model  : %s", en_iyi_model)
    log.info("  AUC-ROC        : %.4f", metrics.get("auc_roc", 0))
    log.info("  Avg Precision  : %.4f", metrics.get("average_precision", 0))
    log.info("  Churn AUC      : %.4f", metrics.get("churn_auc", 0))
    log.info("  Toplam müşteri : %d", ozet["toplam"])
    log.info("  Normal         : %d  (%.1f%%)",
             ozet["normal"],      ozet["normal"]      / max(ozet["toplam"], 1) * 100)
    log.info("  Şüpheli        : %d  (%.1f%%)",
             ozet["supheli"],     ozet["supheli"]     / max(ozet["toplam"], 1) * 100)
    log.info("  Yüksek Risk    : %d  (%.1f%%)",
             ozet["yuksek_risk"], ozet["yuksek_risk"] / max(ozet["toplam"], 1) * 100)
    log.info("  SHAP aktif     : %s", bool(ozet.get("shap_aktif")))
    log.info("  Süre           : %ds", sure)
    log.info("=" * 64)

    return df_client, metrics


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI — sorgulama
# ══════════════════════════════════════════════════════════════════════════════

def predict_client(client_id: int) -> dict:
    """Tek müşterinin ML sonuçlarını döner."""
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        "SELECT * FROM client_ml WHERE client_id = ?", conn, params=(client_id,)
    )
    conn.close()
    return df.iloc[0].to_dict() if not df.empty else {}


def get_all_ml_results() -> pd.DataFrame:
    """Tüm müşterileri fraud skoruna göre sıralı döner."""
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM client_ml ORDER BY fraud_skoru DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_model_metrics() -> dict:
    """En son eğitimin metriklerini döner."""
    p = MODEL_PATH / "model_metrics.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    train_model()