"""
FinanceAI — Açıklanabilirlik Modülü (SHAP)
==========================================

"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger("financeai.explainability")

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "data"
DB_PATH    = DATA_PATH / "financeai.db"

# ══════════════════════════════════════════════════════════════════════════════
# SHAP YÜKLEME — opsiyonel import
# ══════════════════════════════════════════════════════════════════════════════
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("shap kurulu değil. `pip install shap` ile yükleyin.")


# ══════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL AÇIKLANABILIRLIK — tüm modelin davranışı
# ══════════════════════════════════════════════════════════════════════════════

def explain_model(
    X: pd.DataFrame,
    sample_size: int = 1000,
    save_path: Path | None = None,
) -> dict:
    """
    Modelin genel davranışını SHAP ile açıklar.

    Döner:
        {
            "mean_abs_shap": {özellik: ortalama |SHAP|},  ← en önemli özellikler
            "top10": [{"feature": ..., "importance": ...}],
            "shap_available": bool
        }

    Hocana anlatacağın:
        mean_abs_shap = her özelliğin ortalama katkısı
        Yüksek değer → model bu özelliğe çok güveniyor
    """
    if not SHAP_AVAILABLE:
        return {"shap_available": False, "error": "shap kurulu değil"}

    try:
        model_path = MODEL_PATH / "xgboost_fraud.pkl"
        fcols_path = MODEL_PATH / "feature_cols.pkl"

        if not model_path.exists():
            return {"shap_available": False, "error": "Model dosyası bulunamadı"}

        model = joblib.load(model_path)
        fcols = joblib.load(fcols_path) if fcols_path.exists() else X.columns.tolist()

        X_use = X[[c for c in fcols if c in X.columns]].fillna(0)

        # Büyük veri setlerinde örnekleme yap
        if len(X_use) > sample_size:
            X_sample = X_use.sample(sample_size, random_state=42)
        else:
            X_sample = X_use

        # TreeExplainer: XGBoost için optimize edilmiş SHAP hesaplayıcı
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # mean(|SHAP|) — her özelliğin ortalama mutlak katkısı
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = (
            pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        result = {
            "shap_available":  True,
            "sample_size":     len(X_sample),
            "mean_abs_shap":   importance_df.set_index("feature")["importance"].round(4).to_dict(),
            "top10":           importance_df.head(10).to_dict("records"),
            "expected_value":  float(explainer.expected_value)
                               if not hasattr(explainer.expected_value, '__len__')
                               else float(explainer.expected_value[1]),
        }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            log.info("  SHAP global analiz kaydedildi: %s", save_path)

        log.info("  Top 5 SHAP özelliği:")
        for _, r in importance_df.head(5).iterrows():
            log.info("    %-30s  %.4f", r["feature"], r["importance"])

        return result

    except Exception as exc:
        log.error("  SHAP global analiz hatası: %s", exc)
        return {"shap_available": False, "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCAL AÇIKLANABILIRLIK — tek müşteri için neden sorusu
# ══════════════════════════════════════════════════════════════════════════════

def explain_customer(client_id: int) -> dict:
    """
    Belirli bir müşterinin fraud skorunun nedenini açıklar.

    Döner:
        {
            "client_id": 12345,
            "fraud_score": 78.3,
            "base_value": 12.0,       ← modelin ortalama tahmini
            "top_positive": [...],    ← skoru EN ÇOK yükselten özellikler
            "top_negative": [...],    ← skoru EN ÇOK düşüren özellikler
            "human_readable": "..."   ← hocaya gösterebileceğin Türkçe açıklama
        }

    GDPR Article 22 uyumu:
        Otomatik karar verme sistemlerinde kullanıcıya açıklama hakkı.
        Bu fonksiyon o açıklamayı üretir.
    """
    if not SHAP_AVAILABLE:
        return {
            "client_id":    client_id,
            "error":        "shap kurulu değil",
            "shap_available": False,
        }

    try:
        # Müşteri verisi
        conn      = sqlite3.connect(DB_PATH)
        df_client = pd.read_sql(
            "SELECT * FROM client_ml WHERE client_id = ?", conn, params=(client_id,)
        )
        conn.close()

        if df_client.empty:
            return {"client_id": client_id, "error": "Müşteri bulunamadı"}

        model_path = MODEL_PATH / "xgboost_fraud.pkl"
        fcols_path = MODEL_PATH / "feature_cols.pkl"

        model = joblib.load(model_path)
        fcols = joblib.load(fcols_path) if fcols_path.exists() else []

        X_row = df_client[[c for c in fcols if c in df_client.columns]].fillna(0)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)

        shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_val = (
            float(explainer.expected_value)
            if not hasattr(explainer.expected_value, '__len__')
            else float(explainer.expected_value[1])
        )

        contrib_df = pd.DataFrame({
            "feature": X_row.columns,
            "shap_value": shap_row,
            "feature_value": X_row.iloc[0].values,
        }).sort_values("shap_value", ascending=False)

        top_pos = contrib_df[contrib_df["shap_value"] > 0].head(5).to_dict("records")
        top_neg = contrib_df[contrib_df["shap_value"] < 0].tail(5).to_dict("records")

        # İnsan okunabilir Türkçe açıklama
        reasons = []
        for r in top_pos[:3]:
            reasons.append(f"{r['feature']} ({r['shap_value']:+.2f})")

        fraud_score = float(df_client["fraud_skoru"].iloc[0]) if "fraud_skoru" in df_client else None
        human_text = (
            f"Müşteri #{client_id} için fraud skoru "
            f"{fraud_score:.1f}/100 olarak hesaplandı. "
            f"Bu skoru en çok yükselten faktörler: {', '.join(reasons)}."
        )

        return {
            "client_id":      client_id,
            "fraud_score":    fraud_score,
            "base_value":     round(base_val, 4),
            "top_positive":   top_pos,
            "top_negative":   top_neg,
            "human_readable": human_text,
            "shap_available": True,
        }

    except Exception as exc:
        log.error("  SHAP local analiz hatası (client=%d): %s", client_id, exc)
        return {"client_id": client_id, "error": str(exc), "shap_available": False}


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOPLU SHAP SKORU — tüm müşteriler için kaydet
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_save_shap_explanations(
    df_client: pd.DataFrame,
    batch_size: int = 500,
) -> pd.DataFrame:
    """
    Tüm müşteriler için en önemli 3 SHAP özelliğini hesaplar ve
    client_ml tablosuna ekler.

    Yeni sütunlar:
        shap_reason_1, shap_reason_2, shap_reason_3
        shap_val_1,    shap_val_2,    shap_val_3

    ml_model.py'nin train_model() fonksiyonu sonuna ekle:
        df_client = compute_and_save_shap_explanations(df_client)
    """
    if not SHAP_AVAILABLE:
        log.warning("  SHAP kurulu değil — açıklanabilirlik sütunları atlandı.")
        return df_client

    try:
        model_path = MODEL_PATH / "xgboost_fraud.pkl"
        fcols_path = MODEL_PATH / "feature_cols.pkl"

        if not model_path.exists():
            log.warning("  Model bulunamadı — SHAP atlandı.")
            return df_client

        model = joblib.load(model_path)
        fcols = joblib.load(fcols_path) if fcols_path.exists() else []

        X = df_client[[c for c in fcols if c in df_client.columns]].fillna(0)
        explainer = shap.TreeExplainer(model)

        reasons1, reasons2, reasons3 = [], [], []
        vals1,    vals2,    vals3    = [], [], []

        log.info("  SHAP açıklamaları hesaplanıyor (%d müşteri)...", len(X))

        for start in range(0, len(X), batch_size):
            batch       = X.iloc[start : start + batch_size]
            shap_batch  = explainer.shap_values(batch)

            for row_shap in shap_batch:
                idx_sorted = np.argsort(np.abs(row_shap))[::-1]
                cols       = X.columns.tolist()

                r1 = cols[idx_sorted[0]] if len(idx_sorted) > 0 else ""
                r2 = cols[idx_sorted[1]] if len(idx_sorted) > 1 else ""
                r3 = cols[idx_sorted[2]] if len(idx_sorted) > 2 else ""
                v1 = float(row_shap[idx_sorted[0]]) if len(idx_sorted) > 0 else 0.0
                v2 = float(row_shap[idx_sorted[1]]) if len(idx_sorted) > 1 else 0.0
                v3 = float(row_shap[idx_sorted[2]]) if len(idx_sorted) > 2 else 0.0

                reasons1.append(r1); reasons2.append(r2); reasons3.append(r3)
                vals1.append(round(v1, 4))
                vals2.append(round(v2, 4))
                vals3.append(round(v3, 4))

        df_client = df_client.copy()
        df_client["shap_reason_1"] = reasons1
        df_client["shap_reason_2"] = reasons2
        df_client["shap_reason_3"] = reasons3
        df_client["shap_val_1"]    = vals1
        df_client["shap_val_2"]    = vals2
        df_client["shap_val_3"]    = vals3

        log.info("  SHAP açıklamaları eklendi. Örnek:")
        log.info(
            "    İlk müşteri: %s(%+.3f), %s(%+.3f), %s(%+.3f)",
            reasons1[0], vals1[0], reasons2[0], vals2[0], reasons3[0], vals3[0],
        )
        return df_client

    except Exception as exc:
        log.error("  toplu SHAP hatası: %s", exc)
        return df_client


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHAP API ENDPOINT YARDIMCISI — FastAPI entegrasyonu
# ══════════════════════════════════════════════════════════════════════════════

def get_explanation_for_api(client_id: int) -> dict:
    """
    FastAPI endpoint'inden çağrılmak üzere tasarlandı.
    /explain/{client_id} endpointine ekle.

    Döner temiz JSON:
        {
            "client_id": 12345,
            "fraud_score": 78.3,
            "explanation": {
                "main_reason": "dark_web_kart değeri yüksek",
                "factors": [
                    {"feature": "dark_web_kart", "contribution": +32.5, "direction": "risk artırıcı"},
                    ...
                ]
            }
        }
    """
    raw = explain_customer(client_id)

    if not raw.get("shap_available"):
        return {
            "client_id":  client_id,
            "error":      raw.get("error", "Açıklama üretilemedi"),
            "fraud_score": None,
        }

    # İnsan okunabilir yön etiketi
    FEATURE_LABELS = {
        "dark_web_kart":       "Kart dark web'de görüldü",
        "gece_ani_artis":      "Gece işlemleri aniden arttı",
        "tx_hata_oran":        "İşlem hata oranı yüksek",
        "anomali_skoru":       "Anomali tespit edildi",
        "fraud_skoru_xgb":     "XGBoost fraud skoru yüksek",
        "dark_web_oran":       "Dark web kart oranı yüksek",
        "hata_ani_artis":      "Hata oranı aniden arttı",
        "fraud_max_tx":        "Tek işlemde yüksek fraud ihtimali",
        "credit_score":        "Kredi skoru düşük",
        "borc_gelir_orani":    "Borç/gelir oranı yüksek",
    }

    factors = []
    for item in (raw.get("top_positive", []) + raw.get("top_negative", []))[:5]:
        feat  = item["feature"]
        val   = item["shap_value"]
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        factors.append({
            "feature":      feat,
            "label":        label,
            "contribution": round(val, 3),
            "direction":    "risk artırıcı" if val > 0 else "risk azaltıcı",
        })

    main_reason = factors[0]["label"] if factors else "Genel risk profili"

    return {
        "client_id":  client_id,
        "fraud_score": raw.get("fraud_score"),
        "explanation": {
            "main_reason": main_reason,
            "base_score":  round(raw.get("base_value", 0) * 100, 2),
            "factors":     factors,
            "human_text":  raw.get("human_readable", ""),
        },
    }


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) > 1:
        cid = int(sys.argv[1])
        result = get_explanation_for_api(cid)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Kullanım: python explainability.py <client_id>")
        print("Örnek:    python explainability.py 12345")