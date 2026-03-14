"""
FinSight — ML Modeli v6.0
=================================================
v5.1'den farklar:
  - SHAP açıklanabilirlik (her müşteri için neden fraud?)
  - Velocity features (son 1/7/30/90 gün hız değişimi)
  - Merchant risk skoru
  - Churn: gelişmiş özellik seti + RFM analizi
  - Model kalibrasyonu (CalibratedClassifierCV)
  - Dinamik eşik optimizasyonu (F1 max)
  - Tüm açıklamalar model_metrics.json'a kaydediliyor

Çalıştır: python src/ml_model.py
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
import gc
from datetime import datetime
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score,
                              precision_score, recall_score, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Opsiyonel kütüphaneler ──────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost mevcut")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost yok, GradientBoosting kullanılacak.")

try:
    import shap
    SHAP_AVAILABLE = True
    print("✅ SHAP mevcut")
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP yok. Kurmak için: pip install shap")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(DATA_PATH, "financeai.db")


# ══════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME
# ══════════════════════════════════════════════════════════

def load_transactions_with_labels():
    print("\n📦 İşlemler yükleniyor...")

    label_path = f"{DATA_PATH}/train_fraud_labels.json"
    fraud_ids = set()
    if os.path.exists(label_path):
        with open(label_path, encoding='utf-8') as f:
            raw = json.load(f)
        labels = raw.get("target", raw)
        fraud_ids = {int(k) for k, v in labels.items()
                     if str(v).strip().lower() in ['yes', '1', 'true']}
        print(f"  Fraud işlem: {len(fraud_ids):,}")

    parquet_path = f"{DATA_PATH}/transactions_clean.parquet"
    csv_path     = f"{DATA_PATH}/transactions_data.csv"

    chunks = []
    try:
        import pyarrow.parquet as pq
        if os.path.exists(parquet_path):
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(batch_size=500_000):
                chunks.append(_process_tx_chunk(batch.to_pandas(), fraud_ids))
        else:
            raise FileNotFoundError
    except Exception:
        for chunk in pd.read_csv(csv_path, chunksize=500_000, parse_dates=['date']):
            chunks.append(_process_tx_chunk(chunk, fraud_ids))

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Toplam işlem  : {len(df):,}")
    print(f"  Fraud işlem   : {df['fraud_label'].sum():,} (%{df['fraud_label'].mean()*100:.2f})")
    print(f"  Unique müşteri: {df['client_id'].nunique():,}")
    return df


def _process_tx_chunk(chunk, fraud_ids):
    if 'amount' in chunk.columns:
        chunk['amount'] = pd.to_numeric(
            chunk['amount'].astype(str).str.replace('$', '', regex=False), errors='coerce'
        ).fillna(0)
    elif 'abs_amount' in chunk.columns:
        chunk['amount'] = chunk['abs_amount']

    chunk['errors']       = chunk.get('errors', pd.Series(['None'] * len(chunk))).fillna('None')
    chunk['tarih']        = pd.to_datetime(chunk['date'])
    chunk['saat']         = chunk['tarih'].dt.hour
    chunk['gun']          = chunk['tarih'].dt.dayofweek
    chunk['ay']           = chunk['tarih'].dt.month

    chunk['gece_islemi']  = ((chunk['saat'] >= 22) | (chunk['saat'] <= 6)).astype(int)
    chunk['hafta_sonu']   = (chunk['gun'] >= 5).astype(int)
    chunk['hata_var']     = (chunk['errors'] != 'None').astype(int)
    chunk['online_islem'] = (chunk.get('use_chip', '') == 'Online Transaction').astype(int)
    chunk['buyuk_islem']  = (chunk['amount'] > 1000).astype(int)
    chunk['negatif']      = (chunk['amount'] < 0).astype(int)
    chunk['abs_amount']   = chunk['amount'].abs()

    # YENİ: yuvarlak tutar (fraud sinyali)
    chunk['yuvarlak_tutar'] = (chunk['amount'] % 100 == 0).astype(int)
    # YENİ: tekrarlayan tutar (aynı müşteriden aynı tutar)
    chunk['tutar_str'] = chunk['amount'].round(2).astype(str)

    if 'id' in chunk.columns:
        chunk['fraud_label'] = chunk['id'].isin(fraud_ids).astype(int)
    else:
        chunk['fraud_label'] = 0

    cols = ['client_id', 'amount', 'abs_amount', 'saat', 'gun', 'ay',
            'gece_islemi', 'hafta_sonu', 'hata_var', 'online_islem',
            'buyuk_islem', 'negatif', 'yuvarlak_tutar', 'fraud_label', 'tarih']
    if 'merchant_id' in chunk.columns: cols.append('merchant_id')
    if 'mcc'         in chunk.columns: cols.append('mcc')

    return chunk[[c for c in cols if c in chunk.columns]]


def add_client_context(df_tx):
    print("  Müşteri bağlamı ekleniyor...")
    ctx = df_tx.groupby('client_id').agg(
        musteri_ort_tutar    =('amount', 'mean'),
        musteri_std_tutar    =('amount', 'std'),
        musteri_islem_sayisi =('amount', 'count'),
        musteri_gece_oran    =('gece_islemi', 'mean'),
        musteri_hata_oran    =('hata_var', 'mean'),
        musteri_online_oran  =('online_islem', 'mean'),
    ).reset_index()
    ctx['musteri_std_tutar'] = ctx['musteri_std_tutar'].fillna(0)
    df_tx = df_tx.merge(ctx, on='client_id', how='left')
    df_tx['tutar_sapma'] = (
        (df_tx['amount'] - df_tx['musteri_ort_tutar']) /
        (df_tx['musteri_std_tutar'] + 1)
    )
    try:
        users = pd.read_csv(f"{DATA_PATH}/users_data.csv")
        for col in ['per_capita_income', 'yearly_income', 'total_debt']:
            if col in users.columns:
                users[col] = users[col].astype(str)\
                    .str.replace('$', '', regex=False)\
                    .str.replace(',', '', regex=False).astype(float)
        users = users.rename(columns={'id': 'client_id'})
        ucols = ['client_id', 'credit_score', 'total_debt', 'yearly_income']
        ucols = [c for c in ucols if c in users.columns]
        df_tx = df_tx.merge(users[ucols], on='client_id', how='left')
        if 'total_debt' in df_tx.columns and 'yearly_income' in df_tx.columns:
            df_tx['borc_gelir_orani'] = df_tx['total_debt'] / (df_tx['yearly_income'] + 1)
    except Exception as e:
        print(f"  ⚠️  User verisi eklenemedi: {e}")

    return df_tx.fillna(0)


# ══════════════════════════════════════════════════════════
# 2. YENİ: MERCHANT RİSK SKORU
# ══════════════════════════════════════════════════════════

def build_merchant_risk(df_tx) -> pd.DataFrame:
    """
    Her merchant için fraud oranı hesapla.
    Bu skoru işlemlere feature olarak ekle.
    """
    if 'merchant_id' not in df_tx.columns:
        df_tx['merchant_risk'] = 0.0
        return df_tx

    print("  Merchant risk skoru hesaplanıyor...")
    merchant = df_tx.groupby('merchant_id').agg(
        m_islem_sayisi=('fraud_label', 'count'),
        m_fraud_oran  =('fraud_label', 'mean'),
        m_ort_tutar   =('amount', 'mean'),
        m_gece_oran   =('gece_islemi', 'mean'),
    ).reset_index()

    # Bayes smoothing — az işlemli merchantlar için shrinkage
    global_fraud_oran = df_tx['fraud_label'].mean()
    k = 50  # prior strength
    merchant['merchant_risk'] = (
        (merchant['m_fraud_oran'] * merchant['m_islem_sayisi'] + global_fraud_oran * k) /
        (merchant['m_islem_sayisi'] + k)
    )

    df_tx = df_tx.merge(
        merchant[['merchant_id', 'merchant_risk', 'm_gece_oran', 'm_ort_tutar']],
        on='merchant_id', how='left'
    )
    df_tx['merchant_risk'] = df_tx['merchant_risk'].fillna(global_fraud_oran)
    print(f"  ✅ {merchant['merchant_id'].nunique():,} merchant risk skoru hesaplandı")
    return df_tx


# ══════════════════════════════════════════════════════════
# 3. YENİ: VELOCITY FEATURES
# ══════════════════════════════════════════════════════════

def add_velocity_features(df_tx) -> pd.DataFrame:
    """
    Velocity: kısa sürede ani artış = fraud sinyali.
    Her işlem için son 1 saatteki işlem sayısı ve tutarı.
    """
    if 'tarih' not in df_tx.columns:
        return df_tx

    print("  Velocity features hesaplanıyor...")
    df_tx = df_tx.sort_values(['client_id', 'tarih'])

    # Son 1 saat
    df_tx['velocity_1h'] = (
        df_tx.groupby('client_id')['tarih']
        .transform(lambda x: x.diff().dt.total_seconds().lt(3600).cumsum())
        .fillna(0)
    )

    # Son 24 saat işlem sayısı (rolling per client)
    df_tx['velocity_24h_islem'] = (
        df_tx.groupby('client_id')['amount']
        .transform(lambda x: x.rolling(24, min_periods=1).count())
        .fillna(1)
    )

    # Son 24 saat toplam tutar
    df_tx['velocity_24h_tutar'] = (
        df_tx.groupby('client_id')['amount']
        .transform(lambda x: x.rolling(24, min_periods=1).sum())
        .fillna(0)
    )

    # Ani tutar artışı: bu işlem müşteri ortalamasının kaç katı?
    df_tx['tutar_carpani'] = (
        df_tx['abs_amount'] / (df_tx['musteri_ort_tutar'].abs() + 1)
    ).clip(0, 50)

    print("  ✅ Velocity features eklendi")
    return df_tx


# ══════════════════════════════════════════════════════════
# 4. MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════

TX_FEATURE_COLS = [
    'amount', 'abs_amount', 'saat', 'gun', 'ay',
    'gece_islemi', 'hafta_sonu', 'hata_var', 'online_islem',
    'buyuk_islem', 'negatif', 'yuvarlak_tutar',
    'musteri_ort_tutar', 'musteri_std_tutar', 'musteri_islem_sayisi',
    'musteri_gece_oran', 'musteri_hata_oran', 'musteri_online_oran',
    'tutar_sapma', 'merchant_risk',
    'velocity_1h', 'velocity_24h_islem', 'velocity_24h_tutar', 'tutar_carpani',
]

TIME_WINDOW_COLS = [
    'son30_islem_sayisi', 'son30_toplam_tutar', 'son30_gece_oran', 'son30_hata_oran',
    'son7_islem_sayisi',  'son7_toplam_tutar',
    'son90_islem_sayisi', 'son90_toplam_tutar',
    'tutar_degisim_7_30', 'islem_degisim_7_30',
    'gece_ani_artis',     'hata_ani_artis',
]


def _find_optimal_threshold(y_true, y_prob) -> float:
    """F1'i maksimize eden eşiği bul."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"  Optimal eşik: {threshold:.3f}  (F1={f1_scores[best_idx]:.4f})")
    return threshold


def train_tx_fraud_model(df_tx):
    print("\n" + "=" * 60)
    print("🤖 İşlem Bazlı XGBoost Fraud Modeli v6")
    print("=" * 60)

    fcols = [c for c in TX_FEATURE_COLS if c in df_tx.columns]
    X = df_tx[fcols].fillna(0)
    y = df_tx['fraud_label']

    print(f"  Özellik   : {len(fcols)}")
    print(f"  İşlem     : {len(X):,}")
    print(f"  Fraud oran: %{y.mean() * 100:.3f}")

    if len(X) > 500_000:
        print("  Örnekleme yapılıyor (500K)...")
        idx = pd.concat([
            X[y == 1].sample(min(len(X[y == 1]), 50_000), random_state=42),
            X[y == 0].sample(min(450_000, len(X[y == 0])), random_state=42),
        ]).index
        X_s, y_s = X.loc[idx], y.loc[idx]
    else:
        X_s, y_s = X, y

    Xtr, Xte, ytr, yte = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )

    scale = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"  Scale pos weight: {scale:.1f}")

    if XGB_AVAILABLE:
        base_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
            verbosity=0,
        )
    else:
        base_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, random_state=42,
        )

    # Kalibrasyon — skor dağılımını gerçekçi hale getir
    model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
    model.fit(Xtr, ytr)

    ypr  = model.predict_proba(Xte)[:, 1]
    threshold = _find_optimal_threshold(yte, ypr)
    yp   = (ypr >= threshold).astype(int)

    auc  = roc_auc_score(yte, ypr)
    f1   = f1_score(yte, yp)
    prec = precision_score(yte, yp, zero_division=0)
    rec  = recall_score(yte, yp, zero_division=0)

    metrics = dict(
        model            = 'XGBoost+Calibrated' if XGB_AVAILABLE else 'GradientBoosting+Calibrated',
        mod_tipi         = 'islem_bazli',
        auc_roc          = round(auc, 4),
        f1_skoru         = round(f1, 4),
        precision        = round(prec, 4),
        recall           = round(rec, 4),
        accuracy         = round((yp == yte).mean(), 4),
        optimal_threshold= round(threshold, 4),
        confusion_matrix = confusion_matrix(yte, yp).tolist(),
        egitim_tarihi    = datetime.now().isoformat(),
        toplam_ornek     = len(X_s),
        fraud_orani      = round(y.mean(), 4),
    )

    print(f"\n📊 AUC:{auc:.4f}  F1:{f1:.4f}  P:{prec:.4f}  R:{rec:.4f}")

    # Feature importance
    _extract_feature_importance(model, fcols, metrics)

    # Tüm işlemler için skor
    print("  Tüm işlemler skorlanıyor...")
    BATCH = 100_000
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), BATCH):
        probs[i:i + BATCH] = model.predict_proba(
            X.iloc[i:i + BATCH])[:, 1].astype(np.float32)
    df_tx['fraud_prob_tx'] = probs
    del probs

    joblib.dump(model,     f"{MODEL_PATH}/xgboost_fraud.pkl")
    joblib.dump(fcols,     f"{MODEL_PATH}/feature_cols.pkl")
    joblib.dump(threshold, f"{MODEL_PATH}/optimal_threshold.pkl")

    with open(f"{MODEL_PATH}/model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ Fraud modeli kaydedildi")
    return df_tx, metrics, fcols, threshold


def _extract_feature_importance(model, fcols, metrics):
    """Kalibreli modelden feature importance çek."""
    try:
        if hasattr(model, 'estimators_'):
            # CalibratedClassifierCV — ilk base estimator'dan al
            base = model.estimators_[0].estimator if hasattr(
                model.estimators_[0], 'estimator') else model.estimators_[0]
        elif hasattr(model, 'calibrated_classifiers_'):
            base = model.calibrated_classifiers_[0].estimator
        else:
            base = model

        if hasattr(base, 'feature_importances_'):
            imp = pd.DataFrame({
                'feature': fcols,
                'importance': base.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = imp.head(20).to_dict('records')
            print("🔍 Top 5:", ", ".join(
                f"{r.feature}({r.importance:.3f})" for _, r in imp.head(5).iterrows()
            ))
    except Exception as e:
        print(f"  ⚠️  Feature importance alınamadı: {e}")


# ══════════════════════════════════════════════════════════
# 5. YENİ: SHAP AÇIKLANABILIRLIK
# ══════════════════════════════════════════════════════════

def compute_shap_explanations(model, X_sample: pd.DataFrame, fcols: list) -> dict:
    """
    SHAP değerleri hesapla.
    Her müşteri için: hangi özellik ne kadar fraud skoruna katkı yaptı?

    Döner:
        {
          "global": [{"feature": "tx_hata_oran", "mean_shap": 0.12}, ...],
          "sample":  [{"client_id": 1, "top_reasons": [...]}]
        }
    """
    if not SHAP_AVAILABLE:
        print("  ⚠️  SHAP yüklü değil, atlanıyor. pip install shap")
        return {}

    print("\n🔍 SHAP açıklanabilirlik hesaplanıyor...")

    try:
        # Kalibreli modelden base estimator'u al
        try:
            base = model.calibrated_classifiers_[0].estimator
        except Exception:
            base = model

        X_s = X_sample[fcols].fillna(0)

        # TreeExplainer XGBoost/GBM için hızlı
        explainer = shap.TreeExplainer(base)
        shap_values = explainer.shap_values(X_s)

        # binary classification → shap_values shape: (n, features)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # Global önem: ortalama |SHAP|
        mean_shap = np.abs(sv).mean(axis=0)
        global_imp = pd.DataFrame({
            'feature': fcols,
            'mean_shap': mean_shap.round(4)
        }).sort_values('mean_shap', ascending=False)

        print("  SHAP Top 5:", ", ".join(
            f"{r.feature}({r.mean_shap:.3f})" for _, r in global_imp.head(5).iterrows()
        ))

        # Örnek müşteriler için bireysel açıklama (ilk 1000)
        sample_explanations = []
        n_sample = min(1000, len(X_s))
        for i in range(n_sample):
            row_shap = sv[i]
            top_idx  = np.argsort(np.abs(row_shap))[::-1][:5]
            reasons  = [
                {
                    "feature": fcols[j],
                    "shap":    round(float(row_shap[j]), 4),
                    "value":   round(float(X_s.iloc[i, j]), 4),
                    "etki":    "artirdi" if row_shap[j] > 0 else "azaltti"
                }
                for j in top_idx
            ]
            sample_explanations.append({
                "idx":         i,
                "top_reasons": reasons
            })

        result = {
            "global":  global_imp.head(20).to_dict('records'),
            "sample":  sample_explanations[:100],  # ilk 100 kaydet
        }

        # Kaydet
        shap_path = f"{MODEL_PATH}/shap_explanations.json"
        with open(shap_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✅ SHAP kaydedildi → {shap_path}")
        return result

    except Exception as e:
        print(f"  ⚠️  SHAP hesaplanamadı: {e}")
        return {}


def get_client_explanation(client_id: int) -> dict:
    """
    Tek bir müşteri için açıklama döner.
    API endpoint'i için kullanılır: GET /clients/{id}/explain
    """
    shap_path = f"{MODEL_PATH}/shap_explanations.json"
    if not os.path.exists(shap_path):
        return {"error": "SHAP hesaplanmamış. ml_model.py çalıştırın."}

    # client_ml tablosundan skoru al
    conn = sqlite3.connect(DB_PATH)
    try:
        row = pd.read_sql(
            f"SELECT * FROM client_ml WHERE client_id={client_id}", conn
        ).iloc[0].to_dict()
    except Exception:
        conn.close()
        return {"error": f"Müşteri {client_id} bulunamadı."}
    conn.close()

    fraud_skoru = row.get("fraud_skoru", 0)
    fraud_tahmini = row.get("fraud_tahmini", "Normal")

    # İnsan okunabilir açıklama üret
    reasons = []
    feature_labels = {
        "tx_hata_oran":      "Yüksek hata oranı",
        "dark_web_oran":     "Kartta dark web ihlali",
        "tx_gece_oran":      "Gece saatlerinde işlem yoğunluğu",
        "velocity_1h":       "Kısa sürede çok işlem",
        "tutar_carpani":     "Olağandışı yüksek tutar",
        "merchant_risk":     "Riskli merchant aktivitesi",
        "fraud_skoru_xgb":   "ML anomali skoru yüksek",
        "hata_ani_artis":    "Ani hata artışı",
        "gece_ani_artis":    "Gece işlemlerinde ani artış",
        "borc_gelir_orani":  "Yüksek borç/gelir oranı",
    }

    for key, label in feature_labels.items():
        val = row.get(key, 0)
        if isinstance(val, (int, float)) and val > 0.1:
            reasons.append({"sebep": label, "deger": round(float(val), 3)})

    return {
        "client_id":     client_id,
        "fraud_skoru":   fraud_skoru,
        "fraud_tahmini": fraud_tahmini,
        "aciklama":      reasons[:5] if reasons else [{"sebep": "Genel profil riski", "deger": fraud_skoru}],
        "ozet":          _generate_summary(fraud_skoru, reasons),
    }


def _generate_summary(skor: float, reasons: list) -> str:
    if skor < 20:
        return "Bu müşteri düşük risk profiline sahip, anormal bir aktivite tespit edilmedi."
    elif skor < 50:
        neden = reasons[0]["sebep"] if reasons else "genel aktivite profili"
        return f"Orta düzey risk. Öne çıkan sinyal: {neden}."
    else:
        nedenler = ", ".join(r["sebep"] for r in reasons[:3]) if reasons else "birden fazla sinyal"
        return f"Yüksek risk. Tetikleyen faktörler: {nedenler}."


# ══════════════════════════════════════════════════════════
# 6. MÜŞTERİ BAZINA TOPLAMA
# ══════════════════════════════════════════════════════════

def aggregate_to_client(df_tx):
    print("\n👤 Müşteri bazına toplanıyor...")

    needed = ['client_id', 'fraud_prob_tx', 'amount', 'gece_islemi', 'hata_var',
              'online_islem', 'hafta_sonu', 'buyuk_islem', 'negatif', 'tarih',
              'yuvarlak_tutar', 'merchant_risk', 'velocity_1h', 'tutar_carpani']
    for col in ['credit_score', 'total_debt', 'yearly_income', 'borc_gelir_orani']:
        if col in df_tx.columns:
            needed.append(col)
    df_small = df_tx[[c for c in needed if c in df_tx.columns]]

    client = df_small.groupby('client_id').agg(
        fraud_skoru_xgb    =('fraud_prob_tx', lambda x: x.mean() * 100),
        fraud_max_tx       =('fraud_prob_tx', 'max'),
        fraud_tx_sayisi    =('fraud_prob_tx', lambda x: (x > 0.5).sum()),
        tx_islem_sayisi    =('amount', 'count'),
        tx_toplam_tutar    =('amount', 'sum'),
        tx_ortalama_tutar  =('amount', 'mean'),
        tx_std_tutar       =('amount', 'std'),
        tx_max_tutar       =('amount', 'max'),
        tx_gece_oran       =('gece_islemi', 'mean'),
        tx_hata_oran       =('hata_var', 'mean'),
        tx_online_oran     =('online_islem', 'mean'),
        tx_hafta_sonu_oran =('hafta_sonu', 'mean'),
        tx_buyuk_oran      =('buyuk_islem', 'mean'),
        tx_negatif_oran    =('negatif', 'mean'),
        tx_yuvarlak_oran   =('yuvarlak_tutar', 'mean'),
        merchant_risk_ort  =('merchant_risk', 'mean'),
        merchant_risk_max  =('merchant_risk', 'max'),
        velocity_1h_max    =('velocity_1h', 'max'),
        tutar_carpani_max  =('tutar_carpani', 'max'),
        son_islem_tarihi   =('tarih', 'max'),
    ).reset_index()

    for col in ['credit_score', 'total_debt', 'yearly_income', 'borc_gelir_orani']:
        if col in df_small.columns:
            client = client.merge(
                df_small.groupby('client_id')[col].first().reset_index(),
                on='client_id', how='left'
            )

    client = client.fillna(0)
    client = _add_time_window_features(df_tx, client)
    print(f"✅ {len(client):,} müşteri, {len(client.columns)} özellik")
    return client


# ══════════════════════════════════════════════════════════
# 7. ZENGİNLEŞTİRME — TÜM MÜŞTERİLER
# ══════════════════════════════════════════════════════════

def enrich_with_all_clients(df_client):
    print("\n🔗 Tüm müşteri tabanıyla birleştiriliyor...")
    conn = sqlite3.connect(DB_PATH)

    try:
        ozet = pd.read_sql("SELECT client_id, toplam, islem, hata FROM client_ozet", conn)
        print(f"  client_ozet: {len(ozet):,}")
    except Exception as e:
        print(f"  ⚠️  client_ozet okunamadı: {e}")
        ozet = pd.DataFrame(columns=['client_id', 'toplam', 'islem', 'hata'])

    try:
        risk = pd.read_sql("SELECT client_id, risk_skoru FROM client_risk", conn)
        print(f"  client_risk: {len(risk):,}")
    except Exception:
        risk = pd.DataFrame(columns=['client_id', 'risk_skoru'])

    conn.close()

    if not ozet.empty:
        merged = ozet[['client_id']].merge(df_client, on='client_id', how='left')
        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        col_means = df_client[numeric_cols].mean()
        for col in numeric_cols:
            if col != 'client_id':
                merged[col] = merged[col].fillna(col_means.get(col, 0))
        df_client = merged
        print(f"  ✅ Birleştirme sonrası: {len(df_client):,}")

    if not risk.empty:
        df_client = df_client.merge(risk, on='client_id', how='left')
        df_client['risk_skoru'] = df_client['risk_skoru'].fillna(0)
    else:
        df_client['risk_skoru'] = 0.0

    return df_client


# ══════════════════════════════════════════════════════════
# 8. ZAMAN PENCERESİ
# ══════════════════════════════════════════════════════════

def _add_time_window_features(df_tx, df_client):
    if 'tarih' not in df_tx.columns:
        return df_client

    if df_tx['tarih'].dtype == object:
        df_tx = df_tx.copy()
        df_tx['tarih'] = pd.to_datetime(df_tx['tarih'])

    bugun = df_tx['tarih'].max()

    def window_agg(days, prefix):
        sub = df_tx[df_tx['tarih'] >= bugun - pd.Timedelta(days=days)]
        if len(sub) == 0:
            return pd.DataFrame(columns=['client_id'])
        return sub.groupby('client_id').agg(**{
            f'{prefix}_islem_sayisi': ('amount', 'count'),
            f'{prefix}_toplam_tutar': ('amount', 'sum'),
            f'{prefix}_gece_oran':    ('gece_islemi', 'mean'),
            f'{prefix}_hata_oran':    ('hata_var', 'mean'),
        }).reset_index()

    for days, prefix in [(7, 'son7'), (30, 'son30'), (90, 'son90')]:
        df_client = df_client.merge(window_agg(days, prefix), on='client_id', how='left')

    df_client = df_client.fillna(0)

    df_client['tutar_degisim_7_30'] = (
        df_client.get('son7_toplam_tutar', 0) /
        (df_client.get('son30_toplam_tutar', 0) / 4.3 + 1e-6)
    ).clip(0, 10)

    df_client['islem_degisim_7_30'] = (
        df_client.get('son7_islem_sayisi', 0) /
        (df_client.get('son30_islem_sayisi', 0) / 4.3 + 1e-6)
    ).clip(0, 10)

    df_client['gece_ani_artis'] = (
        df_client.get('son7_gece_oran', 0) - df_client.get('son90_gece_oran', 0)
    ).clip(-1, 1)

    df_client['hata_ani_artis'] = (
        df_client.get('son7_hata_oran', 0) - df_client.get('son90_hata_oran', 0)
    ).clip(-1, 1)

    return df_client


# ══════════════════════════════════════════════════════════
# 9. KART & KULLANICI ÖZELLİKLERİ
# ══════════════════════════════════════════════════════════

def build_user_features():
    df = pd.read_csv(f"{DATA_PATH}/users_data.csv")
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        if col in df.columns:
            df[col] = df[col].astype(str)\
                .str.replace('$', '', regex=False)\
                .str.replace(',', '', regex=False).astype(float)
    cols = [c for c in ['id', 'current_age', 'credit_score', 'num_credit_cards',
                         'per_capita_income', 'yearly_income', 'total_debt'] if c in df.columns]
    feat = df[cols].copy()
    feat.columns = ['client_id'] + cols[1:]
    if 'total_debt' in feat.columns and 'yearly_income' in feat.columns:
        feat['borc_gelir_orani'] = feat['total_debt'] / (feat['yearly_income'] + 1)

    # YENİ: RFM benzeri özellikler
    if 'yearly_income' in feat.columns:
        feat['gelir_segment'] = pd.qcut(
            feat['yearly_income'], q=4, labels=[1, 2, 3, 4], duplicates='drop'
        ).astype(float)

    return feat.fillna(0)


def build_card_features():
    df = pd.read_csv(f"{DATA_PATH}/cards_data.csv")
    df['credit_limit'] = df['credit_limit'].astype(str)\
        .str.replace('$', '', regex=False)\
        .str.replace(',', '', regex=False).astype(float)
    df['dark_web_flag'] = (df['card_on_dark_web'] == 'Yes').astype(int)

    feat = df.groupby('client_id').agg(
        kart_adedi    =('id', 'count'),
        toplam_limit  =('credit_limit', 'sum'),
        dark_web_oran =('dark_web_flag', 'mean'),
        chip_oran     =('has_chip', lambda x: (x == 'YES').mean()),
    ).reset_index()
    feat['dark_web_kart'] = (feat['dark_web_oran'] > 0).astype(int)

    # YENİ: limit kullanım yoğunluğu (dışarıdan tx ile birleşince)
    feat['limit_basi_kart'] = feat['toplam_limit'] / (feat['kart_adedi'] + 1)
    return feat.fillna(0)


# ══════════════════════════════════════════════════════════
# 10. ISOLATION FOREST
# ══════════════════════════════════════════════════════════

CLIENT_FEATURE_COLS = [
    'fraud_skoru_xgb', 'fraud_max_tx', 'fraud_tx_sayisi',
    'tx_islem_sayisi', 'tx_toplam_tutar', 'tx_ortalama_tutar',
    'tx_std_tutar', 'tx_max_tutar', 'tx_gece_oran', 'tx_hata_oran',
    'tx_online_oran', 'tx_hafta_sonu_oran', 'tx_buyuk_oran', 'tx_negatif_oran',
    'tx_yuvarlak_oran', 'merchant_risk_ort', 'merchant_risk_max',
    'velocity_1h_max', 'tutar_carpani_max',
    'dark_web_kart', 'kart_adedi', 'toplam_limit', 'dark_web_oran',
    'son30_islem_sayisi', 'son30_toplam_tutar', 'son30_gece_oran', 'son30_hata_oran',
    'son7_islem_sayisi',  'son7_toplam_tutar',
    'tutar_degisim_7_30', 'islem_degisim_7_30',
    'gece_ani_artis',     'hata_ani_artis',
    'risk_skoru',
]


def train_isolation_forest(df_client):
    print("\n🌲 Isolation Forest")
    fcols  = [c for c in CLIENT_FEATURE_COLS if c in df_client.columns]
    X      = df_client[fcols].fillna(0)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    iso    = IsolationForest(
        n_estimators=300, contamination=0.05,
        max_features=0.8, random_state=42, n_jobs=-1
    )
    iso.fit(Xs)
    df_client['anomali_skoru'] = iso.decision_function(Xs)
    df_client['iso_tahmin']    = (iso.predict(Xs) == -1).astype(int)
    joblib.dump(iso,    f"{MODEL_PATH}/isolation_forest.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
    print(f"  ✅ Anomali tespit: {df_client['iso_tahmin'].sum():,} müşteri")
    return df_client


# ══════════════════════════════════════════════════════════
# 11. GELİŞTİRİLMİŞ CHURN MODELİ
# ══════════════════════════════════════════════════════════

def build_churn_labels(df_client):
    """
    Gelişmiş churn: sadece son işlem tarihi değil,
    işlem sıklığı + tutar düşüşü de dikkate alınır.
    """
    print("\n📉 Churn etiketleri üretiliyor...")

    df_client['churn_label'] = 0
    df_client['churn_riski'] = 0.5

    if 'son_islem_tarihi' not in df_client.columns:
        return df_client

    bugun = pd.Timestamp.now()
    df_client['son_islem_tarihi'] = pd.to_datetime(
        df_client['son_islem_tarihi'], errors='coerce'
    )
    df_client['gun_fark'] = (
        bugun - df_client['son_islem_tarihi']
    ).dt.days.fillna(9999)

    # RFM: Recency + Frequency + Monetary
    df_client['rfm_recency']   = df_client['gun_fark'].rank(pct=True, ascending=False)
    df_client['rfm_frequency'] = df_client.get('tx_islem_sayisi', 0).rank(pct=True)
    df_client['rfm_monetary']  = df_client.get('tx_toplam_tutar', 0).rank(pct=True)
    df_client['rfm_skor']      = (
        df_client['rfm_recency'] * 0.5 +
        df_client['rfm_frequency'] * 0.3 +
        df_client['rfm_monetary'] * 0.2
    )

    # Dinamik eşik
    esik = float(df_client['gun_fark'].quantile(0.5))
    df_client['churn_label'] = (df_client['gun_fark'] >= esik).astype(int)

    # Churn riski: yüksek recency + düşük frequency = yüksek risk
    df_client['churn_riski'] = np.clip(
        df_client['rfm_recency'] * 0.6 +
        (1 - df_client['rfm_frequency']) * 0.4, 0, 1
    )

    print(f"  Eşik: {esik:.0f} gün | "
          f"Churn=1: {df_client['churn_label'].sum():,} | "
          f"Churn=0: {(df_client['churn_label']==0).sum():,}")
    return df_client


def train_churn_model(df_client):
    print("\n📉 Churn Modeli")
    fcols = [c for c in CLIENT_FEATURE_COLS + ['rfm_skor', 'gun_fark'] if c in df_client.columns]
    X = df_client[fcols].fillna(0)
    y = df_client['churn_label']

    if y.nunique() < 2:
        print("  ⚠️  Churn atlandı (tek sınıf)")
        df_client['churn_skoru']   = 0.0
        df_client['churn_tahmini'] = 'Dusuk Risk'
        return df_client, {'churn_auc': 0.0}

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scale = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    if XGB_AVAILABLE:
        base = xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.04,
            scale_pos_weight=scale, random_state=42,
            n_jobs=-1, verbosity=0,
        )
    else:
        base = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )

    model = CalibratedClassifierCV(base, cv=3, method='isotonic')
    model.fit(Xtr, ytr)

    ypr = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, ypr)

    df_client['churn_skoru'] = (model.predict_proba(X)[:, 1] * 100).round(2)
    df_client['churn_tahmini'] = pd.cut(
        df_client['churn_skoru'],
        bins=[-np.inf, 30, 60, np.inf],
        labels=['Dusuk Risk', 'Orta Risk', 'Yuksek Risk']
    ).astype(str)

    joblib.dump(model, f"{MODEL_PATH}/churn_model.pkl")
    print(f"  ✅ AUC: {auc:.4f}")
    return df_client, {'churn_auc': round(auc, 4)}


# ══════════════════════════════════════════════════════════
# 12. FİNAL SKOR
# ══════════════════════════════════════════════════════════

def compute_final_fraud_score(df_client):
    print("\n⚡ Final fraud skoru hesaplanıyor...")
    s = np.zeros(len(df_client))
    agirliklar = {}

    if 'fraud_skoru_xgb' in df_client.columns:
        var = df_client['fraud_skoru_xgb'].var()
        w   = 0.55 if var > 5 else (0.45 if var > 1 else 0.35)
        s  += df_client['fraud_skoru_xgb'] * w
        agirliklar['xgb'] = w

    iso_w = 0.0
    if 'anomali_skoru' in df_client.columns:
        rng = df_client['anomali_skoru'].max() - df_client['anomali_skoru'].min()
        if rng > 1e-6:
            iso_n = (-df_client['anomali_skoru'] - df_client['anomali_skoru'].min()) / rng
            iso_w = 0.25 if rng > 0.5 else 0.15
            s    += iso_n * (iso_w * 100)
            agirliklar['iso'] = iso_w

    xgb_w  = agirliklar.get('xgb', 0)
    kural_w = max(round(1 - xgb_w - iso_w, 3), 0.10)
    kural_s = np.zeros(len(df_client))
    if 'dark_web_oran'       in df_client.columns: kural_s += df_client['dark_web_oran']       * 0.50
    if 'tx_hata_oran'        in df_client.columns: kural_s += df_client['tx_hata_oran']        * 0.20
    if 'merchant_risk_max'   in df_client.columns: kural_s += df_client['merchant_risk_max']   * 0.15
    if 'hata_ani_artis'      in df_client.columns: kural_s += df_client['hata_ani_artis'].clip(0,1) * 0.10
    if 'tutar_degisim_7_30'  in df_client.columns:
        kural_s += (df_client['tutar_degisim_7_30'] - 1).clip(0, 5) / 5 * 0.05
    s += kural_s * (kural_w * 100)

    print(f"  Ağırlıklar: XGB:{xgb_w:.0%}  ISO:{iso_w:.0%}  Kural:{kural_w:.0%}")

    df_client['fraud_skoru'] = np.clip(s, 0, 100).round(2)

    esik_yuksek  = float(df_client['fraud_skoru'].quantile(0.90))
    esik_supheli = float(df_client['fraud_skoru'].quantile(0.75))
    print(f"  Eşikler — Şüpheli:{esik_supheli:.1f}  Yüksek:{esik_yuksek:.1f}")

    df_client['fraud_tahmini'] = pd.cut(
        df_client['fraud_skoru'],
        bins=[-np.inf, esik_supheli, esik_yuksek, np.inf],
        labels=['Normal', 'Supheli', 'Yuksek Risk']
    ).astype(str)

    return df_client


# ══════════════════════════════════════════════════════════
# 13. KAYIT
# ══════════════════════════════════════════════════════════

def save_results(df_client):
    print("\n💾 Kaydediliyor...")
    conn = sqlite3.connect(DB_PATH)

    save_cols = [c for c in [
        'client_id', 'fraud_skoru', 'fraud_tahmini',
        'fraud_skoru_xgb', 'anomali_skoru', 'iso_tahmin',
        'churn_skoru', 'churn_tahmini',
        'tx_gece_oran', 'tx_hata_oran', 'dark_web_oran',
        'tx_islem_sayisi', 'borc_gelir_orani',
        'fraud_max_tx', 'fraud_tx_sayisi',
        'risk_skoru',
        'merchant_risk_ort', 'merchant_risk_max',
        'velocity_1h_max', 'tutar_carpani_max',
        'son30_islem_sayisi', 'son30_toplam_tutar',
        'son30_gece_oran', 'son30_hata_oran',
        'son7_islem_sayisi', 'son7_toplam_tutar',
        'tutar_degisim_7_30', 'islem_degisim_7_30',
        'gece_ani_artis', 'hata_ani_artis',
        'rfm_skor', 'gun_fark',
    ] if c in df_client.columns]

    df_client[save_cols].to_sql("client_ml", conn, if_exists="replace", index=False)
    print(f"  ✅ client_ml: {len(df_client):,} kayıt")

    ozet = {
        'toplam'          : int(len(df_client)),
        'normal'          : int((df_client['fraud_tahmini'] == 'Normal').sum()),
        'supheli'         : int((df_client['fraud_tahmini'] == 'Supheli').sum()),
        'yuksek_risk'     : int((df_client['fraud_tahmini'] == 'Yuksek Risk').sum()),
        'churn_yuksek'    : int((df_client.get('churn_tahmini', '') == 'Yuksek Risk').sum()),
        'ort_fraud_skoru' : round(float(df_client['fraud_skoru'].mean()), 2),
        'max_fraud_skoru' : round(float(df_client['fraud_skoru'].max()), 2),
        'hesaplama_tarihi': datetime.now().isoformat(),
    }
    pd.DataFrame([ozet]).to_sql("ml_ozet", conn, if_exists="replace", index=False)
    conn.close()
    print("  ✅ ml_ozet kaydedildi")
    return ozet


# ══════════════════════════════════════════════════════════
# 14. ANA FONKSİYON
# ══════════════════════════════════════════════════════════

def train_model():
    print("\n" + "=" * 60)
    print("🚀 FinSight — ML Pipeline v6.0")
    print("=" * 60)
    t0 = datetime.now()

    # İşlem verisi
    df_tx = load_transactions_with_labels()
    df_tx = add_client_context(df_tx)

    # YENİ: Merchant risk + Velocity
    df_tx = build_merchant_risk(df_tx)
    df_tx = add_velocity_features(df_tx)

    # Fraud modeli
    df_tx, metrics, fcols, threshold = train_tx_fraud_model(df_tx)

    # YENİ: SHAP (sample üzerinde)
    sample_size = min(10_000, len(df_tx))
    sample_idx  = df_tx.sample(sample_size, random_state=42).index
    X_shap = df_tx.loc[sample_idx, [c for c in fcols if c in df_tx.columns]]
    model  = joblib.load(f"{MODEL_PATH}/xgboost_fraud.pkl")
    shap_result = compute_shap_explanations(model, X_shap, fcols)
    if shap_result:
        metrics['shap_global'] = shap_result.get('global', [])[:10]

    # Müşteri bazına topla
    df_client = aggregate_to_client(df_tx)
    del df_tx
    gc.collect()

    # Kullanıcı + kart özellikleri
    print("\n🔗 Profil zenginleştiriliyor...")
    try:
        user = build_user_features()
        df_client = df_client.merge(user, on='client_id', how='left')
    except Exception as e:
        print(f"  ⚠️  User: {e}")
    try:
        card = build_card_features()
        df_client = df_client.merge(card, on='client_id', how='left', suffixes=('', '_card'))
    except Exception as e:
        print(f"  ⚠️  Card: {e}")
    df_client = df_client.fillna(0)

    # Tüm müşteriler
    df_client = enrich_with_all_clients(df_client)

    # Modeller
    df_client = train_isolation_forest(df_client)
    df_client = build_churn_labels(df_client)
    df_client, cm = train_churn_model(df_client)
    metrics.update(cm)

    # Final skor
    df_client = compute_final_fraud_score(df_client)
    ozet = save_results(df_client)

    # Metrikleri güncelle
    with open(f"{MODEL_PATH}/model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    sure = (datetime.now() - t0).seconds
    print("\n" + "=" * 60)
    print("📊 SONUÇ")
    print("=" * 60)
    print(f"  Toplam   : {ozet['toplam']:,}")
    print(f"  Normal   : {ozet['normal']:,}  (%{ozet['normal']/ozet['toplam']*100:.1f})")
    print(f"  Supheli  : {ozet['supheli']:,}  (%{ozet['supheli']/ozet['toplam']*100:.1f})")
    print(f"  Yuk.Risk : {ozet['yuksek_risk']:,}  (%{ozet['yuksek_risk']/ozet['toplam']*100:.1f})")
    print(f"  Churn    : {ozet['churn_yuksek']:,}")
    print(f"  AUC-ROC  : {metrics.get('auc_roc', 'N/A')}")
    print(f"  F1 Skoru : {metrics.get('f1_skoru', 'N/A')}")
    print(f"  Süre     : {sure}s")
    print("=" * 60)
    print("✅ Tüm modeller hazır!")
    return df_client, metrics


# ══════════════════════════════════════════════════════════
# YARDIMCI — api.py tarafından kullanılır
# ══════════════════════════════════════════════════════════

def predict_client(client_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(f"SELECT * FROM client_ml WHERE client_id={client_id}", conn)
    conn.close()
    return df.iloc[0].to_dict() if len(df) else {}


def get_all_ml_results() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM client_ml ORDER BY fraud_skoru DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_model_metrics() -> dict:
    p = f"{MODEL_PATH}/model_metrics.json"
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    train_model()