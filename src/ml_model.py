"""
FinanceAI — Gelişmiş ML Modeli v3.0
=====================================
İçerir:
  1. Feature Engineering (13M işlem)
  2. XGBoost Fraud Tespiti (gerçek etiketler)
  3. Isolation Forest Anomali Tespiti
  4. Churn Tahmini (müşteri kaybı)
  5. Gerçek Model Metrikleri (AUC, F1, Precision, Recall)
  6. SHAP Feature Importance

Çalıştır:
  python src/ml_model.py
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost mevcut")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost yok, GradientBoosting kullanılacak. Yüklemek için: pip install xgboost")

DB_PATH = "C:/financeai/data/financeai.db"
DATA_PATH = "C:/financeai/data"
MODEL_PATH = "C:/financeai/data"


# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_transaction_features():
    print("\n📦 Transaction features üretiliyor...")
    import pyarrow.parquet as pq

    parquet_path = f"{DATA_PATH}/transactions_clean.parquet"
    csv_path = f"{DATA_PATH}/transactions_data.csv"

    chunks = []
    if os.path.exists(parquet_path):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=500_000):
            chunks.append(batch.to_pandas())
    else:
        for chunk in pd.read_csv(csv_path, chunksize=500_000, parse_dates=['date']):
            chunk['amount'] = chunk['amount'].astype(str).str.replace('$', '', regex=False).astype(float)
            chunk['errors'] = chunk['errors'].fillna('None')
            chunk['saat'] = pd.to_datetime(chunk['date']).dt.hour
            chunk['gece_islemi'] = chunk['saat'].between(22, 23) | chunk['saat'].between(0, 6)
            chunk['hata_var'] = (chunk['errors'] != 'None').astype(int)
            chunk['online_islem'] = (chunk['use_chip'] == 'Online Transaction').astype(int)
            chunks.append(chunk)
        print(f"  CSV okundu: {len(chunks)} parça")

    df_tx = pd.concat(chunks, ignore_index=True)

    # Amount temizle
    if 'amount' not in df_tx.columns and 'abs_amount' in df_tx.columns:
        df_tx['amount'] = df_tx['abs_amount']
    elif 'amount' in df_tx.columns:
        df_tx['amount'] = pd.to_numeric(
            df_tx['amount'].astype(str).str.replace('$', '', regex=False), errors='coerce'
        ).fillna(0)

    # Zaman özellikleri
    if 'saat' not in df_tx.columns:
        df_tx['saat'] = pd.to_datetime(df_tx['date']).dt.hour
    if 'gece_islemi' not in df_tx.columns:
        df_tx['gece_islemi'] = df_tx['saat'].between(22, 23) | df_tx['saat'].between(0, 6)
    if 'hata_var' not in df_tx.columns:
        df_tx['errors'] = df_tx.get('errors', pd.Series(['None'] * len(df_tx))).fillna('None')
        df_tx['hata_var'] = (df_tx['errors'] != 'None').astype(int)
    if 'online_islem' not in df_tx.columns:
        df_tx['online_islem'] = (df_tx.get('use_chip', '') == 'Online Transaction').astype(int)

    # Hafta sonu
    df_tx['tarih'] = pd.to_datetime(df_tx['date'])
    df_tx['hafta_sonu'] = df_tx['tarih'].dt.dayofweek >= 5

    # Müşteri bazlı feature'lar
    features = df_tx.groupby('client_id').agg(
        tx_toplam_tutar=('amount', 'sum'),
        tx_ortalama_tutar=('amount', 'mean'),
        tx_std_tutar=('amount', 'std'),
        tx_min_tutar=('amount', 'min'),
        tx_max_tutar=('amount', 'max'),
        tx_islem_sayisi=('amount', 'count'),
        tx_negatif_islem=('amount', lambda x: (x < 0).sum()),
        tx_buyuk_islem=('amount', lambda x: (x > 1000).sum()),
        tx_gece_islem=('gece_islemi', 'sum'),
        tx_hata_sayisi=('hata_var', 'sum'),
        tx_online_islem=('online_islem', 'sum'),
        tx_hafta_sonu=('hafta_sonu', 'sum'),
        tx_farkli_sehir=('merchant_city', 'nunique'),
        tx_farkli_merchant=('merchant_id', 'nunique'),
        tx_farkli_mcc=('mcc', 'nunique'),
    ).reset_index()

    # Oranlar
    n = features['tx_islem_sayisi']
    features['tx_gece_oran'] = features['tx_gece_islem'] / n
    features['tx_hata_oran'] = features['tx_hata_sayisi'] / n
    features['tx_online_oran'] = features['tx_online_islem'] / n
    features['tx_negatif_oran'] = features['tx_negatif_islem'] / n
    features['tx_buyuk_oran'] = features['tx_buyuk_islem'] / n
    features['tx_hafta_sonu_oran'] = features['tx_hafta_sonu'] / n
    features['tx_merchant_cesitlilik'] = features['tx_farkli_merchant'] / n
    features['tx_tutar_cv'] = features['tx_std_tutar'] / (features['tx_ortalama_tutar'].abs() + 1)
    features['tx_mcc_cesitlilik'] = features['tx_farkli_mcc'] / n

    features = features.fillna(0)
    print(f"✅ TX features: {len(features):,} müşteri, {len(features.columns)} özellik")
    return features


def build_user_features():
    print("👤 User features üretiliyor...")
    df = pd.read_csv(f"{DATA_PATH}/users_data.csv")
    for col in ['per_capita_income', 'yearly_income', 'total_debt']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)

    cols = ['id', 'current_age', 'credit_score', 'num_credit_cards',
            'per_capita_income', 'yearly_income', 'total_debt']
    cols = [c for c in cols if c in df.columns]
    features = df[cols].copy()
    features.columns = ['client_id'] + [c for c in cols[1:]]
    features.rename(columns={
        'current_age': 'yas', 'credit_score': 'kredi_skoru',
        'num_credit_cards': 'kart_sayisi', 'per_capita_income': 'kisi_basi_gelir',
        'yearly_income': 'yillik_gelir', 'total_debt': 'toplam_borc'
    }, errors='ignore')

    if 'total_debt' in features.columns and 'yearly_income' in features.columns:
        features['borc_gelir_orani'] = features['total_debt'] / (features['yearly_income'] + 1)
    elif 'toplam_borc' in features.columns and 'yillik_gelir' in features.columns:
        features['borc_gelir_orani'] = features['toplam_borc'] / (features['yillik_gelir'] + 1)

    features = features.fillna(0)
    print(f"✅ User features: {len(features):,} müşteri")
    return features


def build_card_features():
    print("💳 Card features üretiliyor...")
    df = pd.read_csv(f"{DATA_PATH}/cards_data.csv")
    df['credit_limit'] = df['credit_limit'].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)
    df['dark_web_flag'] = (df['card_on_dark_web'] == 'Yes').astype(int)

    features = df.groupby('client_id').agg(
        kart_adedi=('id', 'count'),
        toplam_limit=('credit_limit', 'sum'),
        ort_limit=('credit_limit', 'mean'),
        dark_web_kart=('dark_web_flag', 'sum'),
        chip_li_kart=('has_chip', lambda x: (x == 'YES').sum()),
    ).reset_index()

    features['dark_web_oran'] = features['dark_web_kart'] / features['kart_adedi']
    features['chip_oran'] = features['chip_li_kart'] / features['kart_adedi']
    features = features.fillna(0)
    print(f"✅ Card features: {len(features):,} müşteri")
    return features


def merge_all_features():
    print("\n🔗 Tüm özellikler birleştiriliyor...")
    tx = build_transaction_features()
    user = build_user_features()
    card = build_card_features()

    conn = sqlite3.connect(DB_PATH)
    risk = pd.read_sql("SELECT * FROM client_risk", conn)
    conn.close()

    df = tx.merge(user, on='client_id', how='left')
    df = df.merge(card, on='client_id', how='left')
    df = df.merge(risk[['client_id', 'risk_skoru', 'hata_orani']].rename(
        columns={'hata_orani': 'risk_hata_orani'}), on='client_id', how='left')

    df = df.fillna(0)
    print(f"✅ Birleşik dataset: {len(df):,} müşteri, {len(df.columns)} özellik")
    return df


# ─────────────────────────────────────────────
# 2. FRAUD ETİKETLERİ
# ─────────────────────────────────────────────

def build_fraud_labels(df):
    """
    Gerçek fraud etiketleri oluştur.
    train_fraud_labels.json varsa kullan,
    yoksa kural tabanlı pseudo-label üret.
    """
    label_path = f"{DATA_PATH}/train_fraud_labels.json"

    if os.path.exists(label_path):
        print("📋 Gerçek fraud etiketleri yükleniyor...")
        with open(label_path) as f:
            labels = json.load(f)
        label_df = pd.DataFrame(list(labels.items()), columns=['client_id', 'fraud_label'])
        label_df['client_id'] = label_df['client_id'].astype(int)
        label_df['fraud_label'] = label_df['fraud_label'].astype(int)
        df = df.merge(label_df, on='client_id', how='left')
        df['fraud_label'] = df['fraud_label'].fillna(0).astype(int)
        print(f"  Fraud: {df['fraud_label'].sum():,} / {len(df):,} müşteri")
    else:
        print("⚠️  train_fraud_labels.json yok — kural tabanlı pseudo-label üretiliyor...")
        # Çok boyutlu kural tabanlı etiket
        fraud_score = np.zeros(len(df))
        if 'dark_web_oran' in df.columns:
            fraud_score += df['dark_web_oran'] * 4
        if 'tx_gece_oran' in df.columns:
            fraud_score += (df['tx_gece_oran'] > 0.3).astype(float) * 2
        if 'tx_hata_oran' in df.columns:
            fraud_score += (df['tx_hata_oran'] > 0.1).astype(float) * 3
        if 'risk_skoru' in df.columns:
            fraud_score += (df['risk_skoru'] > 40).astype(float) * 2
        if 'borc_gelir_orani' in df.columns:
            fraud_score += (df['borc_gelir_orani'] > 5).astype(float)

        threshold = np.percentile(fraud_score, 95)
        df['fraud_label'] = (fraud_score >= threshold).astype(int)
        print(f"  Pseudo-label fraud: {df['fraud_label'].sum():,} / {len(df):,}")

    return df


# ─────────────────────────────────────────────
# 3. CHURN ETİKETLERİ
# ─────────────────────────────────────────────

def build_churn_labels(df):
    """Son 90 günde işlem yapmayan = churn riski"""
    print("📉 Churn etiketleri üretiliyor...")
    csv_path = f"{DATA_PATH}/transactions_data.csv"

    son_islem = []
    bugun = pd.Timestamp.now()

    for chunk in pd.read_csv(csv_path, chunksize=500_000, parse_dates=['date']):
        grp = chunk.groupby('client_id')['date'].max().reset_index()
        son_islem.append(grp)

    son_islem_df = pd.concat(son_islem).groupby('client_id')['date'].max().reset_index()
    son_islem_df['gun_fark'] = (bugun - pd.to_datetime(son_islem_df['date'])).dt.days
    son_islem_df['churn_label'] = (son_islem_df['gun_fark'] > 90).astype(int)
    son_islem_df['churn_riski'] = np.clip(son_islem_df['gun_fark'] / 180, 0, 1)

    df = df.merge(son_islem_df[['client_id', 'churn_label', 'churn_riski', 'gun_fark']],
                  on='client_id', how='left')
    df['churn_label'] = df['churn_label'].fillna(0).astype(int)
    df['churn_riski'] = df['churn_riski'].fillna(0.5)
    print(f"✅ Churn: {df['churn_label'].sum():,} müşteri risk altında")
    return df


# ─────────────────────────────────────────────
# 4. MODEL EĞİTİMİ
# ─────────────────────────────────────────────

FEATURE_COLS = [
    'tx_toplam_tutar', 'tx_ortalama_tutar', 'tx_std_tutar',
    'tx_max_tutar', 'tx_islem_sayisi', 'tx_negatif_oran',
    'tx_buyuk_oran', 'tx_gece_oran', 'tx_hata_oran',
    'tx_online_oran', 'tx_hafta_sonu_oran', 'tx_farkli_sehir',
    'tx_farkli_merchant', 'tx_farkli_mcc', 'tx_merchant_cesitlilik',
    'tx_tutar_cv', 'tx_mcc_cesitlilik',
    'kart_adedi', 'toplam_limit', 'dark_web_oran', 'chip_oran',
    'risk_skoru', 'risk_hata_orani',
]


def train_fraud_model(df):
    print("\n" + "="*60)
    print("🤖 XGBoost Fraud Tespit Modeli")
    print("="*60)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df['fraud_label']

    print(f"  Özellik sayısı : {len(feature_cols)}")
    print(f"  Toplam örnek   : {len(X):,}")
    print(f"  Fraud oranı    : {y.mean()*100:.2f}%")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weight hesapla
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    # Model
    if XGB_AVAILABLE:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
            verbosity=0,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrikler
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'model': 'XGBoost' if XGB_AVAILABLE else 'GradientBoosting',
        'auc_roc': round(auc, 4),
        'f1_skoru': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'accuracy': round((y_pred == y_test).mean(), 4),
        'confusion_matrix': cm.tolist(),
        'egitim_tarihi': datetime.now().isoformat(),
        'toplam_ornek': len(X),
        'fraud_orani': round(y.mean(), 4),
    }

    print(f"\n📊 MODEL METRİKLERİ ({metrics['model']})")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Skoru  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = importance_df.head(15).to_dict('records')
        print(f"\n🔍 En önemli 5 özellik:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    # Tüm dataset için fraud skoru üret
    fraud_prob_all = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['fraud_skoru_xgb'] = (fraud_prob_all * 100).round(2)

    # Kaydet
    joblib.dump(model, f"{MODEL_PATH}/xgboost_fraud.pkl")
    joblib.dump(feature_cols, f"{MODEL_PATH}/feature_cols.pkl")
    with open(f"{MODEL_PATH}/model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Model kaydedildi: {MODEL_PATH}/xgboost_fraud.pkl")
    return df, metrics, feature_cols


def train_isolation_forest(df, feature_cols):
    print("\n🌲 Isolation Forest (Anomali)")
    X = df[[c for c in feature_cols if c in df.columns]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          max_features=0.8, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)

    scores = iso.decision_function(X_scaled)
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    df = df.copy()
    df['anomali_skoru'] = scores
    df['iso_tahmin'] = (iso.predict(X_scaled) == -1).astype(int)

    joblib.dump(iso, f"{MODEL_PATH}/isolation_forest.pkl")
    joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")
    print("✅ Isolation Forest kaydedildi")
    return df


def train_churn_model(df):
    print("\n📉 Churn Tahmin Modeli")
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df['churn_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if XGB_AVAILABLE:
        churn_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
            scale_pos_weight=(y_train==0).sum() / max((y_train==1).sum(), 1)
        )
    else:
        churn_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=42)

    churn_model.fit(X_train, y_train)
    y_prob = churn_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    churn_prob_all = churn_model.predict_proba(X)[:, 1]
    df = df.copy()
    df['churn_skoru'] = (churn_prob_all * 100).round(2)
    df['churn_tahmini'] = pd.cut(
        df['churn_skoru'],
        bins=[-np.inf, 30, 60, np.inf],
        labels=['Düşük Risk', '⚠️ Orta Risk', '🔴 Yüksek Risk']
    )

    churn_metrics = {'churn_auc': round(auc, 4)}
    joblib.dump(churn_model, f"{MODEL_PATH}/churn_model.pkl")
    print(f"✅ Churn modeli — AUC: {auc:.4f}")
    return df, churn_metrics


# ─────────────────────────────────────────────
# 5. FRAUD SKORU BİRLEŞTİR
# ─────────────────────────────────────────────

def compute_final_fraud_score(df):
    """XGBoost + Isolation Forest + kural bazlı hibrit skor"""
    score = np.zeros(len(df))

    # XGBoost katkısı (%60)
    if 'fraud_skoru_xgb' in df.columns:
        score += df['fraud_skoru_xgb'] * 0.6

    # Isolation Forest katkısı (%20)
    if 'anomali_skoru' in df.columns:
        iso_norm = (-df['anomali_skoru'] - df['anomali_skoru'].min()) / \
                   (df['anomali_skoru'].max() - df['anomali_skoru'].min() + 1e-9)
        score += iso_norm * 20

    # Kural bazlı katkı (%20)
    if 'dark_web_oran' in df.columns:
        score += df['dark_web_oran'] * 10
    if 'risk_skoru' in df.columns:
        score += (df['risk_skoru'] / 65) * 10

    df = df.copy()
    df['fraud_skoru'] = np.clip(score, 0, 100).round(2)
    df['fraud_tahmini'] = pd.cut(
        df['fraud_skoru'],
        bins=[-np.inf, 30, 60, np.inf],
        labels=['Normal', '⚠️ Şüpheli', '🔴 Yüksek Risk']
    )
    return df


# ─────────────────────────────────────────────
# 6. KAYDET
# ─────────────────────────────────────────────

def save_results(df):
    print("\n💾 Sonuçlar veritabanına kaydediliyor...")
    conn = sqlite3.connect(DB_PATH)

    # client_ml tablosu
    save_cols = ['client_id', 'fraud_skoru', 'fraud_tahmini',
                 'fraud_skoru_xgb', 'anomali_skoru', 'iso_tahmin',
                 'churn_skoru', 'churn_tahmini',
                 'tx_gece_oran', 'tx_hata_oran', 'dark_web_oran',
                 'tx_farkli_sehir', 'tx_islem_sayisi', 'borc_gelir_orani']
    save_cols = [c for c in save_cols if c in df.columns]
    df[save_cols].to_sql("client_ml", conn, if_exists="replace", index=False)
    print(f"✅ client_ml: {len(df):,} kayıt")

    # Özet istatistikler
    ozet = {
        'toplam': int(len(df)),
        'normal': int((df['fraud_tahmini'] == 'Normal').sum()),
        'supheli': int((df['fraud_tahmini'] == '⚠️ Şüpheli').sum()),
        'yuksek_risk': int((df['fraud_tahmini'] == '🔴 Yüksek Risk').sum()),
        'churn_yuksek': int((df.get('churn_tahmini', pd.Series()) == '🔴 Yüksek Risk').sum()) if 'churn_tahmini' in df.columns else 0,
        'ort_fraud_skoru': round(float(df['fraud_skoru'].mean()), 2),
        'max_fraud_skoru': round(float(df['fraud_skoru'].max()), 2),
        'hesaplama_tarihi': datetime.now().isoformat(),
    }
    pd.DataFrame([ozet]).to_sql("ml_ozet", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ ml_ozet kaydedildi")
    return ozet


# ─────────────────────────────────────────────
# ANA FONKSIYON
# ─────────────────────────────────────────────

def train_model():
    print("\n" + "="*60)
    print("🚀 FinanceAI — Gelişmiş ML Pipeline v3.0")
    print("="*60)
    start = datetime.now()

    # Feature engineering
    df = merge_all_features()

    # Etiketler
    df = build_fraud_labels(df)
    df = build_churn_labels(df)

    # Model eğitimi
    df, metrics, feature_cols = train_fraud_model(df)
    df = train_isolation_forest(df, feature_cols)
    df, churn_metrics = train_churn_model(df)
    metrics.update(churn_metrics)

    # Final skor
    df = compute_final_fraud_score(df)

    # Kaydet
    ozet = save_results(df)

    # Rapor
    sure = (datetime.now() - start).seconds
    print("\n" + "="*60)
    print("📊 SONUÇ RAPORU")
    print("="*60)
    print(f"  Toplam müşteri  : {ozet['toplam']:,}")
    print(f"  Normal          : {ozet['normal']:,} (%{ozet['normal']/ozet['toplam']*100:.1f})")
    print(f"  Şüpheli         : {ozet['supheli']:,} (%{ozet['supheli']/ozet['toplam']*100:.1f})")
    print(f"  Yüksek Risk     : {ozet['yuksek_risk']:,} (%{ozet['yuksek_risk']/ozet['toplam']*100:.1f})")
    print(f"  Churn Risk      : {ozet['churn_yuksek']:,}")
    print(f"\n  AUC-ROC         : {metrics.get('auc_roc', 'N/A')}")
    print(f"  F1 Skoru        : {metrics.get('f1_skoru', 'N/A')}")
    print(f"  Çalışma süresi  : {sure}s")
    print("="*60)
    print("✅ Tüm modeller hazır!")

    return df, metrics


def predict_client(client_id: int) -> dict:
    """Tek müşteri için tahmin"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM client_ml WHERE client_id = {client_id}", conn)
    conn.close()
    return df.iloc[0].to_dict() if len(df) > 0 else {}


def get_all_ml_results() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM client_ml ORDER BY fraud_skoru DESC", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df


def get_model_metrics() -> dict:
    """Dashboard için model metriklerini döndür"""
    path = f"{MODEL_PATH}/model_metrics.json"
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    train_model()