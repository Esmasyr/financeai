"""
FinanceAI — Veri İşleme 
=================================================
  - cards_data.csv client_id üzerinden birleştirildi
  - card_on_dark_web → dark_web_target (0/1) hedef değişken
  - credit_limit temizlendi ($ işareti kaldırıldı)
  - avg_transaction, islem_basina_hata feature'ları eklendi
  - client_risk tablosu zenginleştirildi (ML'e hazır)
  - client_ozet tablosu güncellendi

Çalıştır: python src/veri_isle.py
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import pyarrow.parquet as pq

DB_PATH      = "C:/financeai/data/financeai.db"
PARQUET_PATH = "C:/financeai/data/transactions_clean.parquet"
DATA_PATH    = "C:/financeai/data"

print("🚀 Veri işleme başlıyor...")

# ──────────────────────────────────────────────────────────
# 1. KART VERİSİ — cards_data.csv
#  credit_limit temizle, dark_web_target üret
# ──────────────────────────────────────────────────────────
print("\n📇 Kart verisi yükleniyor...")
cards = pd.read_csv(f"{DATA_PATH}/cards_data.csv")

# credit_limit temizle: "$1,000.00" → 1000.0
cards['credit_limit'] = (
    cards['credit_limit']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .astype(float)
)

# card_on_dark_web → bu dataset'te tüm değerler 'No'
# Gerçek bir hedef değişken yok, 0 olarak bırakılıyor
# ML hedef değişkeni olarak fraud_label (train_fraud_labels.json) kullanılacak
cards['dark_web_target'] = 0  # bu dataset'te tum degerler 'No', placeholder

# Müşteri bazında kart özeti
card_ozet = cards.groupby('client_id').agg(
    kart_adedi        =('id', 'count'),
    toplam_limit      =('credit_limit', 'sum'),
    ort_limit         =('credit_limit', 'mean'),
    max_limit         =('credit_limit', 'max'),
    dark_web_target   =('dark_web_target', 'max'),   # herhangi bir kartı dark web'deyse 1
    dark_web_kart_say =('dark_web_target', 'sum'),   # kaç kartı dark web'de
    dark_web_oran     =('dark_web_target', 'mean'),  # oranı
    chip_oran         =('has_chip', lambda x: (x == 'YES').mean()),
).reset_index()

# Fraud etiketlerini yükle — gerçek ML hedef değişkeni
import os
fraud_client_ids = set()
label_path = f"{DATA_PATH}/train_fraud_labels.json"
if os.path.exists(label_path):
    with open(label_path, encoding='utf-8') as f_lbl:
        raw = json.load(f_lbl)
    labels = raw["target"] if "target" in raw else raw
    fraud_client_ids = {int(k) for k, v in labels.items()
                        if str(v).strip().lower() in ['yes','1','true']}

print(f"✅ Kart verisi: {len(cards):,} kart, {len(card_ozet):,} müşteri")
print(f"   Dark web'de kart olan müşteri: {card_ozet['dark_web_target'].sum():,}")
print(f"   Ortalama kredi limiti: ${card_ozet['ort_limit'].mean():,.0f}")

# ──────────────────────────────────────────────────────────
# 2. KULLANICI VERİSİ — users_data.csv
# ──────────────────────────────────────────────────────────
print("\n👤 Kullanıcı verisi yükleniyor...")
users = pd.read_csv(f"{DATA_PATH}/users_data.csv")

# Parasal sütunları temizle
for col in ['per_capita_income', 'yearly_income', 'total_debt']:
    if col in users.columns:
        users[col] = (
            users[col].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .astype(float)
        )

users = users.rename(columns={'id': 'client_id'})
ucols = ['client_id', 'current_age', 'credit_score',
         'per_capita_income', 'yearly_income', 'total_debt']
ucols = [c for c in ucols if c in users.columns]
users = users[ucols].copy()

# Borç/gelir oranı
if 'total_debt' in users.columns and 'yearly_income' in users.columns:
    users['borc_gelir_orani'] = users['total_debt'] / (users['yearly_income'] + 1)

print(f"✅ Kullanıcı verisi: {len(users):,} müşteri")

# ──────────────────────────────────────────────────────────
# 3. MCC KODLARI
# ──────────────────────────────────────────────────────────
with open(f"{DATA_PATH}/mcc_codes.json", "r") as f:
    mcc_dict = json.load(f)

# ──────────────────────────────────────────────────────────
# 4. İŞLEM VERİSİ — Parça parça işle
# ──────────────────────────────────────────────────────────

# Özet sözlükleri
aylik_dict    = {}
kategori_dict = {}
saat_dict     = {}
gun_dict      = {}
client_aylik  = {}

# Müşteri bazlı metrikler (risk skoru için)
#  avg_transaction, islem_basina_hata
client_metrics = {}  # client_id → {toplam, islem, hata, ...}

parquet_file = pq.ParquetFile(PARQUET_PATH)
toplam_parca = 0

for batch in parquet_file.iter_batches(batch_size=500_000):
    chunk = batch.to_pandas()
    toplam_parca += 1
    print(f"  Parça {toplam_parca} işleniyor ({toplam_parca * 500_000:,} satır)...")

    # Amount temizle
    if 'amount' in chunk.columns:
        chunk['abs_amount'] = pd.to_numeric(
            chunk['amount'].astype(str).str.replace('$', '', regex=False),
            errors='coerce'
        ).abs().fillna(0)

    chunk['errors'] = chunk.get('errors', pd.Series(['No Error'] * len(chunk))).fillna('No Error')
    chunk['hata_var'] = (chunk['errors'] != 'No Error').astype(int)

    gider = chunk[chunk['transaction_type'] == 'Gider']

    # 1. Aylık trend
    for (y, m), grp in gider.groupby(['year', 'month']):
        key = f"{y}-{str(m).zfill(2)}"
        aylik_dict[key] = aylik_dict.get(key, 0) + grp['abs_amount'].sum()

    # 2. Kategori
    for kat, grp in gider.groupby('kategori'):
        kategori_dict[kat] = kategori_dict.get(kat, 0) + grp['abs_amount'].sum()

    # 3. Saat bazlı
    for saat, grp in chunk.groupby('hour'):
        saat_dict[saat] = saat_dict.get(saat, 0) + len(grp)

    # 4. Gün bazlı
    chunk['gun'] = pd.to_datetime(chunk['date']).dt.dayofweek
    gun_isim = {0: 'Pzt', 1: 'Sal', 2: 'Çar', 3: 'Per', 4: 'Cum', 5: 'Cmt', 6: 'Paz'}
    for gun, grp in chunk.groupby('gun'):
        isim = gun_isim[gun]
        gun_dict[isim] = gun_dict.get(isim, 0) + len(grp)

    # 5. Client aylık (sadece gider)
    for (cid, y, m), grp in gider.groupby(['client_id', 'year', 'month']):
        key = (int(cid), f"{y}-{str(m).zfill(2)}")
        client_aylik[key] = client_aylik.get(key, 0) + grp['abs_amount'].sum()

    # 6. Müşteri bazlı metrikler (YENİ)
    for cid, grp in chunk.groupby('client_id'):
        cid = int(cid)
        if cid not in client_metrics:
            client_metrics[cid] = {
                'toplam':  0.0,
                'islem':   0,
                'hata':    0,
                'gece':    0,    # 22:00-06:00 arası işlem
                'online':  0,    # online işlem
            }
        m = client_metrics[cid]
        m['toplam'] += grp['abs_amount'].sum()
        m['islem']  += len(grp)
        m['hata']   += grp['hata_var'].sum()

        # Gece işlemi
        if 'hour' in grp.columns:
            m['gece'] += ((grp['hour'] >= 22) | (grp['hour'] <= 6)).sum()

        # Online işlem
        if 'use_chip' in grp.columns:
            m['online'] += (grp['use_chip'] == 'Online Transaction').sum()

print("✅ Tüm parçalar işlendi, veritabanına yazılıyor...")

# ──────────────────────────────────────────────────────────
# 5. MÜŞTERİ RİSK TABLOSU — ZENGİNLEŞTİRİLMİŞ
#  avg_transaction, islem_basina_hata, kart verisiyle birleştir
# ──────────────────────────────────────────────────────────
print("\n📊 Müşteri risk tablosu oluşturuluyor...")

risk_df = pd.DataFrame([
    {
        'client_id': cid,
        'toplam':    m['toplam'],
        'islem':     m['islem'],
        'hata':      m['hata'],
        'gece_islem':m['gece'],
        'online_islem': m['online'],
    }
    for cid, m in client_metrics.items()
])

#  avg_transaction ve islem_basina_hata feature'ları
risk_df['avg_transaction']   = risk_df['toplam'] / (risk_df['islem'] + 1)
risk_df['islem_basina_hata'] = risk_df['hata']   / (risk_df['islem'] + 1)
risk_df['gece_oran']         = risk_df['gece_islem']  / (risk_df['islem'] + 1)
risk_df['online_oran']       = risk_df['online_islem'] / (risk_df['islem'] + 1)

# kart verisiyle client_id üzerinden birleştir
risk_df = risk_df.merge(card_ozet, on='client_id', how='left')
risk_df = risk_df.merge(users, on='client_id', how='left')
risk_df = risk_df.fillna(0)

# Z-score bazlı skor hesapla
def safe_zscore(s):
    std = s.std()
    if std < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

risk_df['harcama_zscore'] = safe_zscore(risk_df['toplam'])
risk_df['hata_orani']     = risk_df['islem_basina_hata']
risk_df['islem_zscore']   = safe_zscore(risk_df['islem'])

# Risk skoru — dinamik eşik 
# Kural bazlı skor: 3 bileşen, ağırlıklı
risk_df['skor_harcama'] = risk_df['harcama_zscore'].clip(0, 3) * 0.15
risk_df['skor_hata']    = risk_df['hata_orani'].clip(0, 1)     * 0.55
risk_df['skor_islem']   = risk_df['islem_zscore'].clip(0, 3)   * 0.15
risk_df['skor_darkweb'] = risk_df['dark_web_oran']             * 0.15  # YENİ

# Fraud müşteri hedefi — train_fraud_labels.json'daki fraud işlem ID'lerine göre
# NOT: fraud_ids işlem ID'si, ama biz client_id'si bilmiyoruz direkt
# Bu yüzden ml_model.py'deki işlem bazlı fraud_label kullanılacak
# Burada sadece flag bırakıyoruz
risk_df['fraud_musteri_target'] = 0  # ml_model.py dolduracak

risk_df['risk_skoru'] = (
    (risk_df['skor_harcama'] +
     risk_df['skor_hata']    +
     risk_df['skor_islem']   +
     risk_df['skor_darkweb']) * 100
).clip(0, 100).round(2)

# Dinamik eşik — herkes düşük çıkmasın diye
esik_yuksek  = risk_df['risk_skoru'].quantile(0.90)
esik_orta    = risk_df['risk_skoru'].quantile(0.70)

risk_df['risk_seviyesi'] = pd.cut(
    risk_df['risk_skoru'],
    bins=[-np.inf, esik_orta, esik_yuksek, np.inf],
    labels=['Dusuk Risk', 'Orta Risk', 'Yuksek Risk']
).astype(str)

print(f"  Dinamik eşikler — Orta: {esik_orta:.1f}  Yüksek: {esik_yuksek:.1f}")
print(f"  Dağılım: {risk_df['risk_seviyesi'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────
# 6. CLIENT_OZET TABLOSU — ML için hazır
# ──────────────────────────────────────────────────────────
ozet_cols = [
    'client_id', 'toplam', 'islem', 'hata',
    'avg_transaction', 'islem_basina_hata',
    'gece_oran', 'online_oran',
    # Kart verileri
    'kart_adedi', 'toplam_limit', 'ort_limit', 'dark_web_oran',
    'dark_web_target',   # ← ML hedef değişkeni
    'dark_web_kart_say', 'chip_oran',
    # Kullanıcı verileri
    'credit_score', 'yearly_income', 'total_debt', 'borc_gelir_orani',
    # Risk skoru
    'risk_skoru', 'risk_seviyesi',
    'harcama_zscore', 'hata_orani', 'islem_zscore',
]
ozet_cols = [c for c in ozet_cols if c in risk_df.columns]
ozet_df   = risk_df[ozet_cols].copy()

# ──────────────────────────────────────────────────────────
# 7. VERİTABANINA YAZ
# ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

# Aylık trend
aylik_df = pd.DataFrame(
    list(aylik_dict.items()), columns=['donem', 'toplam']
).sort_values('donem')
aylik_df.to_sql("aylik_trend", conn, if_exists="replace", index=False)
print(f"✅ Aylık trend: {len(aylik_df)} dönem")

# Kategori
kat_df = pd.DataFrame(
    list(kategori_dict.items()), columns=['kategori', 'toplam']
).sort_values('toplam', ascending=False)
kat_df.to_sql("kategori_harcama", conn, if_exists="replace", index=False)
print(f"✅ Kategori: {len(kat_df)} kategori")

# Saat
saat_df = pd.DataFrame(
    list(saat_dict.items()), columns=['saat', 'islem_sayisi']
).sort_values('saat')
saat_df.to_sql("saat_dagilim", conn, if_exists="replace", index=False)
print(f"✅ Saat dağılımı: {len(saat_df)} saat")

# Gün
gun_sira = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz']
gun_df = pd.DataFrame(
    list(gun_dict.items()), columns=['gun', 'islem_sayisi']
)
gun_df['sira'] = gun_df['gun'].map({g: i for i, g in enumerate(gun_sira)})
gun_df = gun_df.sort_values('sira').drop('sira', axis=1)
gun_df.to_sql("gun_dagilim", conn, if_exists="replace", index=False)
print(f"✅ Gün dağılımı kaydedildi")

# Client aylık
client_aylik_df = pd.DataFrame(
    [{'client_id': k[0], 'donem': k[1], 'harcama': v}
     for k, v in client_aylik.items()]
).sort_values(['client_id', 'donem'])
client_aylik_df.to_sql("client_aylik", conn, if_exists="replace", index=False)
print(f"✅ Client aylık: {len(client_aylik_df)} kayıt")

# Client ozet (YENİ — zenginleştirilmiş)
ozet_df.to_sql("client_ozet", conn, if_exists="replace", index=False)
print(f"✅ Client özet: {len(ozet_df)} müşteri, {len(ozet_df.columns)} sütun")

# Client risk (YENİ — dinamik eşikli)
risk_cols = ['client_id', 'toplam', 'islem', 'hata',
             'harcama_zscore', 'hata_orani', 'islem_zscore',
             'skor_harcama', 'skor_hata', 'skor_islem', 'skor_darkweb',
             'risk_skoru', 'risk_seviyesi',
             'avg_transaction', 'islem_basina_hata',
             'dark_web_target', 'dark_web_oran', 'kart_adedi']
risk_cols = [c for c in risk_cols if c in risk_df.columns]
risk_df[risk_cols].to_sql("client_risk", conn, if_exists="replace", index=False)
print(f"✅ Client risk: {len(risk_df)} müşteri")
print(f"   Risk dağılımı: {risk_df['risk_seviyesi'].value_counts().to_dict()}")

conn.close()
print("\n🎉 Tüm veriler veritabanına aktarıldı!")
print(f"\n📊 Özet:")
print(f"   Toplam müşteri : {len(ozet_df):,}")
print(f"   Dark web hedef : {ozet_df['dark_web_target'].sum():,} müşteri ({ozet_df['dark_web_target'].mean()*100:.1f}%)")
print(f"   Ort. işlem/müş : {ozet_df['islem'].mean():.0f}")
print(f"   Ort. hata oranı: %{ozet_df['islem_basina_hata'].mean()*100:.2f}")