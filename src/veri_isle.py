import pandas as pd
import numpy as np
import sqlite3
import json
import pyarrow.parquet as pq

DB_PATH = "C:/financeai/data/financeai.db"
PARQUET_PATH = "C:/financeai/data/transactions_clean.parquet"

print("🚀 Veri işleme başlıyor...")

# MCC kodları
with open("C:/financeai/data/mcc_codes.json", "r") as f:
    mcc_dict = json.load(f)

# Özet sözlükleri
aylik_dict = {}
kategori_dict = {}
saat_dict = {}
gun_dict = {}
client_aylik = {}

parquet_file = pq.ParquetFile(PARQUET_PATH)
toplam_parca = 0

for batch in parquet_file.iter_batches(batch_size=500_000):
    chunk = batch.to_pandas()
    toplam_parca += 1
    print(f"  Parça {toplam_parca} işleniyor ({toplam_parca * 500_000:,} satır)...")

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
    gun_isim = {0:'Pzt',1:'Sal',2:'Çar',3:'Per',4:'Cum',5:'Cmt',6:'Paz'}
    for gun, grp in chunk.groupby('gun'):
        isim = gun_isim[gun]
        gun_dict[isim] = gun_dict.get(isim, 0) + len(grp)

    # 5. Client aylık (sadece gider)
    for (cid, y, m), grp in gider.groupby(['client_id', 'year', 'month']):
        key = (int(cid), f"{y}-{str(m).zfill(2)}")
        if key not in client_aylik:
            client_aylik[key] = 0
        client_aylik[key] += grp['abs_amount'].sum()

print("✅ Tüm parçalar işlendi, veritabanına yazılıyor...")

conn = sqlite3.connect(DB_PATH)

# Aylık trend tablosu
aylik_df = pd.DataFrame(
    list(aylik_dict.items()), columns=['donem', 'toplam']
).sort_values('donem')
aylik_df.to_sql("aylik_trend", conn, if_exists="replace", index=False)
print(f"✅ Aylık trend: {len(aylik_df)} dönem")

# Kategori tablosu
kat_df = pd.DataFrame(
    list(kategori_dict.items()), columns=['kategori', 'toplam']
).sort_values('toplam', ascending=False)
kat_df.to_sql("kategori_harcama", conn, if_exists="replace", index=False)
print(f"✅ Kategori: {len(kat_df)} kategori")

# Saat tablosu
saat_df = pd.DataFrame(
    list(saat_dict.items()), columns=['saat', 'islem_sayisi']
).sort_values('saat')
saat_df.to_sql("saat_dagilim", conn, if_exists="replace", index=False)
print(f"✅ Saat dağılımı: {len(saat_df)} saat")

# Gün tablosu
gun_sira = ['Pzt','Sal','Çar','Per','Cum','Cmt','Paz']
gun_df = pd.DataFrame(
    list(gun_dict.items()), columns=['gun', 'islem_sayisi']
)
gun_df['sira'] = gun_df['gun'].map({g:i for i,g in enumerate(gun_sira)})
gun_df = gun_df.sort_values('sira').drop('sira', axis=1)
gun_df.to_sql("gun_dagilim", conn, if_exists="replace", index=False)
print(f"✅ Gün dağılımı kaydedildi")

# Client aylık tablosu
client_aylik_df = pd.DataFrame(
    [{'client_id': k[0], 'donem': k[1], 'harcama': v}
     for k, v in client_aylik.items()]
).sort_values(['client_id', 'donem'])
client_aylik_df.to_sql("client_aylik", conn, if_exists="replace", index=False)
print(f"✅ Client aylık: {len(client_aylik_df)} kayıt")

conn.close()
print("\n🎉 Tüm veriler veritabanına aktarıldı!")