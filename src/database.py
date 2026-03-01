import sqlite3
import pandas as pd

DB_PATH = "C:/financeai/data/financeai.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    
    print("Client risk tablosu aktarılıyor...")
    client_risk = pd.read_csv("C:/financeai/data/client_risk.csv")
    client_risk.to_sql("client_risk", conn, if_exists="replace", index=False)
    print(f"✅ {len(client_risk)} client aktarıldı")
    
    print("Client özet tablosu aktarılıyor...")
    client_ozet = pd.read_csv("C:/financeai/data/client_ozet.csv")
    client_ozet.to_sql("client_ozet", conn, if_exists="replace", index=False)
    print(f"✅ {len(client_ozet)} özet aktarıldı")
    
    conn.close()
    print("✅ Veritabanı hazır!")

def get_all_clients():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM client_risk ORDER BY risk_skoru DESC", conn)
    conn.close()
    return df

def get_client_risk(client_id):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM client_risk WHERE client_id = {client_id}", conn)
    conn.close()
    return df

def get_stats():
    conn = get_connection()
    stats = pd.read_sql("""
        SELECT 
            COUNT(*) as toplam_client,
            SUM(CASE WHEN risk_seviyesi LIKE '%Yüksek%' THEN 1 ELSE 0 END) as yuksek_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%Orta%' THEN 1 ELSE 0 END) as orta_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%Düşük%' THEN 1 ELSE 0 END) as dusuk_risk,
            AVG(risk_skoru) as ort_risk_skoru,
            SUM(toplam) as toplam_hacim
        FROM client_risk
    """, conn)
    conn.close()
    return stats.iloc[0]

init_db()