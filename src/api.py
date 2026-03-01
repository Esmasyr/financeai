"""
FinanceAI — FastAPI Backend
============================
Çalıştırmak için:
    pip install fastapi uvicorn pandas sqlite3
    uvicorn api:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# UYGULAMA
# ─────────────────────────────────────────────

app = FastAPI(
    title="FinanceAI API",
    description="Finansal Risk ve Fraud Analiz Sistemi — REST API",
    version="2.0.0",
    contact={"name": "FinanceAI", "email": "admin@financeai.com"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da Streamlit URL'ini yaz
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "C:/financeai/data/financeai.db"


# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB hatası: {str(e)}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# RESPONSE MODELLERİ
# ─────────────────────────────────────────────

class ClientRisk(BaseModel):
    client_id: int
    risk_skoru: float
    risk_seviyesi: str
    toplam: float
    islem: int

class MLResult(BaseModel):
    client_id: int
    fraud_skoru: float
    fraud_tahmini: str
    tx_gece_oran: Optional[float] = None
    tx_hata_oran: Optional[float] = None
    dark_web_oran: Optional[float] = None
    kredi_skoru: Optional[float] = None
    borc_gelir_orani: Optional[float] = None

class Stats(BaseModel):
    toplam_client: int
    yuksek_risk: int
    orta_risk: int
    dusuk_risk: int
    ort_risk_skoru: float
    toplam_hacim: float
    timestamp: str

class FraudStats(BaseModel):
    toplam: int
    normal: int
    supheli: int
    yuksek_risk: int
    ort_fraud_skoru: float
    max_fraud_skoru: float

class FraudPredictRequest(BaseModel):
    risk_skoru: float
    dark_web_oran: float = 0.0
    tx_gece_oran: float = 0.0
    tx_hata_oran: float = 0.0
    borc_gelir_orani: float = 0.0
    tx_islem_sayisi: int = 0

class FraudPredictResponse(BaseModel):
    fraud_skoru: float
    fraud_tahmini: str
    risk_faktörleri: dict


# ─────────────────────────────────────────────
# GENEL
# ─────────────────────────────────────────────

@app.get("/", tags=["Genel"])
def root():
    return {
        "uygulama": "FinanceAI API",
        "versiyon": "2.0.0",
        "durum": "✅ Aktif",
        "dokümantasyon": "/docs",
        "zaman": datetime.now().isoformat()
    }


@app.get("/health", tags=["Genel"])
def health():
    """Sistem sağlık kontrolü"""
    try:
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()
        db_status = "✅ Bağlı"
    except Exception as e:
        db_status = f"❌ Hata: {str(e)}"

    return {
        "api": "✅ Çalışıyor",
        "veritabanı": db_status,
        "zaman": datetime.now().isoformat()
    }


# ─────────────────────────────────────────────
# İSTATİSTİKLER
# ─────────────────────────────────────────────

@app.get("/stats", response_model=Stats, tags=["İstatistikler"])
def get_stats():
    """Genel sistem istatistikleri — Dashboard KPI kartları için"""
    rows = query("""
        SELECT
            COUNT(*) as toplam_client,
            SUM(CASE WHEN risk_seviyesi LIKE '%Yüksek%' THEN 1 ELSE 0 END) as yuksek_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%Orta%' THEN 1 ELSE 0 END) as orta_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%Düşük%' THEN 1 ELSE 0 END) as dusuk_risk,
            COALESCE(AVG(risk_skoru), 0) as ort_risk_skoru,
            COALESCE(SUM(toplam), 0) as toplam_hacim
        FROM client_risk
    """)
    if not rows:
        raise HTTPException(status_code=500, detail="Veri yok")
    row = rows[0]
    # None değerleri 0 yap
    row = {k: (v if v is not None else 0) for k, v in row.items()}
    return {**row, "timestamp": datetime.now().isoformat()}


@app.get("/stats/fraud", response_model=FraudStats, tags=["İstatistikler"])
def get_fraud_stats():
    """ML fraud tahmin istatistikleri"""
    rows = query("""
        SELECT
            COUNT(*) as toplam,
            SUM(CASE WHEN fraud_tahmini = 'Normal' THEN 1 ELSE 0 END) as normal,
            SUM(CASE WHEN fraud_tahmini LIKE '%Şüpheli%' THEN 1 ELSE 0 END) as supheli,
            SUM(CASE WHEN fraud_tahmini LIKE '%Yüksek%' THEN 1 ELSE 0 END) as yuksek_risk,
            AVG(fraud_skoru) as ort_fraud_skoru,
            MAX(fraud_skoru) as max_fraud_skoru
        FROM client_ml
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="ML tablosu bulunamadı. Önce ml_model.py çalıştırın.")
    return rows[0]


@app.get("/stats/kategori", tags=["İstatistikler"])
def get_kategori_stats():
    """Harcama kategorileri dağılımı"""
    rows = query("SELECT * FROM kategori_harcama ORDER BY toplam DESC")
    if not rows:
        raise HTTPException(status_code=404, detail="Kategori verisi bulunamadı.")
    return rows


@app.get("/stats/aylik", tags=["İstatistikler"])
def get_aylik_trend(son_ay: int = Query(24, ge=1, le=60, description="Son kaç ay")):
    """Aylık işlem hacmi trendi"""
    rows = query("SELECT * FROM aylik_trend ORDER BY donem DESC LIMIT ?", (son_ay,))
    return sorted(rows, key=lambda x: x["donem"])


@app.get("/stats/sehir", tags=["İstatistikler"])
def get_sehir_stats():
    """Şehir bazlı özet"""
    # sehir kolonu varsa grupla, yoksa boş döndür
    conn = get_conn()
    try:
        cur = conn.execute("PRAGMA table_info(client_risk)")
        cols = [row[1] for row in cur.fetchall()]
        if "sehir" in cols:
            rows = query("""
                SELECT sehir, COUNT(*) as musteri_sayisi,
                       AVG(risk_skoru) as ort_risk,
                       SUM(toplam) as toplam_hacim
                FROM client_risk
                GROUP BY sehir
                ORDER BY musteri_sayisi DESC
            """)
            return rows
        else:
            # sehir kolonu yok, segment veya risk bazlı döndür
            rows = query("""
                SELECT risk_seviyesi as sehir,
                       COUNT(*) as musteri_sayisi,
                       AVG(risk_skoru) as ort_risk,
                       SUM(toplam) as toplam_hacim
                FROM client_risk
                GROUP BY risk_seviyesi
                ORDER BY musteri_sayisi DESC
            """)
            return rows
    except Exception as e:
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────
# MÜŞTERİLER
# ─────────────────────────────────────────────

@app.get("/clients", tags=["Müşteriler"])
def get_clients(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_seviyesi: Optional[str] = Query(None, description="Düşük Risk | Orta Risk | Yüksek Risk"),
    min_risk: float = Query(0, ge=0),
    max_risk: float = Query(65, le=65),
    siralama: str = Query("risk_skoru", description="risk_skoru | toplam | islem"),
):
    """
    Müşteri listesi — sayfalama ve filtreleme destekli
    """
    where = ["risk_skoru BETWEEN ? AND ?"]
    params: list = [min_risk, max_risk]

    if risk_seviyesi:
        where.append("risk_seviyesi LIKE ?")
        params.append(f"%{risk_seviyesi}%")

    safe_sort = {"risk_skoru", "toplam", "islem"}
    sort_col = siralama if siralama in safe_sort else "risk_skoru"

    sql = f"""
        SELECT * FROM client_risk
        WHERE {' AND '.join(where)}
        ORDER BY {sort_col} DESC
        LIMIT ? OFFSET ?
    """
    params += [limit, offset]
    rows = query(sql, tuple(params))
    return {"toplam_sonuc": len(rows), "offset": offset, "limit": limit, "data": rows}


@app.get("/clients/{client_id}", tags=["Müşteriler"])
def get_client(client_id: int):
    """Tek müşteri detayı — risk + ML sonucu birleşik"""
    risk = query("SELECT * FROM client_risk WHERE client_id = ?", (client_id,))
    if not risk:
        raise HTTPException(status_code=404, detail=f"Müşteri {client_id} bulunamadı")

    ml = query("SELECT * FROM client_ml WHERE client_id = ?", (client_id,))

    return {
        "client_id": client_id,
        "risk": risk[0],
        "ml": ml[0] if ml else None,
        "zaman": datetime.now().isoformat()
    }


@app.get("/clients/{client_id}/islemler", tags=["Müşteriler"])
def get_client_islemler(
    client_id: int,
    limit: int = Query(20, ge=1, le=200),
):
    """Müşterinin son işlemleri"""
    rows = query("""
        SELECT donem, harcama FROM client_aylik
        WHERE client_id = ?
        ORDER BY donem DESC
        LIMIT ?
    """, (client_id, limit))
    if not rows:
        raise HTTPException(status_code=404, detail="Bu müşteri için işlem bulunamadı")
    return {"client_id": client_id, "islemler": rows}


# ─────────────────────────────────────────────
# RİSK & FRAUD
# ─────────────────────────────────────────────

@app.get("/risk/yuksek", tags=["Risk & Fraud"])
def get_yuksek_risk(limit: int = Query(20, ge=1, le=100)):
    """En yüksek riskli müşteriler"""
    rows = query("""
        SELECT cr.client_id, cr.risk_skoru, cr.risk_seviyesi, cr.toplam, cr.islem,
               ml.fraud_skoru, ml.fraud_tahmini
        FROM client_risk cr
        LEFT JOIN client_ml ml ON cr.client_id = ml.client_id
        ORDER BY cr.risk_skoru DESC
        LIMIT ?
    """, (limit,))
    return rows


@app.get("/fraud/supheli", tags=["Risk & Fraud"])
def get_supheli(limit: int = Query(20, ge=1, le=100)):
    """Şüpheli olarak işaretlenen müşteriler"""
    rows = query("""
        SELECT ml.client_id, ml.fraud_skoru, ml.fraud_tahmini,
               ml.tx_gece_oran, ml.tx_hata_oran, ml.dark_web_oran,
               cr.risk_skoru, cr.toplam
        FROM client_ml ml
        LEFT JOIN client_risk cr ON ml.client_id = cr.client_id
        WHERE ml.fraud_tahmini != 'Normal'
        ORDER BY ml.fraud_skoru DESC
        LIMIT ?
    """, (limit,))
    return rows


@app.post("/fraud/predict", response_model=FraudPredictResponse, tags=["Risk & Fraud"])
def predict_fraud(req: FraudPredictRequest):
    """
    Anlık fraud skoru hesapla
    Yeni müşteri veya güncellenen veri için gerçek zamanlı tahmin
    """
    skor = 0.0
    faktorler = {}

    # Risk skoru katkısı (max 30 puan)
    risk_katki = (req.risk_skoru / 65) * 30
    skor += risk_katki
    faktorler["risk_skoru"] = round(risk_katki, 2)

    # Dark web katkısı (max 25 puan)
    dw_katki = req.dark_web_oran * 25
    skor += dw_katki
    faktorler["dark_web"] = round(dw_katki, 2)

    # Gece işlem katkısı (max 15 puan)
    gece_katki = req.tx_gece_oran * 15
    skor += gece_katki
    faktorler["gece_islem"] = round(gece_katki, 2)

    # Hata oranı katkısı (max 20 puan)
    hata_katki = req.tx_hata_oran * 20
    skor += hata_katki
    faktorler["hata_orani"] = round(hata_katki, 2)

    # Borç/gelir katkısı (max 10 puan)
    bg_katki = (min(req.borc_gelir_orani, 10) / 10) * 10
    skor += bg_katki
    faktorler["borc_gelir"] = round(bg_katki, 2)

    skor = round(min(skor, 100), 2)

    if skor < 30:
        tahmin = "Normal"
    elif skor < 60:
        tahmin = "⚠️ Şüpheli"
    else:
        tahmin = "🔴 Yüksek Risk"

    return {
        "fraud_skoru": skor,
        "fraud_tahmini": tahmin,
        "risk_faktörleri": faktorler
    }


# ─────────────────────────────────────────────
# MODEL METRİKLERİ & CHURN
# ─────────────────────────────────────────────

@app.get("/model/metrics", tags=["Model"])
def get_model_metrics():
    """Gerçek model metrikleri (AUC, F1, Precision, Recall)"""
    import os, json
    path = "C:/financeai/data/model_metrics.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model henüz eğitilmedi. ml_model.py çalıştırın.")

@app.get("/model/ozet", tags=["Model"])
def get_ml_ozet():
    """ML özet istatistikleri"""
    rows = query("SELECT * FROM ml_ozet ORDER BY rowid DESC LIMIT 1")
    if not rows:
        raise HTTPException(status_code=404, detail="ML özet yok.")
    return rows[0]

@app.get("/churn/yuksek", tags=["Churn"])
def get_churn_yuksek(limit: int = Query(20, ge=1, le=100)):
    """Churn riski yüksek müşteriler"""
    rows = query("""
        SELECT ml.client_id, ml.churn_skoru, ml.churn_tahmini,
               ml.fraud_skoru, cr.toplam, cr.islem
        FROM client_ml ml
        LEFT JOIN client_risk cr ON ml.client_id = cr.client_id
        WHERE ml.churn_tahmini LIKE '%Yüksek%'
        ORDER BY ml.churn_skoru DESC
        LIMIT ?
    """, (limit,))
    return rows

@app.get("/churn/stats", tags=["Churn"])
def get_churn_stats():
    """Churn istatistikleri"""
    rows = query("""
        SELECT
            COUNT(*) as toplam,
            SUM(CASE WHEN churn_tahmini LIKE '%Yüksek%' THEN 1 ELSE 0 END) as yuksek,
            SUM(CASE WHEN churn_tahmini LIKE '%Orta%' THEN 1 ELSE 0 END) as orta,
            SUM(CASE WHEN churn_tahmini = 'Düşük Risk' THEN 1 ELSE 0 END) as dusuk,
            AVG(churn_skoru) as ort_churn_skoru
        FROM client_ml
        WHERE churn_tahmini IS NOT NULL
    """)
    return rows[0] if rows else {}

# ─────────────────────────────────────────────
# ARAMA
# ─────────────────────────────────────────────

@app.get("/search", tags=["Arama"])
def search_client(
    q: str = Query(..., min_length=1, description="Client ID (tam veya kısmi)"),
    limit: int = Query(10, ge=1, le=50)
):
    """Müşteri ID ile arama"""
    rows = query("""
        SELECT cr.client_id, cr.risk_skoru, cr.risk_seviyesi, cr.toplam,
               ml.fraud_skoru, ml.fraud_tahmini
        FROM client_risk cr
        LEFT JOIN client_ml ml ON cr.client_id = ml.client_id
        WHERE CAST(cr.client_id AS TEXT) LIKE ?
        LIMIT ?
    """, (f"%{q}%", limit))
    return {"sorgu": q, "sonuclar": rows}