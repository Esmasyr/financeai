# -*- coding: utf-8 -*-
"""
FinanceAI — FastAPI Backend v2.1
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os, json, pickle

app = FastAPI(
    title="FinanceAI API",
    description="Finansal Risk ve Fraud Analiz Sistemi REST API",
    version="2.1.0",
    contact={"name": "FinanceAI", "email": "admin@financeai.com"},
)

# FIX 1: allow_credentials=True ile allow_origins=["*"] gecersiz kombinasyon duzeltildi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "financeai.db")
METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.json")
MODEL_PATH   = os.path.join(DATA_DIR, "xgboost_fraud.pkl")
SCALER_PATH  = os.path.join(DATA_DIR, "scaler.pkl")
FEATURE_PATH = os.path.join(DATA_DIR, "feature_cols.pkl")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> list:
    conn = get_conn()
    try:
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB hatasi: {str(e)}")
    finally:
        conn.close()


def safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if (f != f) else f
    except (TypeError, ValueError):
        return default


def safe_int(v, default=0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


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
    risk_faktorleri: dict


@app.get("/", tags=["Genel"])
def root():
    # FIX 2: Versiyon 3.0.0->2.1.0, bozuk emoji kaldirildi
    return {
        "uygulama": "FinanceAI API",
        "versiyon": "2.1.0",
        "durum": "Aktif",
        "dokumantasyon": "/docs",
        "ozellikler": ["JWT Auth", "Rate Limiting", "Cache", "WebSocket"],
        "zaman": datetime.now().isoformat()
    }


@app.get("/health", tags=["Genel"])
def health():
    try:
        conn = get_conn()
        conn.execute("SELECT 1")
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "api": "ok",
        "db": db_status,
        "ml_model": "ready" if os.path.exists(MODEL_PATH) else "not_trained",
        "metrics":  "ready" if os.path.exists(METRICS_PATH) else "missing",
        "zaman": datetime.now().isoformat()
    }


@app.get("/stats", response_model=Stats, tags=["Istatistikler"])
def get_stats():
    rows = query("""
        SELECT
            COUNT(*) as toplam_client,
            SUM(CASE WHEN risk_seviyesi LIKE '%ksek%' THEN 1 ELSE 0 END) as yuksek_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%Orta%' THEN 1 ELSE 0 END) as orta_risk,
            SUM(CASE WHEN risk_seviyesi LIKE '%k Risk%' THEN 1 ELSE 0 END) as dusuk_risk,
            COALESCE(AVG(risk_skoru), 0) as ort_risk_skoru,
            COALESCE(SUM(toplam), 0) as toplam_hacim
        FROM client_risk
    """)
    if not rows:
        raise HTTPException(status_code=500, detail="Veri yok")

    row = rows[0]
    return {
        "toplam_client":  safe_int(row.get("toplam_client")),
        "yuksek_risk":    safe_int(row.get("yuksek_risk")),
        "orta_risk":      safe_int(row.get("orta_risk")),
        "dusuk_risk":     safe_int(row.get("dusuk_risk")),
        "ort_risk_skoru": safe_float(row.get("ort_risk_skoru")),
        "toplam_hacim":   safe_float(row.get("toplam_hacim")),
        "timestamp":      datetime.now().isoformat()
    }


@app.get("/stats/fraud", response_model=FraudStats, tags=["Istatistikler"])
def get_fraud_stats():
    rows = query("""
        SELECT
            COUNT(*) as toplam,
            SUM(CASE WHEN fraud_tahmini = 'Normal' THEN 1 ELSE 0 END) as normal,
            SUM(CASE WHEN fraud_tahmini LIKE '%pheli%' THEN 1 ELSE 0 END) as supheli,
            SUM(CASE WHEN fraud_tahmini LIKE '%ksek%' THEN 1 ELSE 0 END) as yuksek_risk,
            AVG(fraud_skoru) as ort_fraud_skoru,
            MAX(fraud_skoru) as max_fraud_skoru
        FROM client_ml
    """)
    if not rows:
        raise HTTPException(status_code=404, detail="ML tablosu bulunamadi.")

    row = rows[0]
    return {
        "toplam":          safe_int(row.get("toplam")),
        "normal":          safe_int(row.get("normal")),
        "supheli":         safe_int(row.get("supheli")),
        "yuksek_risk":     safe_int(row.get("yuksek_risk")),
        "ort_fraud_skoru": safe_float(row.get("ort_fraud_skoru")),
        "max_fraud_skoru": safe_float(row.get("max_fraud_skoru")),
    }


@app.get("/stats/kategori", tags=["Istatistikler"])
def get_kategori_stats():
    rows = query("SELECT * FROM kategori_harcama ORDER BY toplam DESC")
    if not rows:
        raise HTTPException(status_code=404, detail="Kategori verisi bulunamadi.")
    return rows


@app.get("/stats/aylik", tags=["Istatistikler"])
def get_aylik_trend(son_ay: int = Query(24, ge=1, le=60)):
    conn = get_conn()
    try:
        cur = conn.execute("PRAGMA table_info(aylik_trend)")
        cols = [row[1] for row in cur.fetchall()]

        if not cols:
            raise HTTPException(status_code=404, detail="aylik_trend tablosu bulunamadi.")

        donem_col  = "donem"   if "donem"   in cols else cols[0]
        toplam_col = ("toplam"  if "toplam"  in cols else
                      "harcama" if "harcama" in cols else
                      "tutar"   if "tutar"   in cols else cols[1] if len(cols) > 1 else None)

        if not toplam_col:
            raise HTTPException(status_code=500, detail=f"Toplam kolonu bulunamadi. Mevcut: {cols}")

        rows = conn.execute(
            f"SELECT {donem_col} as donem, {toplam_col} as toplam "
            f"FROM aylik_trend ORDER BY {donem_col} DESC LIMIT ?",
            (son_ay,)
        ).fetchall()

        return sorted([dict(r) for r in rows], key=lambda x: str(x["donem"]))
    finally:
        conn.close()


@app.get("/stats/sehir", tags=["Istatistikler"])
def get_sehir_stats():
    conn = get_conn()
    try:
        for tablo in ["client_ozet", "client_risk"]:
            try:
                cur = conn.execute(f"PRAGMA table_info({tablo})")
                cols = [row[1] for row in cur.fetchall()]
                if "sehir" in cols:
                    rows = conn.execute(f"""
                        SELECT sehir, COUNT(*) as musteri_sayisi,
                               AVG(risk_skoru) as ort_risk,
                               SUM(toplam) as toplam_hacim
                        FROM {tablo}
                        WHERE sehir IS NOT NULL
                        GROUP BY sehir
                        ORDER BY musteri_sayisi DESC
                    """).fetchall()
                    return [dict(r) for r in rows]
            except Exception:
                continue
        return []
    finally:
        conn.close()


@app.get("/clients", tags=["Musteriler"])
def get_clients(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    risk_seviyesi: Optional[str] = Query(None),
    min_risk: float = Query(0, ge=0),
    max_risk: float = Query(100, le=100),
    siralama: str = Query("risk_skoru"),
):
    where = ["risk_skoru BETWEEN ? AND ?"]
    params = [min_risk, max_risk]

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


@app.get("/clients/{client_id}", tags=["Musteriler"])
def get_client(client_id: int):
    risk = query("SELECT * FROM client_risk WHERE client_id = ?", (client_id,))
    if not risk:
        raise HTTPException(status_code=404, detail=f"Musteri {client_id} bulunamadi")
    ml = query("SELECT * FROM client_ml WHERE client_id = ?", (client_id,))
    return {
        "client_id": client_id,
        "risk": risk[0],
        "ml": ml[0] if ml else None,
        "zaman": datetime.now().isoformat()
    }


@app.get("/clients/{client_id}/islemler", tags=["Musteriler"])
def get_client_islemler(client_id: int, limit: int = Query(20, ge=1, le=200)):
    rows = query("""
        SELECT donem, harcama FROM client_aylik
        WHERE client_id = ? ORDER BY donem DESC LIMIT ?
    """, (client_id, limit))
    if not rows:
        raise HTTPException(status_code=404, detail="Bu musteri icin islem bulunamadi")
    return {"client_id": client_id, "islemler": rows}


@app.get("/risk/yuksek", tags=["Risk & Fraud"])
def get_yuksek_risk(limit: int = Query(20, ge=1, le=100)):
    rows = query("""
        SELECT cr.client_id, cr.risk_skoru, cr.risk_seviyesi, cr.toplam, cr.islem,
               ml.fraud_skoru, ml.fraud_tahmini
        FROM client_risk cr
        LEFT JOIN client_ml ml ON cr.client_id = ml.client_id
        ORDER BY cr.risk_skoru DESC LIMIT ?
    """, (limit,))
    return rows


@app.get("/fraud/supheli", tags=["Risk & Fraud"])
def get_supheli(limit: int = Query(20, ge=1, le=100)):
    rows = query("""
        SELECT ml.client_id, ml.fraud_skoru, ml.fraud_tahmini,
               ml.tx_gece_oran, ml.tx_hata_oran, ml.dark_web_oran,
               cr.risk_skoru, cr.toplam
        FROM client_ml ml
        LEFT JOIN client_risk cr ON ml.client_id = cr.client_id
        WHERE ml.fraud_tahmini != 'Normal'
        ORDER BY ml.fraud_skoru DESC LIMIT ?
    """, (limit,))
    return rows


def _kural_bazli_skor(req: FraudPredictRequest):
    # FIX 3: Bozuk encoding "katk─▒lar" -> "katkilar" duzeltildi
    skor = 0.0
    faktorler = {"model": "rule_based"}
    katkilar = {
        "risk_skoru": (req.risk_skoru / 100) * 30,
        "dark_web":   req.dark_web_oran * 25,
        "gece_islem": req.tx_gece_oran * 15,
        "hata_orani": req.tx_hata_oran * 20,
        "borc_gelir": (min(req.borc_gelir_orani, 10) / 10) * 10,
    }
    for k, v in katkilar.items():
        skor += v
        faktorler[k] = round(v, 2)
    return round(min(skor, 100), 2), faktorler


@app.post("/fraud/predict", response_model=FraudPredictResponse, tags=["Risk & Fraud"])
def predict_fraud(req: FraudPredictRequest):
    skor, faktorler = _kural_bazli_skor(req)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURE_PATH):
        try:
            with open(MODEL_PATH,   "rb") as f: model     = pickle.load(f)
            with open(SCALER_PATH,  "rb") as f: scaler    = pickle.load(f)
            with open(FEATURE_PATH, "rb") as f: feat_cols = pickle.load(f)

            row = {col: 0.0 for col in feat_cols}
            row.update({
                "risk_skoru":       req.risk_skoru,
                "dark_web_oran":    req.dark_web_oran,
                "tx_gece_oran":     req.tx_gece_oran,
                "tx_hata_oran":     req.tx_hata_oran,
                "borc_gelir_orani": req.borc_gelir_orani,
                "tx_islem_sayisi":  req.tx_islem_sayisi,
            })
            X = pd.DataFrame([row])[feat_cols]
            prob = float(model.predict_proba(scaler.transform(X))[0][1])
            skor = round(prob * 100, 2)
            faktorler = {
                "model":      "XGBoost",
                "risk_skoru": req.risk_skoru,
                "dark_web":   req.dark_web_oran,
                "gece_islem": req.tx_gece_oran,
                "hata_orani": req.tx_hata_oran,
            }
        except Exception as e:
            faktorler["uyari"] = f"Model yuklenemedi: {str(e)}"

    # FIX 4: Bozuk encoding "şö┤ Yüksek Risk" -> temiz degerler
    # Dashboard LIKE '%ksek%' ve '%pheli%' ile eslestiriyor, bu degerler uyumlu
    if skor >= 60:
        tahmin = "Yuksek Risk"
    elif skor >= 30:
        tahmin = "Supheli"
    else:
        tahmin = "Normal"

    return {"fraud_skoru": skor, "fraud_tahmini": tahmin, "risk_faktorleri": faktorler}


@app.get("/model/metrics", tags=["Model"])
def get_model_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model henuz egitilmedi. ml_model.py calistirin.")


@app.get("/model/ozet", tags=["Model"])
def get_ml_ozet():
    rows = query("SELECT * FROM ml_ozet ORDER BY rowid DESC LIMIT 1")
    if not rows:
        raise HTTPException(status_code=404, detail="ML ozet yok.")
    return rows[0]


@app.get("/churn/yuksek", tags=["Churn"])
def get_churn_yuksek(limit: int = Query(20, ge=1, le=100)):
    rows = query("""
        SELECT ml.client_id, ml.churn_skoru, ml.churn_tahmini,
               ml.fraud_skoru, cr.toplam, cr.islem
        FROM client_ml ml
        LEFT JOIN client_risk cr ON ml.client_id = cr.client_id
        WHERE ml.churn_tahmini LIKE '%ksek%'
        ORDER BY ml.churn_skoru DESC LIMIT ?
    """, (limit,))
    return rows


@app.get("/churn/stats", tags=["Churn"])
def get_churn_stats():
    rows = query("""
        SELECT
            COUNT(*) as toplam,
            SUM(CASE WHEN churn_tahmini LIKE '%ksek%' THEN 1 ELSE 0 END) as yuksek,
            SUM(CASE WHEN churn_tahmini LIKE '%Orta%' THEN 1 ELSE 0 END) as orta,
            SUM(CASE WHEN churn_tahmini LIKE '%k Risk%' THEN 1 ELSE 0 END) as dusuk,
            AVG(churn_skoru) as ort_churn_skoru
        FROM client_ml WHERE churn_tahmini IS NOT NULL
    """)
    return rows[0] if rows else {}


@app.get("/search", tags=["Arama"])
def search_client(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    rows = query("""
        SELECT cr.client_id, cr.risk_skoru, cr.risk_seviyesi, cr.toplam,
               ml.fraud_skoru, ml.fraud_tahmini
        FROM client_risk cr
        LEFT JOIN client_ml ml ON cr.client_id = ml.client_id
        WHERE CAST(cr.client_id AS TEXT) LIKE ?
        LIMIT ?
    """, (f"%{q}%", limit))
    return {"sorgu": q, "sonuclar": rows}


if __name__ == "__main__":
    import uvicorn
    print("FinanceAI API v2.1 baslatiliyor...")
    print(f"   DB   : {DB_PATH}")
    print(f"   Docs : http://localhost:8000/docs")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"]
    )