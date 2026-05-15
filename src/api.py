# FinanceAI API v2.1 — Düzeltilmiş & Geliştirilmiş
# Yeni özellikler:
#   • Yeni müşteri tahmin sistemi (POST /predict)
#   • Müşteri bazlı AI analiz kartı (GET /clients/{id}/ai-card)
#   • Şüpheli işlem tespiti / Fraud Detection (GET /transactions/{id}/fraud-status)
#   • İşlem durum renk sistemi (status: Normal / Suspicious / Fraud Risk)
#   • AI karar açıklama paneli (GET /explain/{id})
#   • Risk filtreleme sistemi (GET /clients/filter)
#   • Müşteri risk geçmişi grafiği (GET /clients/{id}/risk-history)
#   • Dashboard Confusion Matrix düzeltmesi (GET /model/confusion-matrix)
#
#    DÜZELTİLEN YERLER:
#   • filter_clients: SQL BETWEEN + ek koşul çakışması giderildi
#   • _synthetic_tx: conn=None durumunda double-close hatası giderildi
#   • confusion_matrix: sıfır bölme & eksik alan koruması eklendi
#   • _predict_response: fp=0 durumunda skor/100 hesabı düzeltildi
#   • _classify_transaction: chip kontrolü büyük/küçük harf normalize edildi
#   • client_risk_history: random.seed determinizmi güçlendirildi
#   • Tüm endpoint'lerde hata mesajları Türkçeleştirildi

from __future__ import annotations

import json
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, Query, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False

import pandas as pd
import joblib

try:
    from explainability import get_explanation_for_api, explain_model
    SHAP_MODULE_OK = True
except ImportError:
    SHAP_MODULE_OK = False

try:
    from auth import login_user as secure_login, revoke_token, get_current_user_from_token, get_audit_log
    AUTH_V2_OK = True
except ImportError:
    AUTH_V2_OK = False

log      = logging.getLogger("financeai.api")
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "financeai.db"

if not FASTAPI_OK:
    raise SystemExit("FastAPI kurulu değil. pip install fastapi uvicorn")

# ── Uygulama ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinanceAI API",
    description="Fraud tespiti, müşteri analizi ve risk yönetimi platformu",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    return response

security = HTTPBearer(auto_error=False)


# ── Auth Yardımcıları ─────────────────────────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Token gerekli.")
    if not AUTH_V2_OK:
        # Geliştirme modunda auth yoksa dev kullanıcısı döndür
        return {"sub": "0", "username": "dev", "role": "admin"}
    user = get_current_user_from_token(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş token.")
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin yetkisi gerekli.")
    return user


def require_analyst_or_above(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in ("admin", "analyst"):
        raise HTTPException(status_code=403, detail="Analyst veya admin yetkisi gerekli.")
    return user


# ── Pydantic Modeller ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    token:   Optional[str]  = None
    message: Optional[str]  = None
    user:    Optional[dict] = None

class PredictRequest(BaseModel):
    client_id:         Optional[int]   = Field(None,    description="Varsa mevcut müşteri ID'si")
    income:            float           = Field(50000.0, ge=0,       description="Yıllık gelir (TL)")
    transaction_count: int             = Field(30,      ge=0,       description="Toplam işlem sayısı")
    avg_spending:      float           = Field(150.0,   ge=0,       description="Ortalama harcama tutarı")
    category:          str             = Field("Other",             description="Harcama kategorisi")
    debt_ratio:        float           = Field(0.25,    ge=0, le=1, description="Borç/gelir oranı")

class TransactionStatusEnum:
    NORMAL     = "Normal"
    SUSPICIOUS = "Suspicious"
    FRAUD_RISK = "Fraud Risk"


# ── DB Yardımcıları ───────────────────────────────────────────────────────────

def _db_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail="Veritabanı bulunamadı.")
    return sqlite3.connect(DB_PATH)


def _read_table(query: str, params=None) -> list[dict]:
    try:
        conn = _db_conn()
        df   = pd.read_sql(query, conn, params=params)
        conn.close()
        return df.to_dict("records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(body: LoginRequest):
    if not AUTH_V2_OK:
        raise HTTPException(status_code=503, detail="Auth modülü yüklü değil.")
    result = secure_login(body.username, body.password)
    if result["success"]:
        return LoginResponse(success=True, token=result.get("token"), user=result.get("user"))
    return LoginResponse(success=False, message=result.get("message", "Giriş başarısız."))


@app.post("/auth/logout", tags=["Auth"])
async def logout(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if credentials and AUTH_V2_OK:
        revoke_token(credentials.credentials)
    return {"detail": "Çıkış başarılı."}


@app.get("/me", tags=["Auth"])
async def whoami(user: dict = Depends(get_current_user)):
    return {"username": user.get("username"), "role": user.get("role")}


# ── Sistem ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistem"])
async def health():
    return {
        "status":        "ok",
        "timestamp":     datetime.now().isoformat(),
        "db_exists":     DB_PATH.exists(),
        "model_exists":  (BASE_DIR / "data" / "best_model.pkl").exists(),
        "metrics_exist": (BASE_DIR / "data" / "model_metrics.json").exists(),
        "parquet_exist": (BASE_DIR / "data" / "transactions_clean.parquet").exists(),
        "shap_module":   SHAP_MODULE_OK,
        "auth_v2":       AUTH_V2_OK,
    }


# ── İstatistikler ─────────────────────────────────────────────────────────────

@app.get("/stats", tags=["Analiz"])
async def stats(user: dict = Depends(get_current_user)):
    if not DB_PATH.exists():
        return {"toplam": 0, "yuksek_risk": 0, "supheli": 0, "ort_fraud_skoru": 0.0}
    try:
        conn = _db_conn()
        if not _table_exists(conn, "client_ml"):
            conn.close()
            return {"toplam": 0, "yuksek_risk": 0, "supheli": 0, "ort_fraud_skoru": 0.0}
        df = pd.read_sql(
            """
            SELECT
                COUNT(*)                                                         AS toplam,
                SUM(CASE WHEN fraud_tahmini = 'Yuksek Risk' THEN 1 ELSE 0 END)  AS yuksek_risk,
                SUM(CASE WHEN fraud_tahmini = 'Supheli'     THEN 1 ELSE 0 END)  AS supheli,
                AVG(fraud_skoru)                                                  AS ort_fraud_skoru
            FROM client_ml
            """,
            conn,
        )
        conn.close()
        row = df.iloc[0]
        return {
            "toplam":          int(row["toplam"]          or 0),
            "yuksek_risk":     int(row["yuksek_risk"]     or 0),
            "supheli":         int(row["supheli"]         or 0),
            "ort_fraud_skoru": round(float(row["ort_fraud_skoru"] or 0), 2),
        }
    except Exception:
        return {"toplam": 0, "yuksek_risk": 0, "supheli": 0, "ort_fraud_skoru": 0.0}


@app.get("/stats/fraud", tags=["Analiz"])
async def fraud_stats(user: dict = Depends(require_analyst_or_above)):
    try:
        conn = _db_conn()
        df   = pd.read_sql(
            """
            SELECT
                fraud_tahmini,
                COUNT(*)         AS musteri_sayisi,
                AVG(fraud_skoru) AS ort_fraud_skoru,
                MAX(fraud_skoru) AS max_fraud_skoru,
                AVG(churn_skoru) AS ort_churn_skoru
            FROM client_ml
            GROUP BY fraud_tahmini
            """,
            conn,
        )
        conn.close()
        return df.to_dict("records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Müşteriler ────────────────────────────────────────────────────────────────

@app.get("/clients/top-risk", tags=["Analiz"])
async def top_risk_clients(
    limit: int = Query(20, ge=1, le=500),
    user:  dict = Depends(require_analyst_or_above),
):
    return _read_table(
        "SELECT client_id, fraud_skoru, fraud_tahmini, churn_skoru, "
        "dark_web_oran, tx_hata_oran, gece_ani_artis, tx_islem_sayisi "
        "FROM client_ml ORDER BY fraud_skoru DESC LIMIT ?",
        params=(limit,),
    )


@app.get("/clients/filter", tags=["Analiz"])
async def filter_clients(
    risk_level:      Optional[str] = Query(None, description="low | medium | high"),
    only_suspicious: bool          = Query(False, description="Yalnızca şüpheli işlem sahipleri"),
    min_score:       float         = Query(0.0,   ge=0,  le=100),
    max_score:       float         = Query(100.0, ge=0,  le=100),
    limit:           int           = Query(50,    ge=1,  le=500),
    user:            dict          = Depends(require_analyst_or_above),
):
    """
    Risk seviyesine göre müşteri filtreleme.

    risk_level: low (0-29), medium (30-59), high (60-100)
    only_suspicious: True ise fraud_tahmini = 'Supheli' olanlar
    min_score / max_score: fraud_skoru aralığı

    DÜZELTME: risk_level koşulu min_score/max_score BETWEEN ile çakışıyordu.
    Artık risk_level, min_score/max_score'un üzerine yazıyor (override ediyor).
    """
    # risk_level varsa min/max_score'u ezer
    if risk_level:
        level = risk_level.lower()
        if level == "low":
            min_score, max_score = 0.0, 29.99
        elif level == "medium":
            min_score, max_score = 30.0, 59.99
        elif level == "high":
            min_score, max_score = 60.0, 100.0

    conditions: list[str] = ["fraud_skoru BETWEEN ? AND ?"]
    params: list           = [min_score, max_score]

    if only_suspicious:
        conditions.append("fraud_tahmini = 'Supheli'")

    where = " AND ".join(conditions)
    params.append(limit)

    return _read_table(
        f"SELECT client_id, fraud_skoru, fraud_tahmini, churn_skoru, "
        f"dark_web_oran, tx_hata_oran, tx_islem_sayisi "
        f"FROM client_ml WHERE {where} ORDER BY fraud_skoru DESC LIMIT ?",
        params=params,
    )


@app.get("/clients/{client_id}/ai-card", tags=["Müşteri"])
async def client_ai_card(client_id: int, user: dict = Depends(get_current_user)):
    """
    Müşteri AI analiz kartı.
    İşlem geçmişine ek olarak modelin tam yorumunu, risk faktörlerini
    ve önerilerini döndürür.
    """
    rows = _read_table("SELECT * FROM client_ml WHERE client_id = ?", params=(client_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Müşteri {client_id} bulunamadı.")

    d        = rows[0]
    fraud_s  = float(d.get("fraud_skoru",    0) or 0)
    churn_s  = float(d.get("churn_skoru",    0) or 0)
    anomali  = float(d.get("anomali_skoru",  0) or 0)
    dark_web = float(d.get("dark_web_oran",  0) or 0)
    gece     = float(d.get("tx_gece_oran",   0) or 0)
    hata     = float(d.get("tx_hata_oran",   0) or 0)
    gece_ani = float(d.get("gece_ani_artis", 0) or 0)
    hata_ani = float(d.get("hata_ani_artis", 0) or 0)
    tx_sayi  = int(d.get("tx_islem_sayisi",  0) or 0)

    # Risk seviyesi ve aksiyon
    if fraud_s >= 60:
        level      = "High"
        color      = "red"
        behavioral = "Anormal Davranış"
        pattern    = "Yüksek riskli işlem örüntüsü tespit edildi."
        rec        = "🚨 Hesabı geçici kısıtlayın, müşteriyle iletişime geçin."
    elif fraud_s >= 30:
        level      = "Medium"
        color      = "yellow"
        behavioral = "Dikkat Gerektiriyor"
        pattern    = "Bazı alışılmadık işlem örüntüleri mevcut."
        rec        = "⚠️ Ek doğrulama isteyin, işlemleri yakından izleyin."
    else:
        level      = "Low"
        color      = "green"
        behavioral = "Normal"
        pattern    = "İşlem davranışı beklenen aralıkta."
        rec        = "✅ Standart izleme yeterli."

    # AI karar faktörleri
    ai_factors = []
    if dark_web > 0.1:
        ai_factors.append({
            "factor": "Dark Web Uyarısı",
            "detail": f"Dark web işlem oranı: {dark_web:.1%}",
            "weight": "high",
            "color":  "red",
            "icon":   "🌐",
        })
    if gece > 0.3:
        ai_factors.append({
            "factor": "Yüksek Gece İşlem Oranı",
            "detail": f"Gece işlem oranı: {gece:.1%}",
            "weight": "high",
            "color":  "red",
            "icon":   "🌙",
        })
    if hata > 0.2:
        ai_factors.append({
            "factor": "Hata Örüntüsü Tespit Edildi",
            "detail": f"İşlem hata oranı: {hata:.1%}",
            "weight": "medium",
            "color":  "orange",
            "icon":   "⚠️",
        })
    if churn_s > 60:
        ai_factors.append({
            "factor": "Yüksek Müşteri Kaybı Riski",
            "detail": f"Churn skoru: %{churn_s:.0f}",
            "weight": "medium",
            "color":  "orange",
            "icon":   "📉",
        })
    if gece_ani > 0.5:
        ai_factors.append({
            "factor": "Ani Gece İşlem Artışı",
            "detail": f"Gece işlemlerinde ani artış katsayısı: {gece_ani:.2f}",
            "weight": "medium",
            "color":  "orange",
            "icon":   "📈",
        })
    if hata_ani > 0.5:
        ai_factors.append({
            "factor": "Ani Hata Artışı",
            "detail": f"Hata oranında ani artış katsayısı: {hata_ani:.2f}",
            "weight": "medium",
            "color":  "orange",
            "icon":   "❌",
        })
    if anomali > 60:
        ai_factors.append({
            "factor": "Anomali Skoru Yüksek",
            "detail": f"Anomali skoru: {anomali:.1f}",
            "weight": "high",
            "color":  "red",
            "icon":   "🔴",
        })
    if tx_sayi > 200:
        ai_factors.append({
            "factor": "Olağandışı İşlem Frekansı",
            "detail": f"Toplam işlem sayısı: {tx_sayi}",
            "weight": "medium",
            "color":  "orange",
            "icon":   "🔄",
        })
    if not ai_factors:
        ai_factors.append({
            "factor": "Risk Faktörü Tespit Edilmedi",
            "detail": "Tüm göstergeler beklenen aralıkta.",
            "weight": "low",
            "color":  "green",
            "icon":   "✅",
        })

    return {
        "client_id": client_id,
        "ai_risk_assessment": {
            "risk_score":                   round(fraud_s, 2),
            "risk_level":                   level,
            "risk_color":                   color,
            "behavioral_status":            behavioral,
            "transaction_pattern_analysis": pattern,
            "recommendation":               rec,
            "confidence":                   _confidence_from_score(fraud_s),
        },
        "ai_decision_explanation": {
            "summary":           _build_explanation_summary(fraud_s, dark_web, gece, hata),
            "high_tx_amount":    fraud_s > 60 and dark_web > 0.1,
            "unusual_frequency": tx_sayi > 200,
            "error_pattern":     hata > 0.2,
            "spending_shift":    gece_ani > 0.5 or hata_ani > 0.5,
            "anomaly_detected":  anomali > 60,
        },
        "ai_factors": ai_factors,
        "raw_scores": {
            "fraud_skoru":    round(fraud_s,  2),
            "churn_skoru":    round(churn_s,  2),
            "anomali_skoru":  round(anomali,  2),
            "dark_web_oran":  round(dark_web, 4),
            "tx_gece_oran":   round(gece,     4),
            "tx_hata_oran":   round(hata,     4),
            "gece_ani_artis": round(gece_ani, 4),
            "hata_ani_artis": round(hata_ani, 4),
        },
    }


@app.get("/clients/{client_id}/risk-history", tags=["Müşteri"])
async def client_risk_history(
    client_id: int,
    days:      int  = Query(90, ge=7, le=365),
    user:      dict = Depends(get_current_user),
):
    """
    Müşteri risk skoru geçmişi (simüle edilmiş zaman serisi).
    Frontend'de X=Tarih, Y=Risk Score grafiği için kullanılır.
    """

    rows = _read_table("SELECT * FROM client_ml WHERE client_id = ?", params=(client_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Müşteri {client_id} bulunamadı.")

    d       = rows[0]
    current = float(d.get("fraud_skoru", 0) or 0)

    random.seed(client_id * 31 + days)
    history = []
    score   = max(0.0, current - random.uniform(5, 25))

    for i in range(days, -1, -1):
        date  = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        delta = random.uniform(-3.5, 4.0)
        score = min(100.0, max(0.0, score + delta))

        # Son gün her zaman güncel skor
        if i == 0:
            score = current

        history.append({
            "date":        date,
            "fraud_score": round(score, 2),
            "risk_level":  "High" if score >= 60 else "Medium" if score >= 30 else "Low",
        })

    trend = "rising" if history[-1]["fraud_score"] > history[0]["fraud_score"] else "falling"

    return {
        "client_id":     client_id,
        "days":          days,
        "current_score": round(current, 2),
        "trend":         trend,
        "history":       history,
    }


@app.get("/clients/{client_id}", tags=["Müşteri"])
async def client_detail(client_id: int, user: dict = Depends(get_current_user)):
    rows = _read_table("SELECT * FROM client_ml WHERE client_id = ?", params=(client_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Müşteri {client_id} bulunamadı.")
    return rows[0]


# ── Tahmin Yardımcıları ───────────────────────────────────────────────────────

def _confidence_from_score(score: float) -> int:
    """Fraud skoruna göre model güven yüzdesi döndürür."""
    if score >= 70:
        return 88
    elif score >= 50:
        return 80
    elif score >= 30:
        return 72
    elif score >= 15:
        return 78
    return 90


def _build_explanation_summary(fraud_s: float, dark_web: float, gece: float, hata: float) -> str:
    parts = []
    if fraud_s >= 60:
        parts.append("Yüksek fraud skoru tespit edildi")
    if dark_web > 0.1:
        parts.append(f"dark web işlem oranı yüksek ({dark_web:.1%})")
    if gece > 0.3:
        parts.append(f"gece işlem oranı yüksek ({gece:.1%})")
    if hata > 0.2:
        parts.append(f"işlem hata oranı kritik ({hata:.1%})")
    if not parts:
        return "Müşteri profili normal davranış sınırları içinde."
    return "Model kararı: " + "; ".join(parts) + "."


def _rule_score(req: PredictRequest) -> float:
    """ML modeli yoksa kural tabanlı skor üretir."""
    score = 0.0
    if req.debt_ratio > 0.6:
        score += 30
    elif req.debt_ratio > 0.4:
        score += 15
    if req.avg_spending > 500:
        score += 20
    elif req.avg_spending > 300:
        score += 10
    if req.transaction_count > 100:
        score += 15
    elif req.transaction_count > 60:
        score += 8
    if req.category in {"Online Shopping", "Entertainment", "Electronics"}:
        score += 10
    if req.income < 20_000:
        score += 15
    elif req.income < 40_000:
        score += 5
    return min(score, 100.0)


def _predict_response(score: float, client_id=None, fp: float = 0.0) -> dict:
 
    fraud_probability = fp if fp > 0 else min(score / 100.0, 1.0)

    if score >= 60:
        level   = "High"
        rec     = "🚨 Acil inceleme gerekli. Hesabı geçici kısıtlayın."
        conf    = 85
        color   = "red"
        factors = ["Yüksek harcama tutarı", "Riskli borç oranı", "Alışılmadık işlem kategorisi"]
    elif score >= 30:
        level   = "Medium"
        rec     = "⚠️ İzlemeye alın. Ek doğrulama isteyin."
        conf    = 72
        color   = "yellow"
        factors = ["Orta düzey harcama", "Dikkat gerektiren borç oranı"]
    else:
        level   = "Low"
        rec     = "✅ Normal davranış. Standart izleme yeterli."
        conf    = 90
        color   = "green"
        factors = ["Normal davranış profili"]

    return {
        "client_id":         client_id,
        "risk_score":        round(score, 2),
        "risk_level":        level,
        "risk_color":        color,
        "fraud_probability": round(fraud_probability, 4),
        "confidence":        conf,
        "recommendation":    rec,
        "ai_factors":        factors,
        "timestamp":         datetime.now().isoformat(),
    }


# ── Tahmin ────────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["Tahmin"])
async def predict(req: PredictRequest, user: dict = Depends(get_current_user)):
    # 1) Mevcut müşteri DB kaydı
    if req.client_id and DB_PATH.exists():
        try:
            rows = _read_table(
                "SELECT fraud_skoru FROM client_ml WHERE client_id = ?",
                params=(req.client_id,),
            )
            if rows:
                return _predict_response(
                    float(rows[0].get("fraud_skoru", 0) or 0), req.client_id
                )
        except Exception:
            pass  # DB'de bulunamazsa model ile devam et

    # 2) ML modeli
    model_path = BASE_DIR / "data" / "best_model.pkl"
    fcols_path = BASE_DIR / "data" / "feature_cols.pkl"
    score, fp  = 0.0, 0.0

    if model_path.exists() and fcols_path.exists():
        try:
            model = joblib.load(model_path)
            fcols = joblib.load(fcols_path)
            feat  = {
                "amount":                req.avg_spending,
                "abs_amount":            abs(req.avg_spending),
                "saat":                  12,
                "gun":                   2,
                "ay":                    datetime.now().month,
                "gece_islemi":           0,
                "hafta_sonu":            0,
                "hata_var":              0,
                "online_islem":          1 if req.category == "Online Shopping" else 0,
                "buyuk_islem":           1 if req.avg_spending > 1000 else 0,
                "negatif":               0,
                "zscore_flag":           0,
                "velocity_dk":           9999,
                "musteri_ort_tutar":     req.avg_spending,
                "musteri_std_tutar":     req.avg_spending * 0.3,
                "musteri_islem_sayisi":  req.transaction_count,
                "musteri_gece_oran":     0.05,
                "musteri_hata_oran":     0.01,
                "musteri_zscore_oran":   0.02,
                "tutar_sapma":           0.0,
            }
            row_df = pd.DataFrame([{c: feat.get(c, 0) for c in fcols}])
            fp     = float(model.predict_proba(row_df)[0, 1])
            score  = fp * 100
        except Exception as exc:
            log.warning("ML model hatası, kural tabanlı skora geçiliyor: %s", exc)
            score = _rule_score(req)
    else:
        # 3) Kural tabanlı skor
        score = _rule_score(req)

    return _predict_response(score, req.client_id, fp)


# ── İşlem Geçmişi + Fraud Tespiti ────────────────────────────────────────────

_FRAUD_COLORS = {
    TransactionStatusEnum.NORMAL:     "green",
    TransactionStatusEnum.SUSPICIOUS: "yellow",
    TransactionStatusEnum.FRAUD_RISK: "red",
}


def _classify_transaction(
    amount: float,
    threshold: float,
    errors: str,
    chip: str,
    velocity_flag: bool = False,
    unexpected_category: bool = False,
) -> tuple[str, list[str]]:
    """
    İşlemi Normal / Suspicious / Fraud Risk olarak sınıflandırır.

    DÜZELTME: chip kontrolünde büyük/küçük harf normalize edildi (.lower() zaten yapılıyor
    ama "online" yerine "online transaction" gibi tam string ile karşılaştırma düzeltildi).
    """
    reasons      = []
    risk_points  = 0
    chip_lower   = chip.strip().lower()

    if abs(amount) > threshold:
        reasons.append(f"Yüksek tutar (${amount:,.0f})")
        risk_points += 2

    if errors.strip():
        reasons.append(f"Hata kodu: {errors[:40]}")
        risk_points += 2

    # Chip kullanılmayan işlem — online veya swipe
    if "online" in chip_lower or "swipe" in chip_lower:
        reasons.append("Chip'siz işlem (online/swipe)")
        risk_points += 1

    if velocity_flag:
        reasons.append("Kısa sürede çok işlem (velocity)")
        risk_points += 2

    if unexpected_category:
        reasons.append("Beklenmeyen kategori")
        risk_points += 1

    if risk_points >= 4:
        return TransactionStatusEnum.FRAUD_RISK, reasons
    elif risk_points >= 1:
        return TransactionStatusEnum.SUSPICIOUS, reasons
    return TransactionStatusEnum.NORMAL, []


@app.get("/transactions/{client_id}/fraud-status", tags=["İşlemler"])
async def transaction_fraud_status(
    client_id: int,
    limit:     int  = Query(100, ge=1, le=1000),
    user:      dict = Depends(get_current_user),
):
    """
    Müşterinin işlem geçmişini döndürür. Her işlem:
      - status: Normal | Suspicious | Fraud Risk
      - status_color: green | yellow | red
      - reasons: tespit açıklamaları
    listesi ile döner.
    """
    try:
        conn   = _db_conn()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        tx_table = next(
            (t for t in ["transactions_data", "transactions", "transaction"] if t in tables),
            None,
        )
        cid_col = None

        if tx_table:
            cols    = [r[0] for r in conn.execute(f"PRAGMA table_info({tx_table})").fetchall()]
            cid_col = next((c for c in ["client_id", "client", "user_id"] if c in cols), None)

        if not tx_table or not cid_col:
            conn.close()
            # İşlem tablosu yoksa client_ml'den sentetik üret
            return _synthetic_tx(client_id, limit)

        df      = pd.read_sql(
            f"SELECT * FROM {tx_table} WHERE {cid_col}=? LIMIT ?",
            conn,
            params=(client_id, limit),
        )
        avg_row = conn.execute(
            f"SELECT AVG(ABS(amount)) FROM {tx_table} WHERE amount IS NOT NULL"
        ).fetchone()
        conn.close()

        avg_amount = float(avg_row[0] or 150)
        threshold  = avg_amount * 3

        txs = []
        for idx, row in df.iterrows():
            amount = float(
                str(row.get("amount", 0)).replace("$", "").replace(",", "") or 0
            )
            errors = str(row.get("errors", "") or "")
            chip   = str(row.get("use_chip", "") or "")

            # Velocity: ardışık satırlarda aynı dakika kontrolü
            velocity_flag = False
            if idx > 0 and idx in df.index:
                prev_idx  = df.index[df.index.get_loc(idx) - 1]
                prev_date = str(df.loc[prev_idx].get("date", ""))
                curr_date = str(row.get("date", ""))
                if prev_date and curr_date and prev_date[:16] == curr_date[:16]:
                    velocity_flag = True

            mcc = str(row.get("mcc", row.get("category", ""))).lower()
            unexpected_category = any(
                k in mcc for k in ["casino", "gambling", "crypto", "adult"]
            )

            status, reasons = _classify_transaction(
                amount, threshold, errors, chip, velocity_flag, unexpected_category
            )

            txs.append({
                "date":          str(row.get("date", "")),
                "amount":        amount,
                "merchant_city": str(row.get("merchant_city", row.get("city", ""))),
                "category":      str(row.get("mcc", row.get("category", ""))),
                "use_chip":      str(row.get("use_chip", "")),
                "status":        status,
                "status_color":  _FRAUD_COLORS[status],
                "reasons":       reasons,
            })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "client_id": client_id,
        "summary": {
            "total":      len(txs),
            "normal":     sum(1 for t in txs if t["status"] == TransactionStatusEnum.NORMAL),
            "suspicious": sum(1 for t in txs if t["status"] == TransactionStatusEnum.SUSPICIOUS),
            "fraud_risk": sum(1 for t in txs if t["status"] == TransactionStatusEnum.FRAUD_RISK),
            "threshold":  round(threshold, 2),
        },
        "transactions": txs,
    }


def _synthetic_tx(client_id: int, limit: int) -> dict:
    if not DB_PATH.exists():
        return {
            "client_id":    client_id,
            "summary":      {"total": 0, "normal": 0, "suspicious": 0, "fraud_risk": 0, "threshold": 0},
            "transactions": [],
        }

    try:
        _conn = _db_conn()
        df    = pd.read_sql(
            "SELECT * FROM client_ml WHERE client_id=?", _conn, params=(client_id,)
        )
        _conn.close()
    except Exception:
        return {
            "client_id":    client_id,
            "summary":      {"total": 0, "normal": 0, "suspicious": 0, "fraud_risk": 0, "threshold": 0},
            "transactions": [],
        }

    if df.empty:
        return {
            "client_id":    client_id,
            "summary":      {"total": 0, "normal": 0, "suspicious": 0, "fraud_risk": 0, "threshold": 0},
            "transactions": [],
        }

    d          = df.iloc[0]
    avg_amount = float(d.get("tx_ortalama_tutar", 150) or 150)
    hata_oran  = float(d.get("tx_hata_oran",     0)   or 0)
    gece_oran  = float(d.get("tx_gece_oran",     0)   or 0)
    threshold  = avg_amount * 2.5
    n          = min(int(d.get("tx_islem_sayisi", 10) or 10), limit)

    random.seed(client_id * 13)
    cities     = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]
    categories = ["Market", "Restaurant", "Online Shopping", "Fuel", "Health"]
    chips      = ["Chip Transaction", "Online Transaction", "Swipe Transaction"]

    txs = []
    for _ in range(n):
        amount         = avg_amount * random.uniform(0.3, 3.0)
        has_error      = random.random() < hata_oran
        is_night       = random.random() < gece_oran
        chip           = random.choice(chips)
        errors         = "ERR_DECLINED" if has_error else ""
        velocity_flag  = random.random() < 0.05
        unexpected_cat = random.random() < 0.03

        status, reasons = _classify_transaction(
            amount, threshold, errors, chip, velocity_flag, unexpected_cat
        )
        if is_night and status == TransactionStatusEnum.NORMAL:
            reasons.append("Gece işlemi")
            status = TransactionStatusEnum.SUSPICIOUS

        txs.append({
            "date":          (datetime.now() - timedelta(days=random.randint(0, 90))).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "amount":        round(amount, 2),
            "merchant_city": random.choice(cities),
            "category":      random.choice(categories),
            "use_chip":      chip,
            "status":        status,
            "status_color":  _FRAUD_COLORS[status],
            "reasons":       reasons,
        })

    return {
        "client_id": client_id,
        "summary": {
            "total":      len(txs),
            "normal":     sum(1 for t in txs if t["status"] == TransactionStatusEnum.NORMAL),
            "suspicious": sum(1 for t in txs if t["status"] == TransactionStatusEnum.SUSPICIOUS),
            "fraud_risk": sum(1 for t in txs if t["status"] == TransactionStatusEnum.FRAUD_RISK),
            "threshold":  round(threshold, 2),
        },
        "transactions": txs,
    }


# ── Model ─────────────────────────────────────────────────────────────────────

@app.get("/model/metrics", tags=["Model"])
async def model_metrics(user: dict = Depends(require_analyst_or_above)):
    metrics_path = BASE_DIR / "data" / "model_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrikler bulunamadı. Pipeline'ı çalıştırın.",
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/model/confusion-matrix", tags=["Model"])
async def confusion_matrix_endpoint(user: dict = Depends(require_analyst_or_above)):
    """
    Confusion matrix döndürür.

    DÜZELTME:
      - model_metrics.json yoksa uyarılı boş yanıt döner
      - confusion_matrix anahtarı eksikse sıfır değerleri kullanılır
      - total == 0 durumunda warning eklenir
      - precision + recall == 0 durumunda sıfır bölme koruması
      - f1_score hesabı korunmuş
    """
    metrics_path = BASE_DIR / "data" / "model_metrics.json"
    if not metrics_path.exists():
        return {
            "cells":     {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy":  0.0,
            "precision": 0.0,
            "recall":    0.0,
            "f1_score":  0.0,
            "total":     0,
            "warning":   "model_metrics.json bulunamadı. Pipeline'ı çalıştırın.",
        }

    try:
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
    except json.JSONDecodeError as exc:
        return {
            "cells":     {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy":  0.0,
            "precision": 0.0,
            "recall":    0.0,
            "f1_score":  0.0,
            "total":     0,
            "warning":   f"model_metrics.json okunamadı: {exc}",
        }

    cm_raw = m.get("confusion_matrix", [[0, 0], [0, 0]])
    try:
        if isinstance(cm_raw, list) and len(cm_raw) == 2:
            tn = int(cm_raw[0][0])
            fp = int(cm_raw[0][1])
            fn = int(cm_raw[1][0])
            tp = int(cm_raw[1][1])
        else:
            tn = fp = fn = tp = 0
    except (IndexError, TypeError, ValueError):
        tn = fp = fn = tp = 0

    total   = tn + fp + fn + tp
    warning = None
    if total == 0:
        warning = "Confusion matrix değerleri sıfır. Model evaluation çalıştırılmamış olabilir."

    precision = float(m.get("precision", 0.0) or 0.0)
    recall    = float(m.get("recall",    0.0) or 0.0)
    f1        = float(m.get("f1_score",  0.0) or 0.0)

    # Sıfır bölme koruması
    if f1 == 0.0 and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "cells":     {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy":  float(m.get("accuracy", 0.0) or 0.0),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
        "total":     total,
        "warning":   warning,
    }


# ── Açıklanabilirlik ──────────────────────────────────────────────────────────

@app.get("/explain/{client_id}", tags=["Açıklanabilirlik"])
async def explain_client(client_id: int, user: dict = Depends(require_analyst_or_above)):
    """SHAP değerleri ile müşteri bazlı model açıklaması."""
    if not SHAP_MODULE_OK:
        raise HTTPException(
            status_code=503,
            detail="SHAP modülü yüklü değil. pip install shap",
        )
    return get_explanation_for_api(client_id)


@app.get("/model/shap-summary", tags=["Açıklanabilirlik"])
async def shap_summary(user: dict = Depends(require_admin)):
    if not SHAP_MODULE_OK:
        raise HTTPException(status_code=503, detail="SHAP modülü yüklü değil.")
    try:
        conn = _db_conn()
        df   = pd.read_sql("SELECT * FROM client_ml LIMIT 2000", conn)
        conn.close()
        return explain_model(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.get("/admin/audit", tags=["Admin"])
async def audit_log(limit: int = Query(50, ge=1, le=500), user: dict = Depends(require_admin)):
    if not AUTH_V2_OK:
        raise HTTPException(status_code=503, detail="Auth v2 modülü yüklü değil.")
    return get_audit_log(limit=limit)


@app.get("/admin/users", tags=["Admin"])
async def list_users(user: dict = Depends(require_admin)):
    return _read_table(
        "SELECT id, username, email, role, status, created_at FROM users ORDER BY id"
    )


@app.get("/admin/db-stats", tags=["Admin"])
async def db_stats(user: dict = Depends(require_admin)):
    if not DB_PATH.exists():
        return {"durum": "DB bulunamadı"}
    try:
        conn   = _db_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        result = {"db_path": str(DB_PATH)}
        for (tbl,) in tables:
            try:
                result[tbl] = conn.execute(f"SELECT COUNT(*) FROM '{tbl}'").fetchone()[0]
            except Exception:
                result[tbl] = "?"
        conn.close()
        return result
    except Exception as exc:
        return {"hata": str(exc)}


# ── Global Error Handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error("Beklenmeyen hata: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Sunucu hatası. Loglara bakın."},
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)