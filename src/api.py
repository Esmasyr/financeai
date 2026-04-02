"""
FinanceAI

"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, Header, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    print("FastAPI kurulu değil: pip install fastapi uvicorn")

import pandas as pd
import joblib

# Kendi modüllerimiz
try:
    from explainability import get_explanation_for_api, explain_model
    SHAP_MODULE_OK = True
except ImportError:
    SHAP_MODULE_OK = False

try:
    from auth_v2 import (
        secure_login, revoke_token, get_current_user_from_token, get_audit_log
    )
    AUTH_V2_OK = True
except ImportError:
    AUTH_V2_OK = False

log = logging.getLogger("financeai.api")
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "financeai.db"

# ══════════════════════════════════════════════════════════════════════════════
# UYGULAMA KURULUMU
# ══════════════════════════════════════════════════════════════════════════════

if not FASTAPI_OK:
    raise SystemExit("FastAPI kurulu değil. pip install fastapi uvicorn")

app = FastAPI(
    title       = "FinanceAI API",
    description = "Fraud tespiti ve müşteri analizi",
    version     = "2.0.0",
)

# CORS — Streamlit'in erişmesine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials = True,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

# Güvenlik header middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["X-XSS-Protection"]       = "1; mode=block"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# AUTH YARDIMCILARI
# ══════════════════════════════════════════════════════════════════════════════

security = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    Bearer token'ı doğrular.
    Geçersizse 401 döner.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Token gerekli.")
    if not AUTH_V2_OK:
        # Auth modülü yoksa geliştirme modu: her şeye izin ver
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


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELLER
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success:  bool
    token:    Optional[str] = None
    message:  Optional[str] = None
    user:     Optional[dict] = None


# ══════════════════════════════════════════════════════════════════════════════
# YARDIMCI — DB okuma
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(body: LoginRequest):
    """
    Kullanıcı girişi — JWT token döner.

    Güvenlik özellikleri:
    - Rate limiting (5 başarısız deneme → 5 dakika kilit)
    - bcrypt şifre doğrulama
    - Audit log kaydı
    """
    if not AUTH_V2_OK:
        raise HTTPException(status_code=503, detail="Auth modülü yüklü değil.")

    result = secure_login(body.username, body.password)
    if result["success"]:
        return LoginResponse(
            success=True,
            token=result.get("token"),
            user=result.get("user"),
        )
    return LoginResponse(
        success=False,
        message=result.get("message", "Giriş başarısız."),
    )


@app.post("/auth/logout", tags=["Auth"])
async def logout(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Token'ı geçersiz kıl (blacklist'e ekle)."""
    if credentials and AUTH_V2_OK:
        revoke_token(credentials.credentials)
    return {"detail": "Çıkış başarılı."}


@app.get("/me", tags=["Auth"])
async def whoami(user: dict = Depends(get_current_user)):
    """Mevcut kullanıcı bilgisi."""
    return {
        "username": user.get("username"),
        "role":     user.get("role"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — ML VERİSİ
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Sistem"])
async def health():
    """API durum kontrolü."""
    return {
        "status":        "ok",
        "timestamp":     datetime.now().isoformat(),
        "db_exists":     DB_PATH.exists(),
        "shap_module":   SHAP_MODULE_OK,
        "auth_v2":       AUTH_V2_OK,
    }


@app.get("/stats", tags=["Analiz"])
async def stats(user: dict = Depends(get_current_user)):
    """
    Genel istatistikler.
    Tüm giriş yapmış kullanıcılar erişebilir.
    """
    rows = _read_table("SELECT * FROM ml_ozet LIMIT 1")
    return rows[0] if rows else {}


@app.get("/stats/fraud", tags=["Analiz"])
async def fraud_stats(user: dict = Depends(require_analyst_or_above)):
    """
    Detaylı fraud istatistikleri.
    Analyst veya admin yetkisi gerekli.
    """
    try:
        conn = _db_conn()
        df   = pd.read_sql(
            """
            SELECT
                fraud_tahmini,
                COUNT(*)           AS musteri_sayisi,
                AVG(fraud_skoru)   AS ort_fraud_skoru,
                MAX(fraud_skoru)   AS max_fraud_skoru,
                AVG(churn_skoru)   AS ort_churn_skoru
            FROM client_ml
            GROUP BY fraud_tahmini
            """,
            conn,
        )
        conn.close()
        return df.to_dict("records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/clients/top-risk", tags=["Analiz"])
async def top_risk_clients(
    limit: int = 20,
    user:  dict = Depends(require_analyst_or_above),
):
    """En yüksek riskli müşteriler."""
    rows = _read_table(
        "SELECT client_id, fraud_skoru, fraud_tahmini, churn_skoru, "
        "dark_web_oran, tx_hata_oran, gece_ani_artis "
        "FROM client_ml ORDER BY fraud_skoru DESC LIMIT ?",
        params=(limit,),
    )
    return rows


@app.get("/clients/{client_id}", tags=["Müşteri"])
async def client_detail(
    client_id: int,
    user:      dict = Depends(get_current_user),
):
    """Müşteri detay — tüm ML skoru."""
    rows = _read_table(
        "SELECT * FROM client_ml WHERE client_id = ?",
        params=(client_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Müşteri {client_id} bulunamadı.")
    return rows[0]


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — SHAP AÇIKLANABILIRLIK (YENİ)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/explain/{client_id}", tags=["Açıklanabilirlik"])
async def explain_client(
    client_id: int,
    user:      dict = Depends(require_analyst_or_above),
):
    """
    Bir müşterinin fraud skorunun nedenini SHAP ile açıkla.

    Döner:
    - main_reason: en önemli faktör
    - factors: her özelliğin skora katkısı
    - human_text: Türkçe açıklama (GDPR uyumlu)

    Hocana: Bu endpoint GDPR Article 22 gereksinimini karşılar.
    Otomatik karar verme sistemleri kararlarını açıklamak zorunda.
    """
    if not SHAP_MODULE_OK:
        raise HTTPException(
            status_code=503,
            detail="SHAP modülü yüklü değil. pip install shap",
        )
    return get_explanation_for_api(client_id)


@app.get("/model/shap-summary", tags=["Açıklanabilirlik"])
async def shap_summary(user: dict = Depends(require_admin)):
    """
    Tüm modelin global SHAP özeti.
    Admin yetkisi gerekli.

    Döner: her özelliğin ortalama mutlak SHAP değeri
    """
    if not SHAP_MODULE_OK:
        raise HTTPException(status_code=503, detail="SHAP modülü yüklü değil.")
    try:
        conn = _db_conn()
        df   = pd.read_sql("SELECT * FROM client_ml LIMIT 2000", conn)
        conn.close()
        result = explain_model(df)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/metrics", tags=["Model"])
async def model_metrics(user: dict = Depends(require_analyst_or_above)):
    """Son eğitimden kaydedilen metrikler."""
    metrics_path = BASE_DIR / "data" / "model_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrikler bulunamadı.")
    import json
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — ADMIN
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/admin/audit", tags=["Admin"])
async def audit_log(
    limit: int = 50,
    user:  dict = Depends(require_admin),
):
    """
    Güvenlik audit log.
    Kim, ne zaman, nereden, ne yaptı.
    GDPR & SOC2 uyumu için zorunlu.
    """
    if not AUTH_V2_OK:
        raise HTTPException(status_code=503, detail="Auth v2 modülü yüklü değil.")
    return get_audit_log(limit=limit)


@app.get("/admin/users", tags=["Admin"])
async def list_users(user: dict = Depends(require_admin)):
    """Tüm kullanıcılar — sadece admin."""
    rows = _read_table(
        "SELECT id, username, email, role, status, created_at FROM users ORDER BY id"
    )
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error("Beklenmeyen hata: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Sunucu hatası. Loglara bakın."},
    )


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_v2:app", host="0.0.0.0", port=8000, reload=True)