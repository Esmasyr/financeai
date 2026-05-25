"""
FinanceAI — Güvenlik ve Auth Modülü
====================================
dashboard.py'nin beklediği fonksiyonlar:
  login_user, register_user, get_all_users, get_pending_users,
  approve_user, reject_user, update_user_role, delete_user,
  get_login_stats, change_password, init_db, admin_exists

Ek güvenlik katmanları:
  - bcrypt ile şifre hash (fallback: sha256+salt)
  - JWT access token
  - Rate limiting — brute force koruması
  - Token blacklist — logout gerçekten çalışır
  - Audit log — kim ne zaman ne yaptı
  - Refresh token mekanizması

Gereksinim (opsiyonel, yoksa fallback devreye girer):
    pip install bcrypt PyJWT
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

log = logging.getLogger("financeai.auth")

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "financeai.db"

# ── Güvenlik sabitleri ────────────────────────────────────────────────────────
JWT_SECRET         = os.environ.get("FINSIGHT_JWT_SECRET",   "DEGISTIR_BUNU_PRODUCTION_DA")
REFRESH_SECRET     = os.environ.get("FINSIGHT_REFRESH_SECRET","REFRESH_SECRET_DEGISTIR")
JWT_ALGORITHM      = "HS256"
TOKEN_EXPIRE_MIN   = 60 * 8   # access token: 8 saat
REFRESH_TOKEN_DAYS = 30       # refresh token: 30 gün
MAX_LOGIN_TRIES    = 5        # başarısız denemeden sonra kilitle
LOCKOUT_SECONDS    = 300      # 5 dakika kilit

# ── Opsiyonel kütüphaneler ────────────────────────────────────────────────────
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    log.warning("bcrypt kurulu değil — sha256 fallback aktif (pip install bcrypt önerilir)")

try:
    # pyrefly: ignore [missing-import]
    import jwt as pyjwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    log.warning("PyJWT kurulu değil — token desteği devre dışı (pip install PyJWT önerilir)")


# ══════════════════════════════════════════════════════════════════════════════
# VERİTABANI
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Tüm tabloları oluşturur. dashboard.py başlangıçta çağırır.
    """
    with _get_conn() as conn:
        # Kullanıcılar
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'viewer',
                status        TEXT    NOT NULL DEFAULT 'pending',
                display_name  TEXT,
                email         TEXT,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                last_login    TEXT
            )
        """)
        # Audit log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                zaman       TEXT    NOT NULL,
                kullanici   TEXT    NOT NULL,
                aksiyon     TEXT    NOT NULL,
                ip          TEXT,
                basarili    INTEGER NOT NULL,
                detay       TEXT
            )
        """)
        # Refresh token tablosu
        conn.execute("""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                token_hash  TEXT    NOT NULL UNIQUE,
                created_at  TEXT    NOT NULL,
                expires_at  TEXT    NOT NULL,
                revoked     INTEGER NOT NULL DEFAULT 0,
                ip          TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    log.info("DB tabloları hazır: %s", DB_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# ŞIFRE HASH
# ══════════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()
    salt   = os.urandom(32).hex()
    hashed = hashlib.sha256((plain + salt).encode()).hexdigest()
    return f"sha256:{salt}:{hashed}"


def verify_password(plain: str, hashed: str) -> bool:
    if hashed.startswith("sha256:"):
        _, salt, stored = hashed.split(":", 2)
        return hashlib.sha256((plain + salt).encode()).hexdigest() == stored
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except Exception:
            return False
    return False


# ══════════════════════════════════════════════════════════════════════════════
# JWT ACCESS TOKEN
# ══════════════════════════════════════════════════════════════════════════════

_blacklist: set[str] = set()  # production'da Redis kullan


def create_token(user_id: int, username: str, role: str) -> str | None:
    if not JWT_AVAILABLE:
        return None
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "exp":      now + timedelta(minutes=TOKEN_EXPIRE_MIN),
        "iat":      now,
    }
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict | None:
    if not JWT_AVAILABLE or token in _blacklist:
        return None
    try:
        return pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except pyjwt.exceptions.ExpiredSignatureError:
        log.info("Access token süresi doldu.")
        return None
    except pyjwt.exceptions.InvalidTokenError as exc:
        log.warning("Geçersiz token: %s", exc)
        return None


def revoke_token(token: str) -> None:
    _blacklist.add(token)
    log.info("Token blacklist'e eklendi.")


def get_current_user_from_token(token: str) -> dict | None:
    return verify_token(token)


# ══════════════════════════════════════════════════════════════════════════════
# REFRESH TOKEN
# ══════════════════════════════════════════════════════════════════════════════

def _hash_refresh(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_refresh_token(user_id: int, ip: str = "unknown") -> str:
    """
    Güvenli rastgele refresh token üretir, hash'ini DB'ye kaydeder.
    Ham token'ı döner (bir kez gösterilir, sonra hash'i saklanır).
    """
    raw   = secrets.token_urlsafe(64)
    thash = _hash_refresh(raw)
    now   = datetime.now(tz=timezone.utc)
    exp   = now + timedelta(days=REFRESH_TOKEN_DAYS)

    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (user_id, token_hash, created_at, expires_at, ip) "
            "VALUES (?,?,?,?,?)",
            (user_id, thash, now.isoformat(), exp.isoformat(), ip),
        )
        conn.commit()
    return raw


def use_refresh_token(raw_token: str) -> dict | None:
    """
    Refresh token ile yeni access+refresh token çifti döner.
    Kullanılan token iptal edilir (rotation).

    Döner:
        {"access_token": "...", "refresh_token": "...", "user": {...}}
        None — geçersiz/süresi dolmuş/iptal edilmiş
    """
    thash = _hash_refresh(raw_token)
    now   = datetime.now(tz=timezone.utc)

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT rt.id, rt.user_id, rt.expires_at, rt.revoked, "
            "       u.username, u.role, u.status, u.display_name "
            "FROM refresh_tokens rt "
            "JOIN users u ON u.id = rt.user_id "
            "WHERE rt.token_hash = ?",
            (thash,),
        ).fetchone()

        if not row:
            log.warning("Refresh token bulunamadı.")
            return None
        if row["revoked"]:
            log.warning("İptal edilmiş refresh token kullanım girişimi — user_id=%s", row["user_id"])
            # Tüm refresh token'larını iptal et (token theft şüphesi)
            conn.execute(
                "UPDATE refresh_tokens SET revoked=1 WHERE user_id=?",
                (row["user_id"],),
            )
            conn.commit()
            return None

        exp = datetime.fromisoformat(row["expires_at"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if now > exp:
            log.info("Refresh token süresi dolmuş.")
            return None

        if row["status"] not in ("active", "approved"):
            return None

        # Eski token'ı iptal et (rotation)
        conn.execute(
            "UPDATE refresh_tokens SET revoked=1 WHERE id=?",
            (row["id"],),
        )
        conn.commit()

    # Yeni token çifti üret
    new_access  = create_token(row["user_id"], row["username"], row["role"])
    new_refresh = create_refresh_token(row["user_id"])

    audit(row["username"], "TOKEN_REFRESH", True, detay="rotation")
    return {
        "access_token":  new_access,
        "refresh_token": new_refresh,
        "user": {
            "id":           row["user_id"],
            "username":     row["username"],
            "role":         row["role"],
            "display_name": row["display_name"] or row["username"],
        },
    }


def revoke_all_refresh_tokens(user_id: int) -> None:
    """Logout — kullanıcının tüm refresh token'larını iptal et."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE refresh_tokens SET revoked=1 WHERE user_id=?",
            (user_id,),
        )
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════

_attempts: dict[str, list[float]] = {}


def check_rate_limit(identifier: str) -> tuple[bool, int]:
    now   = time.time()
    times = [t for t in _attempts.get(identifier, []) if now - t < LOCKOUT_SECONDS]
    _attempts[identifier] = times
    if len(times) >= MAX_LOGIN_TRIES:
        return False, max(int(LOCKOUT_SECONDS - (now - min(times))), 0)
    return True, 0


def record_attempt(identifier: str, success: bool) -> None:
    if success:
        _attempts.pop(identifier, None)
    else:
        _attempts.setdefault(identifier, []).append(time.time())


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

def audit(
    kullanici: str,
    aksiyon: str,
    basarili: bool,
    ip: Optional[str] = None,
    detay: Optional[str] = None,
) -> None:
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (zaman,kullanici,aksiyon,ip,basarili,detay) "
                "VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(), kullanici, aksiyon,
                 ip or "unknown", int(basarili), detay),
            )
            conn.commit()
    except Exception as exc:
        log.error("Audit log hatası: %s", exc)


def get_audit_log(limit: int = 100, kullanici: Optional[str] = None) -> list[dict]:
    try:
        conn   = _get_conn()
        query  = "SELECT * FROM audit_log"
        params: list = []
        if kullanici:
            query += " WHERE kullanici=?"
            params.append(kullanici)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.error("get_audit_log hatası: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD.PY UYUMLU CORE FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def login_user(username: str, password: str, ip: str = "unknown") -> dict:
    """
    dashboard.py'nin beklediği login fonksiyonu.
    Başarılıysa access_token + refresh_token döner.
    """
    allowed, wait = check_rate_limit(ip)
    if not allowed:
        audit(username, "LOGIN_BLOCKED", False, ip)
        return {"success": False, "message": f"Çok fazla hatalı deneme. {wait} saniye bekleyin."}

    try:
        conn = _get_conn()
        row  = conn.execute(
            "SELECT id,username,password,role,status,display_name,avatar FROM users WHERE username=?",
            (username,),
        ).fetchone()
        conn.close()
    except Exception as exc:
        log.error("DB hatası: %s", exc)
        return {"success": False, "message": "Sunucu hatası."}

    # Timing attack koruması
    if not row:
        record_attempt(ip, False)
        audit(username, "LOGIN_FAILED", False, ip, "kullanıcı bulunamadı")
        if BCRYPT_AVAILABLE:
            bcrypt.checkpw(b"x", bcrypt.hashpw(b"x", bcrypt.gensalt()))
        return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

    uid, uname, pw_hash, role, status, display, avatar = (
        row["id"], row["username"], row["password"],
        row["role"], row["status"], row["display_name"], row["avatar"],
    )

    if not verify_password(password, pw_hash):
        record_attempt(ip, False)
        audit(uname, "LOGIN_FAILED", False, ip, "yanlış şifre")
        return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

    if status == "pending":
        audit(uname, "LOGIN_PENDING", False, ip)
        return {"success": False, "message": "Hesabınız henüz onaylanmadı."}
    if status == "rejected":
        audit(uname, "LOGIN_REJECTED", False, ip)
        return {"success": False, "message": "Hesabınız reddedildi."}
    if status not in ("active", "approved"):
        return {"success": False, "message": "Hesabınız aktif değil."}

    record_attempt(ip, True)
    access_token  = create_token(uid, uname, role)
    refresh_token = create_refresh_token(uid, ip)
    audit(uname, "LOGIN_SUCCESS", True, ip)

    return {
        "success":       True,
        "token":         access_token,
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "user": {
            "id":           uid,
            "username":     uname,
            "role":         role,
            "display_name": display or uname,
            "avatar":       avatar or "👤",
        },
    }


def register_user(
    username: str,
    password: str,
    display_name: str = "",
    email: str = "",
    role: str = "viewer",
) -> dict:
    """Yeni kullanıcı kaydı. İlk kullanıcı otomatik admin+active olur."""
    if len(password) < 6:
        return {"success": False, "message": "Şifre en az 6 karakter olmalı."}
    if len(username) < 3:
        return {"success": False, "message": "Kullanıcı adı en az 3 karakter olmalı."}

    # İlk kullanıcı → admin
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        is_first = count == 0

    status    = "active"  if is_first else "pending"
    real_role = "admin"   if is_first else role
    pw_hash   = hash_password(password)

    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username,password,role,status,display_name,email) "
                "VALUES (?,?,?,?,?,?)",
                (username, pw_hash, real_role, status, display_name or username, email),
            )
            conn.commit()
        audit(username, "REGISTER", True, detay=f"rol={real_role}, durum={status}")
        msg = "Kayıt başarılı! Giriş yapabilirsiniz." if is_first else \
              "Kayıt başarılı! Admin onayı bekleniyor."
        return {"success": True, "message": msg}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "Bu kullanıcı adı zaten alınmış."}
    except Exception as exc:
        log.error("register_user hatası: %s", exc)
        return {"success": False, "message": "Kayıt sırasında hata oluştu."}


def admin_exists() -> bool:
    """Sistemde en az bir admin var mı?"""
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND status IN ('active','approved')"
            ).fetchone()
            return row[0] > 0
    except Exception:
        return False


def change_password(username: str, old_password: str, new_password: str) -> dict:
    if len(new_password) < 6:
        return {"success": False, "message": "Yeni şifre en az 6 karakter olmalı."}
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT id,password FROM users WHERE username=?", (username,)
            ).fetchone()
        if not row:
            return {"success": False, "message": "Kullanıcı bulunamadı."}
        if not verify_password(old_password, row["password"]):
            audit(username, "PASSWORD_CHANGE_FAILED", False, detay="eski şifre yanlış")
            return {"success": False, "message": "Mevcut şifre hatalı."}
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET password=? WHERE username=?",
                (hash_password(new_password), username),
            )
            conn.commit()
        audit(username, "PASSWORD_CHANGED", True)
        return {"success": True, "message": "Şifre başarıyla değiştirildi."}
    except Exception as exc:
        log.error("change_password hatası: %s", exc)
        return {"success": False, "message": "Şifre değiştirme sırasında hata oluştu."}


# ── Kullanıcı yönetimi (admin fonksiyonları) ──────────────────────────────────

def get_all_users() -> list[dict]:
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT id,username,role,status,display_name,email,created_at,avatar "
                "FROM users ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.error("get_all_users hatası: %s", exc)
        return []


def get_pending_users() -> list[dict]:
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT id,username,role,status,display_name,email,created_at "
                "FROM users WHERE status='pending' ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.error("get_pending_users hatası: %s", exc)
        return []


def approve_user(username: str, approved_by: str = "admin") -> dict:
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET status='active' WHERE username=?", (username,)
            )
            conn.commit()
        audit(username, "USER_APPROVED", True, detay=f"onaylayan={approved_by}")
        return {"success": True, "message": f"{username} onaylandı."}
    except Exception as exc:
        log.error("approve_user hatası: %s", exc)
        return {"success": False, "message": "Onay sırasında hata oluştu."}


def reject_user(username: str, rejected_by: str = "admin") -> dict:
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET status='rejected' WHERE username=?", (username,)
            )
            conn.commit()
        audit(username, "USER_REJECTED", True, detay=f"reddeden={rejected_by}")
        return {"success": True, "message": f"{username} reddedildi."}
    except Exception as exc:
        log.error("reject_user hatası: %s", exc)
        return {"success": False, "message": "Red işlemi sırasında hata oluştu."}


def update_user_role(username: str, new_role: str, updated_by: str = "admin") -> dict:
    if new_role not in ("admin", "analyst", "viewer"):
        return {"success": False, "message": "Geçersiz rol. (admin/analyst/viewer)"}
    try:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET role=? WHERE username=?", (new_role, username)
            )
            conn.commit()
        audit(username, "ROLE_UPDATED", True, detay=f"yeni_rol={new_role}, yapan={updated_by}")
        return {"success": True, "message": f"{username} rolü '{new_role}' olarak güncellendi."}
    except Exception as exc:
        log.error("update_user_role hatası: %s", exc)
        return {"success": False, "message": "Rol güncelleme sırasında hata oluştu."}


def delete_user(username: str, deleted_by: str = "admin") -> dict:
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE username=?", (username,)
            ).fetchone()
            if not row:
                return {"success": False, "message": "Kullanıcı bulunamadı."}
            conn.execute("DELETE FROM users WHERE username=?", (username,))
            conn.execute(
                "DELETE FROM refresh_tokens WHERE user_id=?", (row["id"],)
            )
            conn.commit()
        audit(username, "USER_DELETED", True, detay=f"silen={deleted_by}")
        return {"success": True, "message": f"{username} silindi."}
    except Exception as exc:
        log.error("delete_user hatası: %s", exc)
        return {"success": False, "message": "Silme işlemi sırasında hata oluştu."}


def get_login_stats() -> dict:
    """
    Son 30 günlük login istatistikleri.
    dashboard.py'nin Yönetici sekmesinde kullanılır.
    """
    try:
        with _get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM users WHERE status IN ('active','approved')"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM users WHERE status='pending'"
            ).fetchone()[0]
            success_logins = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE aksiyon='LOGIN_SUCCESS'"
            ).fetchone()[0]
            failed_logins = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE aksiyon='LOGIN_FAILED'"
            ).fetchone()[0]
            # Son 30 gün günlük login
            daily = conn.execute("""
                SELECT DATE(zaman) as gun, COUNT(*) as sayi
                FROM audit_log
                WHERE aksiyon='LOGIN_SUCCESS'
                  AND zaman >= DATE('now','-30 days')
                GROUP BY DATE(zaman)
                ORDER BY gun
            """).fetchall()

        return {
            "toplam":         success_logins,  # dashboard uyumluluğu
            "basarili":       success_logins,
            "basarisiz":      failed_logins,
            "total_users":    total,
            "active_users":   active,
            "pending_users":  pending,
            "success_logins": success_logins,
            "failed_logins":  failed_logins,
            "daily_logins":   [{"date": r["gun"], "count": r["sayi"]} for r in daily],
        }
    except Exception as exc:
        log.error("get_login_stats hatası: %s", exc)
        return {
            "toplam": 0, "basarili": 0, "basarisiz": 0,
            "total_users": 0, "active_users": 0, "pending_users": 0,
            "success_logins": 0, "failed_logins": 0, "daily_logins": [],
        }


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI DEPENDS YARDIMCISI
# ══════════════════════════════════════════════════════════════════════════════

def require_role(*roles: str):
    """
    FastAPI endpoint'leri için kullanım:

        def get_current_user(credentials = Depends(HTTPBearer())):
            user = get_current_user_from_token(credentials.credentials)
            if not user:
                raise HTTPException(401, "Geçersiz token")
            return user

        @app.get("/admin")
        def admin_panel(user = Depends(require_role("admin"))):
            ...
    """
    def role_checker(user: dict = None):
        if not user or user.get("role") not in roles:
            raise PermissionError(f"Bu işlem için {roles} rolü gerekli.")
        return user
    return role_checker


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Test için geçici DB
    DB_PATH = Path("test_financeai.db")

    print("=== DB init ===")
    init_db()

    print("\n=== Kayıt testi ===")
    print(register_user("admin", "admin123", "Admin User"))
    print(register_user("testuser", "test123", "Test User"))
    print(register_user("admin", "xxx"))  # duplicate

    print("\n=== Login testi ===")
    res = login_user("admin", "admin123")
    print("Başarılı:", res["success"], "| Token:", str(res.get("access_token",""))[:30], "...")
    print("Başarısız:", login_user("admin", "yanlis")["message"])

    print("\n=== Refresh token testi ===")
    if res["success"]:
        rt  = res["refresh_token"]
        new = use_refresh_token(rt)
        print("Yeni access token:", str(new["access_token"])[:30], "..." if new else "HATA")
        print("Eski refresh tekrar:", use_refresh_token(rt))  # None olmalı

    print("\n=== Rate limit testi ===")
    for i in range(7):
        r = login_user("admin", "yanlis", ip="10.0.0.1")
        print(f"  {i+1}: {r['message'][:50]}")

    print("\n=== Kullanıcı yönetimi ===")
    print("Tüm kullanıcılar:", [(u["username"], u["status"]) for u in get_all_users()])
    print("Bekleyenler:", get_pending_users())
    print("Onayla:", approve_user("testuser"))
    print("Stats:", get_login_stats())

    # Temizlik
    Path("test_financeai.db").unlink(missing_ok=True)
    print("\n✓ Tüm testler tamamlandı.")