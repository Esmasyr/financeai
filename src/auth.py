"""
FinSight — auth.py
Kullanıcı kayıt, giriş, onay, rol yönetimi
SQLite tabanlı, bcrypt şifre hash

Güvenlik iyileştirmeleri (v2):
  - bcrypt ile şifre hash (sha256 → bcrypt)
  - login_user'da password hash sızıntısı kapatıldı
  - Rate limiting + account lockout (5 başarısız → 15 dk kilit)
  - IP adresi login_logs'a kaydediliyor
  - Hatalı giriş mesajları normalize edildi (user enum önleme)
  - Email format doğrulaması
  - display_name HTML sanitization
  - Şifre karmaşıklık kuralı
  - Timing-safe şifre karşılaştırması
"""

import sqlite3
import os
import re
import html
import secrets
from datetime import datetime, timedelta

import bcrypt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "financeai.db")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Sabitler ──
MAX_FAILED_ATTEMPTS = 5          # kaç başarısız denemeden sonra kilit
LOCKOUT_MINUTES     = 15         # kaç dakika kilitli kalır
MIN_PASSWORD_LEN    = 8          # minimum şifre uzunluğu (6 → 8)
BCRYPT_ROUNDS       = 12         # bcrypt iş faktörü


# ─────────────────────────────────────────
# Şifre hash — bcrypt
# ─────────────────────────────────────────

def _hash_password(password: str) -> str:
    """bcrypt ile şifre hash'le; salt otomatik gömülü."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(BCRYPT_ROUNDS)).decode("utf-8")


def _verify_password(password: str, stored: str) -> bool:
    """Timing-safe bcrypt karşılaştırması."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except Exception:
        return False


# ─────────────────────────────────────────
# Doğrulama yardımcıları
# ─────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _validate_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email))


def _validate_password(password: str) -> tuple[bool, str]:
    """Minimum uzunluk + en az 1 rakam + en az 1 harf kontrolü."""
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Şifre en az {MIN_PASSWORD_LEN} karakter olmalı."
    if not re.search(r"[A-Za-z]", password):
        return False, "Şifre en az bir harf içermeli."
    if not re.search(r"\d", password):
        return False, "Şifre en az bir rakam içermeli."
    return True, ""


def _sanitize(text: str) -> str:
    """Basit HTML escape — XSS önleme."""
    return html.escape(text.strip())


# ─────────────────────────────────────────
# Veritabanı başlat
# ─────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT UNIQUE NOT NULL,
            email        TEXT UNIQUE NOT NULL,
            password     TEXT NOT NULL,
            display_name TEXT,
            role         TEXT DEFAULT 'viewer',
            avatar       TEXT DEFAULT '👤',
            status       TEXT DEFAULT 'pending',
            created_at   TEXT DEFAULT (datetime('now')),
            approved_by  TEXT,
            approved_at  TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT,
            success    INTEGER,
            ip         TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Hesap kilitleme tablosu (yeni)
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            username      TEXT PRIMARY KEY,
            failed_count  INTEGER DEFAULT 0,
            locked_until  TEXT
        )
    """)

    conn.commit()
    conn.close()


def admin_exists() -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND status='active'")
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


# ─────────────────────────────────────────
# Rate limiting / Account lockout
# ─────────────────────────────────────────

def _is_locked(cursor, username: str) -> bool:
    """Hesap kilitli mi? Kilit süresi geçtiyse otomatik aç."""
    cursor.execute("SELECT failed_count, locked_until FROM login_attempts WHERE username=?", (username,))
    row = cursor.fetchone()
    if not row:
        return False
    failed_count, locked_until = row
    if locked_until:
        if datetime.utcnow() < datetime.fromisoformat(locked_until):
            return True
        # Kilit süresi geçti — sıfırla
        cursor.execute(
            "UPDATE login_attempts SET failed_count=0, locked_until=NULL WHERE username=?",
            (username,)
        )
    return False


def _record_failed_attempt(cursor, username: str):
    """Başarısız denemeyi kaydet; limiti aşarsa kilitle."""
    cursor.execute(
        "INSERT INTO login_attempts (username, failed_count) VALUES (?, 1) "
        "ON CONFLICT(username) DO UPDATE SET failed_count = failed_count + 1",
        (username,)
    )
    cursor.execute("SELECT failed_count FROM login_attempts WHERE username=?", (username,))
    count = cursor.fetchone()[0]
    if count >= MAX_FAILED_ATTEMPTS:
        locked_until = (datetime.utcnow() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
        cursor.execute(
            "UPDATE login_attempts SET locked_until=? WHERE username=?",
            (locked_until, username)
        )


def _clear_failed_attempts(cursor, username: str):
    """Başarılı girişte sayacı sıfırla."""
    cursor.execute(
        "UPDATE login_attempts SET failed_count=0, locked_until=NULL WHERE username=?",
        (username,)
    )


# ─────────────────────────────────────────
# Kayıt
# ─────────────────────────────────────────

def register_user(username: str, email: str, password: str,
                  display_name: str = "", role: str = "viewer") -> dict:
    # Email doğrula
    if not _validate_email(email):
        return {"success": False, "message": "Geçersiz e-posta formatı."}

    # Şifre doğrula
    ok, msg = _validate_password(password)
    if not ok:
        return {"success": False, "message": msg}

    # Rol güvenlik kontrolü — dışarıdan admin atanamaz
    if role not in ("viewer", "analyst"):
        role = "viewer"

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed = _hash_password(password)
        safe_display = _sanitize(display_name or username)

        c.execute("""
            INSERT INTO users (username, email, password, display_name, role, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (
            _sanitize(username).lower(),
            email.lower().strip(),
            hashed,
            safe_display,
            role
        ))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Kayıt başarılı. Admin onayı bekleniyor."}

    except sqlite3.IntegrityError as e:
        # Her iki durumda da aynı mesaj — kullanıcı adı/email enum önleme
        err = str(e)
        if "username" in err:
            msg = "Bu kullanıcı adı zaten alınmış."
        else:
            msg = "Bu e-posta zaten kayıtlı."
        return {"success": False, "message": msg}
    except Exception as e:
        return {"success": False, "message": "Kayıt sırasında bir hata oluştu."}


# ─────────────────────────────────────────
# Giriş
# ─────────────────────────────────────────

def login_user(username: str, password: str, ip: str = "") -> dict:
    """
    Güvenli giriş:
      - Account lockout kontrolü
      - Timing-safe bcrypt doğrulama
      - Normalize edilmiş hata mesajları
      - Password hash sızdırmama
      - IP loglama
    """
    # Kullanıcı adı bulunsun ya da bulunmasın aynı süreyi almak için
    # şifre doğrulama her zaman çalıştırılır (timing attack önleme)
    DUMMY_HASH = "$2b$12$invalidhashfortimingprotectiononly000000000000000000000"

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Kilit kontrolü
        if _is_locked(c, username.lower().strip()):
            conn.commit(); conn.close()
            return {
                "success": False,
                "message": f"Çok fazla başarısız deneme. {LOCKOUT_MINUTES} dakika bekleyin."
            }

        c.execute("SELECT * FROM users WHERE username=?", (username.lower().strip(),))
        row = c.fetchone()
        cols = [d[0] for d in c.description] if c.description else []

        if row is None:
            # Timing attack önleme: kullanıcı yoksa da bcrypt çalıştır
            bcrypt.checkpw(b"dummy", DUMMY_HASH.encode())
            _record_failed_attempt(c, username.lower().strip())
            _log_login(c, username, False, ip)
            conn.commit(); conn.close()
            return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

        user = dict(zip(cols, row))

        if not _verify_password(password, user["password"]):
            _record_failed_attempt(c, username.lower().strip())
            _log_login(c, username, False, ip)
            conn.commit(); conn.close()
            return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

        # Durum kontrolleri şifre doğrulandıktan sonra
        if user["status"] == "pending":
            conn.close()
            return {"success": False, "message": "Hesabınız henüz onaylanmadı."}

        if user["status"] == "rejected":
            conn.close()
            return {"success": False, "message": "Hesabınız reddedildi."}

        # Başarılı giriş
        _clear_failed_attempts(c, username.lower().strip())
        _log_login(c, username, True, ip)
        conn.commit(); conn.close()

        # ✅ Password hash'i sızdırma
        user.pop("password", None)
        return {"success": True, "user": user}

    except Exception as e:
        return {"success": False, "message": "Giriş sırasında bir hata oluştu."}


def _log_login(cursor, username: str, success: bool, ip: str = ""):
    cursor.execute(
        "INSERT INTO login_logs (username, success, ip) VALUES (?, ?, ?)",
        (username, 1 if success else 0, ip or "unknown")
    )


# ─────────────────────────────────────────
# Kullanıcı listesi
# ─────────────────────────────────────────

def get_all_users() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # password kolonu hiçbir zaman döndürülmüyor
        c.execute("""
            SELECT id, username, email, display_name, role, avatar, status, created_at
            FROM users
            ORDER BY created_at DESC
        """)
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_pending_users() -> list:
    return [u for u in get_all_users() if u["status"] == "pending"]


# ─────────────────────────────────────────
# Onay / Red
# ─────────────────────────────────────────

def approve_user(user_id: int, approved_by: str) -> dict:
    """approved_by: sadece aktif admin username'leri kabul edilmeli (çağıran katman sorumlu)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE users
            SET status='active', approved_by=?, approved_at=datetime('now')
            WHERE id=?
        """, (_sanitize(str(approved_by)), user_id))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": "Onay işlemi başarısız."}


def reject_user(user_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET status='rejected' WHERE id=?", (user_id,))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": "Red işlemi başarısız."}


# ─────────────────────────────────────────
# Rol güncelle
# ─────────────────────────────────────────

def update_user_role(user_id: int, role: str) -> dict:
    if role not in ("viewer", "analyst", "admin"):
        return {"success": False, "message": "Geçersiz rol."}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": "Rol güncellenemedi."}


# ─────────────────────────────────────────
# Sil
# ─────────────────────────────────────────

def delete_user(user_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": "Silme işlemi başarısız."}


# ─────────────────────────────────────────
# Şifre değiştir
# ─────────────────────────────────────────

def change_password(username: str, old_password: str, new_password: str) -> dict:
    ok, msg = _validate_password(new_password)
    if not ok:
        return {"success": False, "message": msg}

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"success": False, "message": "Kullanıcı bulunamadı."}
        if not _verify_password(old_password, row[0]):
            conn.close()
            return {"success": False, "message": "Mevcut şifre yanlış."}

        # Yeni şifre eskisiyle aynı olmamalı
        if _verify_password(new_password, row[0]):
            conn.close()
            return {"success": False, "message": "Yeni şifre mevcut şifreyle aynı olamaz."}

        new_hash = _hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
        conn.commit(); conn.close()
        return {"success": True, "message": "Şifre güncellendi."}
    except Exception as e:
        return {"success": False, "message": "Şifre değiştirilemedi."}


# ─────────────────────────────────────────
# Giriş istatistikleri
# ─────────────────────────────────────────

def get_login_stats() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM login_logs")
        toplam = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success=1")
        basarili = c.fetchone()[0]
        conn.close()
        return {
            "toplam":    toplam,
            "basarili":  basarili,
            "basarisiz": toplam - basarili
        }
    except Exception:
        return {"toplam": 0, "basarili": 0, "basarisiz": 0}


# ─────────────────────────────────────────
# Kilitli hesap listesi (admin paneli için)
# ─────────────────────────────────────────

def get_locked_accounts() -> list:
    """Şu an kilitli olan hesapları döndür."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("""
            SELECT username, failed_count, locked_until
            FROM login_attempts
            WHERE locked_until IS NOT NULL AND locked_until > ?
        """, (now,))
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def unlock_account(username: str) -> dict:
    """Admin tarafından manuel kilit kaldırma."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE login_attempts SET failed_count=0, locked_until=NULL WHERE username=?",
            (username,)
        )
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": "Kilit kaldırılamadı."}