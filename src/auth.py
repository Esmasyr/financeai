"""
FinSight — auth.py
Kullanıcı kayıt, giriş, onay, rol yönetimi
SQLite tabanlı, bcrypt şifre hash
"""

import sqlite3
import hashlib
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "financeai.db")

os.makedirs(DATA_DIR, exist_ok=True)


# ── Şifre hash (basit sha256 + salt, bcrypt yoksa) ──
def _hash_password(password: str, salt: str = "") -> str:
    if not salt:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":", 1)
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == hashed
    except Exception:
        return False


# ── Veritabanı başlat ──
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


# ── Kayıt ──
def register_user(username: str, email: str, password: str,
                  display_name: str = "", role: str = "viewer") -> dict:
    if len(password) < 6:
        return {"success": False, "message": "Şifre en az 6 karakter olmalı."}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        hashed = _hash_password(password)
        c.execute("""
            INSERT INTO users (username, email, password, display_name, role, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (username.lower().strip(), email.lower().strip(), hashed,
              display_name or username, role))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Kayıt başarılı. Admin onayı bekleniyor."}
    except sqlite3.IntegrityError as e:
        msg = "Bu kullanıcı adı zaten alınmış." if "username" in str(e) else "Bu e-posta zaten kayıtlı."
        return {"success": False, "message": msg}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Giriş ──
def login_user(username: str, password: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username.lower().strip(),))
        row = c.fetchone()
        cols = [d[0] for d in c.description]

        if row is None:
            _log_login(c, username, False)
            conn.commit(); conn.close()
            return {"success": False, "message": "Kullanıcı bulunamadı."}

        user = dict(zip(cols, row))

        if not _verify_password(password, user["password"]):
            _log_login(c, username, False)
            conn.commit(); conn.close()
            return {"success": False, "message": "Şifre yanlış."}

        if user["status"] == "pending":
            conn.close()
            return {"success": False, "message": "Hesabınız henüz onaylanmadı."}

        if user["status"] == "rejected":
            conn.close()
            return {"success": False, "message": "Hesabınız reddedildi."}

        _log_login(c, username, True)
        conn.commit(); conn.close()
        return {"success": True, "user": user}
    except Exception as e:
        return {"success": False, "message": str(e)}


def _log_login(cursor, username, success):
    cursor.execute(
        "INSERT INTO login_logs (username, success) VALUES (?, ?)",
        (username, 1 if success else 0)
    )


# ── Kullanıcı listesi ──
def get_all_users() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id,username,email,display_name,role,avatar,status,created_at FROM users ORDER BY created_at DESC")
        cols = [d[0] for d in c.description]
        rows = [dict(zip(cols, r)) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def get_pending_users() -> list:
    return [u for u in get_all_users() if u["status"] == "pending"]


# ── Onay / Red ──
def approve_user(user_id: int, approved_by: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            UPDATE users SET status='active', approved_by=?, approved_at=datetime('now')
            WHERE id=?
        """, (approved_by, user_id))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}


def reject_user(user_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET status='rejected' WHERE id=?", (user_id,))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Rol güncelle ──
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
        return {"success": False, "message": str(e)}


# ── Sil ──
def delete_user(user_id: int) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit(); conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Şifre değiştir ──
def change_password(username: str, old_password: str, new_password: str) -> dict:
    if len(new_password) < 6:
        return {"success": False, "message": "Yeni şifre en az 6 karakter olmalı."}
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
        new_hash = _hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
        conn.commit(); conn.close()
        return {"success": True, "message": "Şifre güncellendi."}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Giriş istatistikleri ──
def get_login_stats() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM login_logs")
        toplam = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success=1")
        basarili = c.fetchone()[0]
        conn.close()
        return {"toplam": toplam, "basarili": basarili, "basarisiz": toplam - basarili}
    except Exception:
        return {"toplam": 0, "basarili": 0, "basarisiz": 0}   