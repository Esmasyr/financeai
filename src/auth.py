"""
FinanceAI — Kimlik Doğrulama Modülü (auth.py) v5.2
Admin setup.py ile kurulur — kodda şifre yok.
"""

import sqlite3
import hashlib
import secrets
import os
from datetime import datetime

DB_PATH = "C:/financeai/data/users_auth.db"


def init_db():
    """Tabloları oluştur. Admin setup.py ile ayrıca kurulur."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'viewer',
            display_name  TEXT,
            avatar        TEXT DEFAULT '👤',
            status        TEXT DEFAULT 'pending',
            created_at    TEXT,
            last_login    TEXT,
            approved_by   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT,
            success   INTEGER,
            ip_note   TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def admin_exists() -> bool:
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND status='active'")
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except:
        return False


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return _hash_password(password, salt) == stored_hash


def register_user(username, email, password, display_name, role="viewer") -> dict:
    if len(username) < 3:
        return {"success": False, "message": "Kullanıcı adı en az 3 karakter olmalı."}
    if len(password) < 6:
        return {"success": False, "message": "Şifre en az 6 karakter olmalı."}
    if "@" not in email:
        return {"success": False, "message": "Geçerli bir email girin."}

    avatar_map = {"admin": "👨‍💼", "analyst": "📊", "viewer": "👁️"}

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        salt = secrets.token_hex(16)
        pw_hash = _hash_password(password, salt)

        c.execute("""
            INSERT INTO users
                (username, email, password_hash, salt, role,
                 display_name, avatar, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, (
            username.strip().lower(),
            email.strip().lower(),
            pw_hash,
            salt,
            role,
            display_name.strip() or username,
            avatar_map.get(role, "👤"),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()
        return {"success": True, "message": "Kayıt başarılı!", "pending": True}

    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"success": False, "message": "Bu kullanıcı adı zaten kullanılıyor."}
        elif "email" in str(e):
            return {"success": False, "message": "Bu email zaten kayıtlı."}
        return {"success": False, "message": f"Kayıt hatası: {str(e)}"}


# ✅ DÜZELTİLMİŞ LOGIN (ADMIN BYPASS VAR)
def login_user(username, password) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
        user = c.fetchone()

        if not user:
            _log_login(conn, username, False)
            conn.close()
            return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

        if not verify_password(password, user["salt"], user["password_hash"]):
            _log_login(conn, username, False)
            conn.close()
            return {"success": False, "message": "Kullanıcı adı veya şifre hatalı."}

        # 🔥 ADMIN HER ZAMAN GİREBİLİR
        if user["role"] != "admin":
            if user["status"] == "pending":
                conn.close()
                return {"success": False, "message": "⏳ Hesabınız yönetici onayını bekliyor."}
            if user["status"] == "rejected":
                conn.close()
                return {"success": False, "message": "❌ Hesabınız onaylanmadı. Yönetici ile iletişime geçin."}

        c.execute("UPDATE users SET last_login = ? WHERE username = ?",
                  (datetime.now().isoformat(), username))
        _log_login(conn, username, True)
        conn.commit()

        user_dict = dict(user)
        conn.close()
        return {"success": True, "user": user_dict}

    except Exception as e:
        return {"success": False, "message": f"Sistem hatası: {str(e)}"}


def _log_login(conn, username, success):
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO login_history (username, success, timestamp) VALUES (?,?,?)",
            (username, 1 if success else 0, datetime.now().isoformat())
        )
    except:
        pass


def get_all_users() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, username, email, role, display_name,
                   avatar, status, created_at, last_login, approved_by
            FROM users
            ORDER BY
                CASE status
                    WHEN 'pending' THEN 0
                    WHEN 'active' THEN 1
                    ELSE 2
                END,
                created_at DESC
        """)
        users = [dict(row) for row in c.fetchall()]
        conn.close()
        return users
    except:
        return []


def get_pending_users() -> list:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, username, email, role, display_name, created_at
            FROM users
            WHERE status='pending'
            ORDER BY created_at
        """)
        users = [dict(row) for row in c.fetchall()]
        conn.close()
        return users
    except:
        return []


def approve_user(user_id, approved_by) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE users SET status='active', approved_by=? WHERE id=?",
            (approved_by, user_id)
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False


def reject_user(user_id) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE users SET status='rejected' WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def update_user_role(user_id, new_role) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        avatar_map = {"admin": "👨‍💼", "analyst": "📊", "viewer": "👁️"}
        c.execute(
            "UPDATE users SET role=?, avatar=? WHERE id=?",
            (new_role, avatar_map.get(new_role, "👤"), user_id)
        )
        conn.commit()
        conn.close()
        return True
    except:
        return False


def delete_user(user_id) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return True
    except:
        return False


def get_login_stats() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*), SUM(success), COUNT(*)-SUM(success) FROM login_history")
        row = c.fetchone()
        conn.close()
        return {
            "toplam": row[0] or 0,
            "basarili": row[1] or 0,
            "basarisiz": row[2] or 0
        }
    except:
        return {"toplam": 0, "basarili": 0, "basarisiz": 0}


def change_password(username, old_password, new_password) -> dict:
    if len(new_password) < 6:
        return {"success": False, "message": "Yeni şifre en az 6 karakter olmalı."}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()

        if not user or not verify_password(old_password, user["salt"], user["password_hash"]):
            conn.close()
            return {"success": False, "message": "Mevcut şifre hatalı."}

        new_salt = secrets.token_hex(16)
        new_hash = _hash_password(new_password, new_salt)

        c.execute(
            "UPDATE users SET password_hash=?, salt=? WHERE username=?",
            (new_hash, new_salt, username)
        )

        conn.commit()
        conn.close()

        return {"success": True, "message": "Şifre güncellendi."}

    except Exception as e:
        return {"success": False, "message": str(e)}


init_db()