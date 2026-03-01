"""
FinanceAI — İlk Kurulum (setup.py)
════════════════════════════════════

Bu script'i sadece BİR KEZ çalıştırın:
    python setup.py

Ne yapar:
- Admin kullanıcı adı ve şifresi sorار
- Güvenli şekilde veritabanına kaydeder
- Tekrar çalıştırırsanız mevcut admini değiştirme seçeneği sunar
"""

import sqlite3
import hashlib
import secrets
import os
import sys
import getpass
from datetime import datetime

DB_PATH = "C:/financeai/data/users_auth.db"


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()


def ensure_db_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def check_admin_exists():
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except:
        return False


def create_tables():
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


def create_admin(username, password, email, display_name):
    salt    = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Varsa eski admini sil
    c.execute("DELETE FROM users WHERE role = 'admin'")

    c.execute("""
        INSERT INTO users
            (username, email, password_hash, salt, role,
             display_name, avatar, status, created_at, approved_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username.strip().lower(),
        email.strip().lower(),
        pw_hash,
        salt,
        "admin",
        display_name.strip() or username,
        "👨‍💼",
        "active",
        datetime.now().isoformat(),
        "setup"
    ))
    conn.commit()
    conn.close()


def validate_password(pw):
    """Şifre güvenlik kontrolü"""
    if len(pw) < 8:
        return False, "En az 8 karakter olmalı."
    if not any(c.isupper() for c in pw):
        return False, "En az 1 büyük harf içermeli."
    if not any(c.isdigit() for c in pw):
        return False, "En az 1 rakam içermeli."
    return True, "OK"


def print_banner():
    print("\n" + "═" * 55)
    print("  💎  FinanceAI — İlk Kurulum Sihirbazı")
    print("═" * 55)
    print()


def print_success(username):
    print()
    print("═" * 55)
    print("  ✅  Admin hesabı başarıyla oluşturuldu!")
    print("═" * 55)
    print(f"  Kullanıcı Adı : {username}")
    print(f"  Rol           : Admin (Tam Yetki)")
    print(f"  Veritabanı    : {DB_PATH}")
    print()
    print("  🚀 Uygulamayı başlatmak için:")
    print("     streamlit run dashboard.py")
    print()
    print("  ⚠️  Şifrenizi kimseyle paylaşmayın!")
    print("═" * 55)
    print()


def main():
    print_banner()

    # DB klasörü oluştur
    ensure_db_dir()

    # Mevcut admin var mı?
    if check_admin_exists():
        print("⚠️  Sistemde zaten bir admin hesabı mevcut.")
        print()
        cevap = input("   Mevcut admini sıfırlamak istiyor musunuz? (evet/hayır): ").strip().lower()
        if cevap not in ["evet", "e", "yes", "y"]:
            print()
            print("   ℹ️  İşlem iptal edildi. Mevcut admin korundu.")
            print()
            sys.exit(0)
        print()

    # Tablo oluştur
    create_tables()

    print("  Admin bilgilerini girin:")
    print("  (Şifre ekranda görünmez — güvenli giriş)\n")

    # ── KULLANICI ADI ──
    while True:
        username = input("  👤 Admin kullanıcı adı: ").strip()
        if len(username) < 3:
            print("     ❌ En az 3 karakter olmalı.\n")
            continue
        if " " in username:
            print("     ❌ Boşluk kullanılamaz.\n")
            continue
        break

    # ── GÖRÜNEN AD ──
    display_name = input(f"  📛 Görünen ad (boş bırakılırsa '{username}'): ").strip()
    if not display_name:
        display_name = username

    # ── E-POSTA ──
    while True:
        email = input("  📧 E-posta adresi: ").strip()
        if "@" not in email or "." not in email:
            print("     ❌ Geçerli bir e-posta girin.\n")
            continue
        break

    # ── ŞİFRE ──
    print()
    print("  Şifre kuralları: Min. 8 karakter, 1 büyük harf, 1 rakam")
    while True:
        try:
            password = getpass.getpass("  🔑 Şifre: ")
        except Exception:
            password = input("  🔑 Şifre: ")  # getpass çalışmazsa fallback

        valid, msg = validate_password(password)
        if not valid:
            print(f"     ❌ {msg}\n")
            continue

        try:
            password2 = getpass.getpass("  🔑 Şifre onayı: ")
        except Exception:
            password2 = input("  🔑 Şifre onayı: ")

        if password != password2:
            print("     ❌ Şifreler eşleşmiyor.\n")
            continue

        break

    # ── KAYDET ──
    print()
    print("  Kaydediliyor...", end=" ")
    create_admin(username, password, email, display_name)
    print("✅")

    print_success(username)


if __name__ == "__main__":
    main()