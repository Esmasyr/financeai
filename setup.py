"""
FinSight — setup.py
İlk kurulum: tabloları oluşturur + admin hesabı açar.

Çalıştır: python setup.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

def main():
    print("=" * 50)
    print("  FinSight Kurulum")
    print("=" * 50)

    # 1. Tablolar
    print("\n[1/3] Veritabanı tabloları oluşturuluyor...")
    try:
        from database import init_db
        init_db()
        print("      ✅ Tablolar hazır.")
    except Exception as e:
        print(f"      ❌ database.py hatası: {e}")
        sys.exit(1)

    # 2. Auth tablosu
    try:
        from auth import init_db as auth_init_db, admin_exists
        auth_init_db()
        print("      ✅ Auth tablosu hazır.")
    except Exception as e:
        print(f"      ❌ auth.py hatası: {e}")
        sys.exit(1)

    # 3. Admin hesabı
    print("\n[2/3] Admin hesabı oluşturuluyor...")
    if admin_exists():
        print("      ℹ️  Zaten bir admin hesabı var, atlanıyor.")
    else:
        print()
        username     = input("      Admin kullanıcı adı : ").strip()
        email        = input("      Admin e-posta       : ").strip()
        display_name = input("      Ad Soyad            : ").strip()
        password     = input("      Şifre (min 6 kar.)  : ").strip()

        from auth import register_user, approve_user, update_user_role, get_all_users
        result = register_user(username, email, password, display_name, role="admin")
        if not result["success"]:
            print(f"      ❌ {result['message']}")
            sys.exit(1)

        # Yeni kaydı bul ve direkt active yap
        users = get_all_users()
        new_user = next((u for u in users if u["username"] == username.lower()), None)
        if new_user:
            approve_user(new_user["id"], "setup")
            update_user_role(new_user["id"], "admin")
            print(f"      ✅ Admin hesabı oluşturuldu: {username}")
        else:
            print("      ❌ Kullanıcı oluşturulamadı.")
            sys.exit(1)

    # 4. Paket kontrolü
    print("\n[3/3] Gerekli paketler kontrol ediliyor...")
    required = {
        "streamlit":  "streamlit",
        "pandas":     "pandas",
        "numpy":      "numpy",
        "plotly":     "plotly",
        "requests":   "requests",
        "fastapi":    "fastapi",
        "uvicorn":    "uvicorn[standard]",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
            print(f"      ✅ {mod}")
        except ImportError:
            print(f"      ❌ {mod} — eksik")
            missing.append(pkg)

    if missing:
        print(f"\n      Eksik paketleri yükle:")
        print(f"      pip install {' '.join(missing)}")
    else:
        print("\n      Tüm paketler kurulu.")

    print("\n" + "=" * 50)
    print("  Kurulum tamamlandı!")
    print()
    print("  Uygulamayı başlatmak için:")
    print("    Terminal 1: uvicorn src.api:app --reload --port 8000")
    print("    Terminal 2: streamlit run src/app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()