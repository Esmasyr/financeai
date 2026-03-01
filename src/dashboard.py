"""
FinSight v5.2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from auth import (
    login_user, register_user, get_all_users, get_pending_users,
    approve_user, reject_user, update_user_role, delete_user,
    get_login_stats, change_password, init_db, admin_exists
)

init_db()

st.set_page_config(
    page_title="FinSight",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════

for key, default in [
    ("logged_in", False), ("username", ""), ("role", ""),
    ("display_name", ""), ("avatar", "👤"), ("user_id", None),
    ("dark_mode", True), ("auth_tab", "login"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════
# TEMA
# ═══════════════════════════════════════════

def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg_base": "#06090F", "bg_card": "#0C1118", "bg_card2": "#101820",
            "text_primary": "#E8EDF5", "text_secondary": "#A8B4C0", "text_muted": "#5A6A7A",
            "border": "rgba(201,168,76,0.12)", "border_bright": "rgba(201,168,76,0.28)",
            "sidebar_bg": "linear-gradient(180deg,#08101A 0%,#050A12 100%)",
            "plot_bg": "rgba(12,17,24,0.8)", "input_bg": "#0C1118",
            "shadow_opacity": "0.45", "toggle_icon": "☀️",
        }
    else:
        return {
            "bg_base": "#F0F4FA", "bg_card": "#FFFFFF", "bg_card2": "#E8EEF7",
            "text_primary": "#1A2332", "text_secondary": "#2D3F55", "text_muted": "#5A6A7A",
            "border": "rgba(26,35,50,0.15)", "border_bright": "rgba(140,100,30,0.45)",
            "sidebar_bg": "linear-gradient(180deg,#FFFFFF 0%,#E8EEF7 100%)",
            "plot_bg": "rgba(255,255,255,0.97)", "input_bg": "#FFFFFF",
            "shadow_opacity": "0.12", "toggle_icon": "🌙",
        }

T = get_theme()

GOLD   = "#C9A84C"
CYAN   = "#00D4FF"
RED    = "#FF4560"
GREEN  = "#00E396"
ORANGE = "#FF6B35"
PURPLE = "#8B5CF6"

# ═══════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {{
    --bg-base:{T['bg_base']}; --bg-card:{T['bg_card']}; --bg-card2:{T['bg_card2']};
    --gold:{GOLD}; --cyan:{CYAN}; --red:{RED}; --green:{GREEN};
    --text-primary:{T['text_primary']}; --text-secondary:{T['text_secondary']};
    --text-muted:{T['text_muted']}; --border:{T['border']}; --border-bright:{T['border_bright']};
    --shadow:0 4px 24px rgba(0,0,0,{T['shadow_opacity']}); --input-bg:{T['input_bg']};
}}

html,body,[class*="css"] {{
    background-color:var(--bg-base) !important;
    color:var(--text-primary) !important;
    font-family:'DM Sans',sans-serif !important;
}}

section[data-testid="stSidebar"] {{
    background:{T['sidebar_bg']} !important;
    border-right:1px solid var(--border) !important;
    width:240px !important;
}}
section[data-testid="stSidebar"] * {{ color:var(--text-primary) !important; }}

[data-testid="metric-container"] {{
    background:linear-gradient(135deg,var(--bg-card),var(--bg-card2)) !important;
    border:1px solid var(--border) !important; border-radius:14px !important;
    padding:18px 20px !important; box-shadow:var(--shadow) !important;
    transition:border-color .2s,transform .15s !important;
}}
[data-testid="metric-container"]:hover {{
    border-color:var(--border-bright) !important; transform:translateY(-2px) !important;
}}
[data-testid="stMetricValue"] {{
    font-family:'Syne',sans-serif !important; font-size:1.8rem !important;
    font-weight:800 !important;
    color:{"#C9A84C" if st.session_state.dark_mode else "#8B6820"} !important;
}}
[data-testid="stMetricLabel"] {{
    font-family:'DM Mono',monospace !important; font-size:.62rem !important;
    color:var(--text-muted) !important; text-transform:uppercase !important;
    letter-spacing:.18em !important;
}}

/* TABS */
.stTabs [data-baseweb="tab-list"] {{
    background:var(--bg-card) !important; border-radius:10px !important;
    padding:4px !important; border:1px solid var(--border) !important;
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:var(--text-muted) !important;
    font-family:'DM Mono',monospace !important; font-size:.7rem !important;
    text-transform:uppercase !important; letter-spacing:.1em !important;
    border-radius:7px !important; border:none !important; padding:7px 14px !important;
}}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,rgba(201,168,76,.18),rgba(201,168,76,.04)) !important;
    color:{GOLD} !important; border:1px solid rgba(201,168,76,.25) !important;
}}

/* RADIO/NAV */
div[data-testid="stRadio"]>label {{ display:none; }}
div[data-testid="stRadio"]>div {{ display:flex; flex-direction:column; gap:1px; }}
div[data-testid="stRadio"]>div>label {{
    display:flex !important; align-items:center; padding:9px 12px !important;
    border-radius:8px !important; font-family:'DM Sans',sans-serif !important;
    font-size:.82rem !important; color:var(--text-muted) !important; cursor:pointer;
    transition:all .15s ease; border:1px solid transparent; margin:0 !important;
}}
div[data-testid="stRadio"]>div>label:hover {{
    background:rgba(201,168,76,.07) !important; color:{GOLD} !important;
    border-color:rgba(201,168,76,.12) !important;
}}
div[data-testid="stRadio"]>div>label[aria-checked="true"] {{
    background:linear-gradient(135deg,rgba(201,168,76,.14),rgba(201,168,76,.04)) !important;
    color:{GOLD} !important; border-color:rgba(201,168,76,.28) !important; font-weight:600 !important;
}}
div[data-testid="stRadio"] input[type="radio"] {{ display:none !important; }}
div[data-testid="stRadio"]>div>label>div:first-child {{ display:none !important; }}

/* BUTONLAR */
.stButton>button {{
    background:linear-gradient(135deg,{GOLD},#A07830) !important;
    color:#06090F !important; border:none !important; border-radius:8px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    letter-spacing:.04em !important; padding:8px 20px !important;
    transition:all .2s !important; box-shadow:0 2px 12px rgba(201,168,76,.2) !important;
}}
.stButton>button:hover {{
    opacity:.88 !important; transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(201,168,76,.3) !important;
}}
.btn-ghost>button {{
    background:transparent !important; color:var(--text-muted) !important;
    border:1px solid var(--border) !important; box-shadow:none !important;
    font-size:.78rem !important;
}}
.btn-ghost>button:hover {{
    border-color:rgba(255,69,96,.4) !important; color:{RED} !important;
    box-shadow:none !important; transform:none !important;
    background:rgba(255,69,96,.06) !important;
}}
.btn-green>button {{
    background:linear-gradient(135deg,{GREEN},#00B377) !important;
    color:#001A0F !important; font-size:.75rem !important; padding:5px 14px !important;
}}
.btn-red>button {{
    background:linear-gradient(135deg,{RED},#CC2040) !important;
    font-size:.75rem !important; padding:5px 14px !important;
}}
.btn-purple>button {{
    background:linear-gradient(135deg,{PURPLE},#6D3FD0) !important;
    font-size:.75rem !important; padding:5px 14px !important;
}}
.btn-dim>button {{
    background:rgba(201,168,76,.08) !important; color:{GOLD} !important;
    border:1px solid rgba(201,168,76,.2) !important; box-shadow:none !important;
    font-size:.78rem !important; font-family:'DM Mono',monospace !important;
}}
.btn-dim>button:hover {{
    background:rgba(201,168,76,.14) !important;
    border-color:rgba(201,168,76,.35) !important;
    box-shadow:none !important; transform:none !important;
}}

/* INPUTS */
.stTextInput>div>div>input,.stNumberInput>div>div>input {{
    background:var(--input-bg) !important; border:1px solid var(--border) !important;
    color:var(--text-primary) !important; border-radius:8px !important;
    transition:border-color .2s,box-shadow .2s !important;
}}
.stTextInput>div>div>input:focus {{
    border-color:rgba(201,168,76,.5) !important;
    box-shadow:0 0 0 3px rgba(201,168,76,.08) !important;
}}
.stSelectbox>div>div,.stMultiSelect>div>div {{
    background:var(--input-bg) !important; border:1px solid var(--border) !important;
    border-radius:8px !important;
}}
.stSlider>div>div>div>div {{ background:{GOLD} !important; }}
.stDataFrame {{ border:1px solid var(--border) !important; border-radius:12px !important; }}
.streamlit-expanderHeader {{
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; color:var(--text-primary) !important;
}}
.stAlert {{ border-radius:10px !important; }}

::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:var(--bg-base); }}
::-webkit-scrollbar-thumb {{ background:rgba(201,168,76,.2); border-radius:10px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(201,168,76,.4); }}
hr {{ border-color:var(--border) !important; margin:1.5rem 0 !important; }}

@keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
.fade-in {{ animation:fadeIn .3s ease forwards; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# YARDIMCILAR
# ═══════════════════════════════════════════

def section_header(title, subtitle="", g1=None, g2=None):
    g1 = g1 or T['text_primary']
    g2 = g2 or GOLD
    return f"""
    <div style='margin-bottom:28px;animation:fadeIn .3s ease;'>
        <h1 style='font-family:Syne;font-size:1.9rem;font-weight:800;margin:0;letter-spacing:-.02em;
                   background:linear-gradient(135deg,{g1},{g2});
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>{title}</h1>
        {f"<p style='font-family:DM Mono;font-size:.72rem;color:{T['text_muted']};letter-spacing:.1em;margin-top:6px;text-transform:uppercase;'>{subtitle}</p>" if subtitle else ""}
    </div>"""


def hr():
    return f"<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,.15),transparent);margin:20px 0;'></div>"


# ═══════════════════════════════════════════
# GİRİŞ / KAYIT SAYFASI
# ═══════════════════════════════════════════

def show_auth_page():

    # ── SETUP.PY UYARISI ──
    if not admin_exists():
        st.markdown(f"""
        <div style='max-width:520px;margin:60px auto 0 auto;
                    background:rgba(255,107,53,.08);
                    border:1px solid rgba(255,107,53,.35);
                    border-left:4px solid {ORANGE};
                    border-radius:14px;padding:24px 28px;'>
            <div style='font-family:Syne;font-size:1.1rem;font-weight:800;
                        color:{ORANGE};margin-bottom:10px;'>
                ⚙️ Kurulum Gerekli
            </div>
            <div style='font-family:DM Mono;font-size:.72rem;
                        color:{T['text_secondary']};line-height:2;'>
                Henüz yönetici hesabı oluşturulmadı.<br>
                Uygulamayı başlatmadan önce terminalde çalıştırın:
            </div>
            <div style='background:rgba(0,0,0,.3);border-radius:8px;
                        padding:12px 16px;margin-top:12px;font-family:DM Mono;
                        font-size:.8rem;color:{CYAN};letter-spacing:.05em;'>
                python setup.py
            </div>
            <div style='font-family:DM Mono;font-size:.63rem;
                        color:{T['text_muted']};margin-top:10px;'>
                Kurulum tamamlandıktan sonra bu sayfayı yenileyin.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── NORMAL GİRİŞ SAYFASI ──
    _, col, _ = st.columns([1, 1.1, 1])

    with col:
        # Logo
        st.markdown(f"""
        <div style='text-align:center;padding:36px 0 24px 0;animation:fadeIn .5s ease;'>
            <div style='font-size:2.2rem;margin-bottom:8px;'>💎</div>
            <div style='font-family:Syne;font-size:2rem;font-weight:800;
                        background:linear-gradient(135deg,{GOLD},#FFE4A0);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                FinSight
            </div>
            <div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};
                        letter-spacing:.22em;text-transform:uppercase;margin-top:4px;'>
                Analiz Sistemi
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Sekme seçici
        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("🔐 Giriş Yap", use_container_width=True, key="tab_login"):
                st.session_state.auth_tab = "login"
                st.rerun()
        with tc2:
            if st.button("✨ Kayıt Ol", use_container_width=True, key="tab_register"):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        # ── GİRİŞ FORMU ──
        if st.session_state.auth_tab == "login":
            # DEĞIŞIKLIK 1: "Sürtünmeza Giriş Yapın" başlığı kaldırıldı
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("👤 Kullanıcı Adı", placeholder="kullaniciadi")
                password = st.text_input("🔑 Şifre", type="password", placeholder="••••••••")

                ca, cb = st.columns(2)
                with ca: submit   = st.form_submit_button("🔐 Giriş Yap", use_container_width=True)
                with cb: tema_btn = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)

                if tema_btn:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()

                if submit:
                    if not username or not password:
                        st.error("Kullanıcı adı ve şifre giriniz.")
                    else:
                        result = login_user(username, password)
                        if result["success"]:
                            u = result["user"]
                            st.session_state.logged_in    = True
                            st.session_state.username     = u["username"]
                            st.session_state.role         = u["role"]
                            st.session_state.display_name = u["display_name"] or u["username"]
                            st.session_state.avatar       = u["avatar"] or "👤"
                            st.session_state.user_id      = u["id"]
                            st.rerun()
                        else:
                            st.error(result["message"])

            # Admin paneli küçük linki
            pending_count = len(get_pending_users())
            badge = f" ({pending_count} onay bekliyor)" if pending_count > 0 else ""
            st.markdown(f"""
            <div style='text-align:center;margin-top:14px;'>
                <span style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};'>
                    Yönetici misiniz? Giriş yapıp
                    <span style='color:{GOLD};'>⚙️ Yönetici</span> panelini kullanın{badge}.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── KAYIT FORMU ──
        else:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});
                        border:1px solid {T['border_bright']};border-radius:16px;
                        padding:22px 26px 6px 26px;
                        box-shadow:0 8px 40px rgba(0,0,0,{T['shadow_opacity']});
                        animation:fadeIn .3s ease;'>
                <div style='font-family:Syne;font-size:1rem;font-weight:700;
                            color:{T['text_primary']};margin-bottom:2px;'>
                    Yeni Hesap Oluştur
                </div>
                <div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};
                            margin-bottom:16px;'>
                    Kayıt sonrası yönetici onayı beklenir
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("register_form", clear_on_submit=True):
                r_display  = st.text_input("📛 Ad Soyad", placeholder="Ahmet Yılmaz")
                r_username = st.text_input("👤 Kullanıcı Adı", placeholder="küçük harf, boşluksuz")
                r_email    = st.text_input("📧 E-posta", placeholder="ornek@email.com")

                rc1, rc2 = st.columns(2)
                with rc1: r_pw1 = st.text_input("🔑 Şifre", type="password", placeholder="Min. 6 karakter")
                with rc2: r_pw2 = st.text_input("🔑 Şifre Onayı", type="password", placeholder="••••••••")

                r_role = st.selectbox("🎭 Talep Edilen Rol", ["viewer","analyst"],
                    format_func=lambda x: {
                        "viewer":  "👁️ İzleyici — Genel sayfaları görüntüler",
                        "analyst": "📊 Analist — Detaylı analizler yapabilir",
                    }[x])

                ra, rb = st.columns(2)
                with ra: r_submit = st.form_submit_button("✨ Kayıt Ol", use_container_width=True)
                with rb: r_tema   = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)

                if r_tema:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()

                if r_submit:
                    err = None
                    if not all([r_display, r_username, r_email, r_pw1, r_pw2]):
                        err = "Tüm alanları doldurun."
                    elif " " in r_username:
                        err = "Kullanıcı adında boşluk olamaz."
                    elif r_pw1 != r_pw2:
                        err = "Şifreler eşleşmiyor."

                    if err:
                        st.error(err)
                    else:
                        result = register_user(r_username, r_email, r_pw1, r_display, r_role)
                        if result["success"]:
                            st.success("✅ Kayıt başarılı!")
                            st.markdown(f"""
                            <div style='background:rgba(255,107,53,.07);
                                        border:1px solid rgba(255,107,53,.25);
                                        border-left:3px solid {ORANGE};border-radius:10px;
                                        padding:14px 18px;margin-top:6px;'>
                                <div style='font-family:Syne;font-size:.85rem;font-weight:700;
                                            color:{ORANGE};margin-bottom:6px;'>
                                    ⏳ Onay bekleniyor
                                </div>
                                <div style='font-family:DM Mono;font-size:.66rem;
                                            color:{T['text_secondary']};line-height:1.9;'>
                                    Yönetici hesabınızı onayladıktan sonra giriş yapabilirsiniz.<br>
                                    Onay için yönetici ile iletişime geçin.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(result["message"])


# ─────────────────────────────────
# GİRİŞ KONTROLÜ
# ─────────────────────────────────

if not st.session_state.logged_in:
    show_auth_page()
    st.stop()


# ═══════════════════════════════════════════
# ROL / SAYFA
# ═══════════════════════════════════════════

ROLE_PAGES = {
    "admin":   ["📊 Genel Bakış","📈 Aylık İşlem Hacmi","💎 Segment",
                "📂 Kategori Ortalamaları","🎯 Harcama × Risk",
                "📈 Trend Analizi","🔍 Müşteri Analizi",
                "⚠️ Risk & Fraud","🗺️ Coğrafi Analiz","🤖 AI İçgörüleri",
                "👤 Müşteri Detay","⚙️ Yönetici"],
    "analyst": ["📊 Genel Bakış","📈 Aylık İşlem Hacmi","💎 Segment",
                "📂 Kategori Ortalamaları","🎯 Harcama × Risk",
                "📈 Trend Analizi","🔍 Müşteri Analizi",
                "⚠️ Risk & Fraud","🗺️ Coğrafi Analiz","🤖 AI İçgörüleri",
                "👤 Müşteri Detay"],
    "viewer":  ["📊 Genel Bakış","📈 Aylık İşlem Hacmi","💎 Segment",
                "📂 Kategori Ortalamaları","🎯 Harcama × Risk",
                "📈 Trend Analizi","🗺️ Coğrafi Analiz"],
}

current_role  = st.session_state.role
allowed_pages = ROLE_PAGES.get(current_role, ROLE_PAGES["viewer"])


# ─────────────────────────────────
# API
# ─────────────────────────────────

API_URL = "http://localhost:8000"

@st.cache_data(ttl=30, show_spinner=False)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=0.5)
        return r.status_code == 200
    except:
        return False

API_ALIVE  = check_api()
api_status = "🟢 API Aktif" if API_ALIVE else "🔴 API Kapalı"

@st.cache_data(ttl=300, show_spinner=False)
def api_get(endpoint, params=None):
    if not API_ALIVE: return None
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=1.5)
        return r.json() if r.status_code == 200 else None
    except:
        return None


# ─────────────────────────────────
# VERİ
# ─────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def generate_demo_data(n=1219):
    np.random.seed(42)
    kat   = ['Market','Restoran','Yakıt','Online Alışveriş','Sağlık','Eğlence','Ulaşım','Eğitim','Giyim','Elektronik']
    sehir = ['İstanbul','Ankara','İzmir','Bursa','Antalya','Adana','Konya','Gaziantep','Şanlıurfa','Mersin']
    df = pd.DataFrame({
        'client_id':      range(1, n+1),
        'sehir':          np.random.choice(sehir, n),
        'yas':            np.random.randint(18, 75, n),
        'toplam_harcama': np.random.lognormal(10, 1.2, n),
        'islem_sayisi':   np.random.randint(5, 250, n),
        'risk_skoru':     np.random.exponential(8, n).clip(0, 65),
        'kategori':       np.random.choice(kat, n),
        'aktif_ay':       np.random.randint(1, 36, n),
    })
    df['risk_seviyesi'] = pd.cut(df['risk_skoru'],
        bins=[-np.inf,20,40,np.inf], labels=['Düşük Risk','Orta Risk','Yüksek Risk'])
    df['fraud_tahmini'] = np.where(
        np.random.random(n) < (df['risk_skoru']/65)*0.1, 'Şüpheli', 'Normal')
    df['segment'] = pd.cut(df['toplam_harcama'],
        bins=[0,5000,20000,50000,np.inf], labels=['Bronze','Silver','Gold','Platinum'])
    return df

def adapt_real_data(df):
    df = df.copy(); np.random.seed(42); n = len(df)
    df['toplam_harcama'] = df['toplam']
    df['islem_sayisi']   = df['islem']
    sehir = ['İstanbul','Ankara','İzmir','Bursa','Antalya','Adana','Konya','Gaziantep','Şanlıurfa','Mersin']
    kat   = ['Market','Restoran','Yakıt','Online Alışveriş','Sağlık','Eğlence','Ulaşım','Eğitim','Giyim','Elektronik']
    df['sehir']     = np.random.choice(sehir, n)
    df['yas']       = np.random.randint(18,75,n)
    df['aktif_ay']  = np.random.randint(1,36,n)
    df['kategori']  = np.random.choice(kat,n)
    df['risk_seviyesi'] = df['risk_seviyesi'].astype(str).str.replace(r'[🟢🟡🔴]','',regex=True).str.strip()
    df['fraud_tahmini'] = np.where(df['risk_skoru']>30,'Şüpheli','Normal')
    df['segment'] = pd.cut(df['toplam_harcama'],
        bins=[0,100000,400000,700000,np.inf], labels=['Bronze','Silver','Gold','Platinum'])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_main_data():
    try:
        from database import get_all_clients
        return adapt_real_data(get_all_clients()), "🟢 Canlı Veritabanı"
    except:
        return generate_demo_data(), "🟡 Demo Modu"

@st.cache_data(ttl=3600, show_spinner=False)
def filter_data(risk_filtre, segment_filtre, risk_min, risk_max):
    df_all, _ = load_main_data()
    return df_all[
        df_all['risk_seviyesi'].isin(risk_filtre) &
        df_all['segment'].isin(segment_filtre) &
        (df_all['risk_skoru'] >= risk_min) &
        (df_all['risk_skoru'] <= risk_max)
    ].copy()

df_main, data_source = load_main_data()

def PL(**kw):
    b = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=T['plot_bg'],
        font=dict(family='DM Sans',color=T['text_secondary'],size=11),
        title_font=dict(family='Syne',size=14,color=T['text_primary']),
        colorway=[GOLD,CYAN,GREEN,RED,ORANGE,PURPLE],
        xaxis=dict(gridcolor='rgba(90,106,122,.12)',linecolor='rgba(90,106,122,.15)',
                   tickfont=dict(family='DM Mono',size=9,color=T['text_muted']),zeroline=False),
        yaxis=dict(gridcolor='rgba(90,106,122,.12)',linecolor='rgba(90,106,122,.15)',
                   tickfont=dict(family='DM Mono',size=9,color=T['text_muted']),zeroline=False),
        legend=dict(bgcolor=T['bg_card'],bordercolor=T['border'],borderwidth=1,
                    font=dict(family='DM Mono',size=9,color=T['text_secondary'])),
        margin=dict(l=40,r=20,t=50,b=40),
        hoverlabel=dict(bgcolor=T['bg_card'],bordercolor='rgba(201,168,76,.3)',
                        font=dict(family='DM Mono',size=11,color=T['text_primary'])),
    )
    b.update(kw)
    return b


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 14px 0;'>
        <div style='font-size:1.2rem;margin-bottom:4px;'>💎</div>
        <div style='font-family:Syne;font-size:1.2rem;font-weight:800;
                    background:linear-gradient(135deg,{GOLD},#FFE4A0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            FinSight
        </div>
        <div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};
                    letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>
            v5.2 — {current_role.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    role_color    = {"admin":RED,"analyst":GOLD,"viewer":GREEN}.get(current_role,GOLD)
    pending_count = len(get_pending_users()) if current_role=="admin" else 0
    p_badge       = (f' <span style="background:{RED};color:#fff;border-radius:10px;'
                     f'padding:1px 6px;font-size:.55rem;">{pending_count}</span>'
                     if pending_count > 0 else "")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(201,168,76,.08),rgba(201,168,76,.03));
                border:1px solid rgba(201,168,76,.18);border-radius:12px;
                padding:10px 13px;margin:0 4px 10px 4px;'>
        <div style='display:flex;align-items:center;gap:9px;'>
            <div style='font-size:1.2rem;'>{st.session_state.avatar}</div>
            <div>
                <div style='font-family:Syne;font-size:.82rem;font-weight:700;
                            color:{T['text_primary']};'>{st.session_state.display_name}</div>
                <div style='font-family:DM Mono;font-size:.55rem;color:{role_color};
                            text-transform:uppercase;letter-spacing:.12em;'>
                    {current_role}{p_badge}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding:0 4px 8px 4px;'>
        <div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};
                    display:flex;align-items:center;gap:6px;margin-bottom:3px;'>
            <span style='width:5px;height:5px;border-radius:50%;background:{GREEN};
                         display:inline-block;box-shadow:0 0 5px {GREEN};'></span>
            {data_source}
        </div>
        <div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};
                    display:flex;align-items:center;gap:6px;'>
            <span style='width:5px;height:5px;border-radius:50%;
                         background:{"#00E396" if API_ALIVE else "#FF4560"};
                         display:inline-block;'></span>
            {api_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(hr(), unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Navigasyon</div>", unsafe_allow_html=True)
    sayfa = st.radio("", allowed_pages, label_visibility="collapsed")

    st.markdown(hr(), unsafe_allow_html=True)

    if current_role in ["admin","analyst"]:
        st.markdown(f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Filtreler</div>", unsafe_allow_html=True)
        risk_filtre    = st.multiselect("Risk", ['Düşük Risk','Orta Risk','Yüksek Risk'],
            default=['Düşük Risk','Orta Risk','Yüksek Risk'], label_visibility="collapsed")
        segment_filtre = st.multiselect("Segment", ['Bronze','Silver','Gold','Platinum'],
            default=['Bronze','Silver','Gold','Platinum'], label_visibility="collapsed")
        risk_range     = st.slider("Risk", 0, 65, (0,65), label_visibility="collapsed")
    else:
        risk_filtre    = ['Düşük Risk','Orta Risk','Yüksek Risk']
        segment_filtre = ['Bronze','Silver','Gold','Platinum']
        risk_range     = (0, 65)

    df = filter_data(tuple(sorted(risk_filtre)), tuple(sorted(segment_filtre)), risk_range[0], risk_range[1])

    st.markdown(f"""
    <div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};
                margin-top:6px;padding:6px 10px;background:rgba(201,168,76,.06);
                border-radius:8px;border:1px solid rgba(201,168,76,.1);'>
        <span style='color:{GOLD};font-weight:600;'>{len(df):,}</span> müşteri seçili
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tc, cc = st.columns(2)
    with tc:
        st.markdown('<div class="btn-dim">', unsafe_allow_html=True)
        if st.button(f"{T['toggle_icon']} Tema", use_container_width=True, key="tema_sb"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with cc:
        st.markdown('<div class="btn-ghost">', unsafe_allow_html=True)
        if st.button("🚪 Çıkış", use_container_width=True, key="logout_sb"):
            for k in ["logged_in","username","role","display_name","avatar","user_id"]:
                st.session_state[k] = False if k=="logged_in" else ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SAYFALAR
# ═══════════════════════════════════════════

# ── GENEL BAKIŞ ──
if sayfa == "📊 Genel Bakış":
    st.markdown(section_header("Finansal Analiz Paneli",
        f"Gerçek zamanlı — {datetime.now().strftime('%d %B %Y, %H:%M')}"),
        unsafe_allow_html=True)

    api_stats = api_get("/stats")
    api_fraud = api_get("/stats/fraud")

    if api_stats:
        toplam   = api_stats.get("toplam_client", len(df))
        yuksek   = api_stats.get("yuksek_risk", 0)
        orta     = api_stats.get("orta_risk", 0)
        hacim    = api_stats.get("toplam_hacim", 0) or 0
        ort_risk = api_stats.get("ort_risk_skoru", df['risk_skoru'].mean())
    else:
        toplam   = len(df)
        yuksek   = int((df['risk_seviyesi']=='Yüksek Risk').sum())
        orta     = int((df['risk_seviyesi']=='Orta Risk').sum())
        hacim    = float(df['toplam_harcama'].sum())
        ort_risk = df['risk_skoru'].mean()

    if api_fraud:
        supheli    = api_fraud.get("supheli",0) + api_fraud.get("yuksek_risk",0)
        fraud_oran = round(supheli/max(api_fraud.get("toplam",1),1)*100,1)
    else:
        supheli    = int((df['fraud_tahmini']=='Şüpheli').sum())
        fraud_oran = round(supheli/max(len(df),1)*100,1)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("👥 Toplam Müşteri", f"{toplam:,}", delta="Canlı veri")
    with k2: st.metric("⚠️ Şüpheli", f"{supheli:,}", delta=f"%{fraud_oran}")
    with k3: st.metric("🔴 Yüksek Risk", f"{yuksek:,}")
    with k4: st.metric("🟡 Orta Risk", f"{orta:,}")
    with k5: st.metric("💰 Hacim", f"${hacim/1e6:.1f}M" if hacim>1e6 else f"${hacim:,.0f}")

    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

    # ── 2 YENİ WİDGET ──
    w1, w2 = st.columns([3, 2])

    with w1:
        # Günlük İşlem Trendi — son 30 gün simüle
        np.random.seed(99)
        gunler = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
        gunluk = np.random.lognormal(np.log(float(df['toplam_harcama'].sum())/30), 0.15, 30)
        gunluk_smoothed = pd.Series(gunluk).rolling(3, min_periods=1).mean()
        trend_renk = GREEN if gunluk[-1] > gunluk[-2] else RED
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=gunler, y=gunluk,
            fill='tozeroy', fillcolor='rgba(0,212,255,0.05)',
            line=dict(color=CYAN, width=1.5), opacity=0.5, name='Günlük', showlegend=False
        ))
        fig_trend.add_trace(go.Scatter(
            x=gunler, y=gunluk_smoothed,
            line=dict(color=trend_renk, width=2.5), name='3G Ort.', showlegend=False
        ))
        fig_trend.add_annotation(
            x=gunler[-1], y=float(gunluk_smoothed.iloc[-1]),
            text=f"${float(gunluk_smoothed.iloc[-1])/1e6:.2f}M",
            showarrow=False, font=dict(family='Syne', size=12, color=trend_renk),
            xanchor='right', yanchor='bottom'
        )
        _pl = PL()
        _pl['margin'] = dict(l=30, r=20, t=45, b=30)
        fig_trend.update_layout(title="📅 Son 30 Günlük İşlem Trendi", height=240, **_pl)
        fig_trend.update_xaxes(tickformat='%d %b', nticks=8)
        st.plotly_chart(fig_trend, use_container_width=True)

    with w2:
        # Risk Özeti — hızlı bakış kartları
        dusuk  = int((df['risk_seviyesi']=='Düşük Risk').sum())
        orta_r = int((df['risk_seviyesi']=='Orta Risk').sum())
        yuk_r  = int((df['risk_seviyesi']=='Yüksek Risk').sum())
        total  = len(df)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});
                    border:1px solid {T['border']};border-radius:14px;
                    padding:20px 22px;height:240px;display:flex;flex-direction:column;justify-content:space-between;'>
            <div style='font-family:DM Mono;font-size:.6rem;color:{T["text_muted"]};
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:12px;'>
                🔥 Risk Özeti
            </div>
            <div style='display:flex;flex-direction:column;gap:10px;'>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <div style='display:flex;align-items:center;gap:8px;'>
                        <div style='width:8px;height:8px;border-radius:50%;background:{GREEN};box-shadow:0 0 6px {GREEN};'></div>
                        <span style='font-family:DM Sans;font-size:.82rem;color:{T["text_secondary"]};'>Düşük Risk</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <div style='background:rgba(0,227,150,.08);border-radius:4px;padding:1px 8px;'>
                            <span style='font-family:DM Mono;font-size:.7rem;color:{GREEN};'>{dusuk/total*100:.0f}%</span>
                        </div>
                        <span style='font-family:Syne;font-size:.95rem;font-weight:700;color:{T["text_primary"]};'>{dusuk:,}</span>
                    </div>
                </div>
                <div style='width:100%;background:{T["border"]};height:2px;border-radius:2px;'>
                    <div style='width:{dusuk/total*100:.0f}%;background:{GREEN};height:100%;border-radius:2px;'></div>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <div style='display:flex;align-items:center;gap:8px;'>
                        <div style='width:8px;height:8px;border-radius:50%;background:{ORANGE};box-shadow:0 0 6px {ORANGE};'></div>
                        <span style='font-family:DM Sans;font-size:.82rem;color:{T["text_secondary"]};'>Orta Risk</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <div style='background:rgba(255,107,53,.08);border-radius:4px;padding:1px 8px;'>
                            <span style='font-family:DM Mono;font-size:.7rem;color:{ORANGE};'>{orta_r/total*100:.0f}%</span>
                        </div>
                        <span style='font-family:Syne;font-size:.95rem;font-weight:700;color:{T["text_primary"]};'>{orta_r:,}</span>
                    </div>
                </div>
                <div style='width:100%;background:{T["border"]};height:2px;border-radius:2px;'>
                    <div style='width:{orta_r/total*100:.0f}%;background:{ORANGE};height:100%;border-radius:2px;'></div>
                </div>
                <div style='display:flex;align-items:center;justify-content:space-between;'>
                    <div style='display:flex;align-items:center;gap:8px;'>
                        <div style='width:8px;height:8px;border-radius:50%;background:{RED};box-shadow:0 0 6px {RED};'></div>
                        <span style='font-family:DM Sans;font-size:.82rem;color:{T["text_secondary"]};'>Yüksek Risk</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:10px;'>
                        <div style='background:rgba(255,69,96,.08);border-radius:4px;padding:1px 8px;'>
                            <span style='font-family:DM Mono;font-size:.7rem;color:{RED};'>{yuk_r/total*100:.0f}%</span>
                        </div>
                        <span style='font-family:Syne;font-size:.95rem;font-weight:700;color:{T["text_primary"]};'>{yuk_r:,}</span>
                    </div>
                </div>
                <div style='width:100%;background:{T["border"]};height:2px;border-radius:2px;'>
                    <div style='width:{yuk_r/total*100:.0f}%;background:{RED};height:100%;border-radius:2px;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Müşteri Tablosu",expanded=False):
        cols = ['client_id','sehir','yas','toplam_harcama','islem_sayisi',
                'risk_skoru','risk_seviyesi','fraud_tahmini','segment']
        cA,cB = st.columns([4,1])
        with cB:
            st.download_button("⬇️ CSV", df[cols].to_csv(index=False).encode('utf-8'),
                "musteri.csv","text/csv",use_container_width=True)
        styled = df[cols].head(100).copy()
        styled['toplam_harcama'] = styled['toplam_harcama'].apply(lambda x: f"${x:,.0f}")
        styled['risk_skoru']     = styled['risk_skoru'].apply(lambda x: f"{x:.1f}")
        st.dataframe(styled, use_container_width=True, height=380)



# ── AYLIK İŞLEM HACMİ ──
elif sayfa == "📈 Aylık İşlem Hacmi":
    st.markdown(section_header("Aylık İşlem Hacmi","Zaman serisi — işlem hacmi trendi",None,GOLD), unsafe_allow_html=True)
    api_aylik = api_get("/stats/aylik", {"son_ay":24})
    fig = go.Figure()
    if api_aylik:
        da = pd.DataFrame(api_aylik)
        fig.add_trace(go.Scatter(x=da['donem'],y=da['toplam'],fill='tozeroy',
            fillcolor='rgba(201,168,76,.07)',line=dict(color=GOLD,width=2.5),name='Aylık Hacim'))
        fig.add_trace(go.Scatter(x=da['donem'],
            y=pd.Series(da['toplam'].values).rolling(3).mean(),
            line=dict(color=CYAN,width=1.5,dash='dot'),name='3A Ort.'))
    fig.update_layout(title="📈 Aylık İşlem Hacmi",height=520,**PL())
    st.plotly_chart(fig, use_container_width=True)


# ── SEGMENT ──
elif sayfa == "💎 Segment":
    st.markdown(section_header("Segment Dağılımı","Müşteri segmentlerine göre dağılım",None,CYAN), unsafe_allow_html=True)
    seg = df['segment'].value_counts()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Pie(labels=seg.index,values=seg.values,hole=.62,
            marker=dict(colors=[GOLD,CYAN,GREEN,RED],line=dict(color=T['bg_base'],width=2)),
            textfont=dict(family='DM Mono',size=11)))
        fig.add_annotation(text=f"<b>{len(df):,}</b>",x=.5,y=.5,showarrow=False,
            font=dict(family='Syne',size=28,color=GOLD))
        fig.update_layout(title="💎 Segment Dağılımı",height=500,**PL())
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_df = df.groupby('segment').agg(
            musteri=('client_id','count'),
            ort_harcama=('toplam_harcama','mean'),
            ort_risk=('risk_skoru','mean')
        ).reset_index()
        fig2 = go.Figure(go.Bar(x=seg_df['segment'],y=seg_df['musteri'],
            marker=dict(color=[GOLD,CYAN,GREEN,RED]),
            text=seg_df['musteri'],textposition='outside',
            textfont=dict(family='DM Mono',size=10,color=T['text_secondary'])))
        fig2.update_layout(title="👥 Segment Müşteri Sayısı",height=500,**PL())
        st.plotly_chart(fig2, use_container_width=True)


# ── KATEGORİ ORTALAMALARI ──
elif sayfa == "📂 Kategori Ortalamaları":
    st.markdown(section_header("Kategori Ortalamaları","Kategoriye göre ortalama harcama",None,ORANGE), unsafe_allow_html=True)
    top = df.groupby('kategori')['toplam_harcama'].mean().nlargest(10).reset_index()
    fig = go.Figure(go.Bar(
        y=top['kategori'], x=top['toplam_harcama'], orientation='h',
        marker=dict(color=list(range(len(top))), colorscale=[[0,'rgba(201,168,76,.25)'],[1,GOLD]]),
        text=[f'${v/1000:.0f}K' for v in top['toplam_harcama']],
        textposition='outside',
        textfont=dict(family='DM Mono',size=10,color=T['text_secondary'])
    ))
    fig.update_layout(title="📂 Kategoriye Göre Ortalama Harcama",height=560,**PL())
    st.plotly_chart(fig, use_container_width=True)


# ── HARCAMA × RİSK ──
elif sayfa == "🎯 Harcama × Risk":
    st.markdown(section_header("Harcama × Risk","Fraud tespiti dağılım analizi",None,RED), unsafe_allow_html=True)
    fig = go.Figure()
    for ft,col_c in [('Normal',GREEN),('Şüpheli',RED)]:
        mask = df['fraud_tahmini']==ft
        if mask.sum()>0:
            s = df[mask].sample(min(500,mask.sum()),random_state=42)
            fig.add_trace(go.Scatter(x=s['toplam_harcama'],y=s['risk_skoru'],
                mode='markers',name=ft,
                marker=dict(color=col_c,size=5,opacity=.7),
                text=s['client_id'].astype(str)))
    fig.update_xaxes(type='log',title='Harcama ($)')
    fig.update_yaxes(title='Risk Skoru')
    fig.update_layout(title="🎯 Harcama × Risk Dağılımı",height=580,**PL())
    st.plotly_chart(fig, use_container_width=True)


# ── TREND ANALİZİ ──
elif sayfa == "📈 Trend Analizi":
    st.markdown(section_header("Trend Analizi","Zaman serisi ve kategori",None,CYAN), unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Aylık Hacim","🏷️ Kategori","🌡️ Yoğunluk"])

    with tab1:
        api_aylik = api_get("/stats/aylik",{"son_ay":24})
        if api_aylik:
            da = pd.DataFrame(api_aylik)
            da['degisim']    = pd.Series(da['toplam'].values).pct_change()*100
            da['kumulative'] = da['toplam'].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=da['donem'],y=da['toplam'],fill='tozeroy',
                fillcolor='rgba(201,168,76,.07)',line=dict(color=GOLD,width=2.5),name='Aylık Hacim'))
            fig.add_trace(go.Scatter(x=da['donem'],
                y=pd.Series(da['toplam'].values).rolling(3).mean(),
                line=dict(color=CYAN,width=1.5,dash='dot'),name='3A Ortalama'))
            fig.update_layout(title="📈 Aylık İşlem Hacmi",height=400,**PL())
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            with c1:
                fig2 = go.Figure(go.Bar(x=da['donem'].iloc[1:],y=da['degisim'].iloc[1:],
                    marker_color=np.where(da['degisim'].iloc[1:]>=0,GREEN,RED)))
                fig2.update_layout(title="📉 Aylık Değişim (%)",height=300,**PL())
                st.plotly_chart(fig2, use_container_width=True)
            with c2:
                fig3 = go.Figure(go.Scatter(x=da['donem'],y=da['kumulative'],fill='tozeroy',
                    fillcolor='rgba(0,212,255,.07)',line=dict(color=CYAN,width=2)))
                fig3.update_layout(title="📈 Kümülatif Toplam",height=300,**PL())
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("📡 API bağlantısı yok.")

    with tab2:
        api_kategori = api_get("/stats/kategori")
        dk = pd.DataFrame(api_kategori) if api_kategori else \
             df.groupby('kategori')['toplam_harcama'].sum().reset_index().rename(columns={'toplam_harcama':'toplam'})
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=dk['kategori'],y=dk['toplam'],
                marker=dict(color=list(range(len(dk))),colorscale=[[0,'rgba(201,168,76,.3)'],[1,GOLD]])))
            fig.update_layout(title="💰 Kategori Toplam Hacim",height=380,**PL())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Pie(labels=dk['kategori'],values=dk['toplam'],hole=.4,
                textfont=dict(family='DM Mono',size=9),
                marker=dict(line=dict(color=T['bg_base'],width=2))))
            fig2.update_layout(title="🥧 Dağılım",height=380,**PL())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        gunler = ['Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar']
        heat   = np.random.lognormal(0,.5,(7,24))
        heat[5:7,10:22] *= 2.5; heat[:5,8:10] *= 1.8
        fig = go.Figure(go.Heatmap(z=heat,x=[f"{h:02d}:00" for h in range(24)],y=gunler,
            colorscale=[[0,T['bg_base']],[.3,'rgba(201,168,76,.3)'],[1,GOLD]]))
        fig.update_layout(title="🌡️ İşlem Yoğunluk Haritası",height=440,**PL())
        st.plotly_chart(fig, use_container_width=True)


# ── MÜŞTERİ ANALİZİ ──
elif sayfa == "🔍 Müşteri Analizi":
    st.markdown(section_header("Müşteri Analizi","Segment, demografi ve davranış",None,GREEN), unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["💎 Segment","👥 Demografi","🔵 Davranış"])

    with tab1:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            seg = df.groupby('segment').agg(ort_harcama=('toplam_harcama','mean')).reset_index()
            fig = go.Figure(go.Bar(x=seg['segment'],y=seg['ort_harcama'],
                marker=dict(color=['#5A6A7A',CYAN,GOLD,RED]),
                text=[f"${v/1000:.0f}K" for v in seg['ort_harcama']],textposition='outside',
                textfont=dict(family='DM Mono',size=9,color=T['text_secondary'])))
            fig.update_layout(title="💎 Segment Ort. Harcama",height=340,**PL())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            seg2 = df.groupby('segment').size().reset_index(name='n')
            fig2 = go.Figure(go.Bar(x=seg2['segment'],y=seg2['n'],
                marker=dict(color=['#5A6A7A',CYAN,GOLD,RED]),
                text=seg2['n'],textposition='outside',
                textfont=dict(family='DM Mono',size=9,color=T['text_secondary'])))
            fig2.update_layout(title="👥 Segment Müşteri Sayısı",height=340,**PL())
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        cols5 = st.columns(4)
        for i,(sn,col_c) in enumerate({'Bronze':'#8A7A6A','Silver':CYAN,'Gold':GOLD,'Platinum':RED}.items()):
            sd = df[df['segment']==sn]
            with cols5[i]:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,{col_c}18,{col_c}05);
                            border:1px solid {col_c}30;border-radius:14px;padding:18px 20px;'>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{col_c};
                                text-transform:uppercase;letter-spacing:.15em;margin-bottom:8px;'>{sn}</div>
                    <div style='font-family:Syne;font-size:1.6rem;font-weight:800;
                                color:{T["text_primary"]};'>{len(sd):,}</div>
                    <div style='font-family:DM Mono;font-size:.63rem;color:{T["text_muted"]};margin-top:4px;'>
                        Ort. ${sd["toplam_harcama"].mean()/1000:.0f}K
                    </div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df,x='yas',nbins=25,color_discrete_sequence=[CYAN])
            fig.update_traces(marker_line_width=0,opacity=.8)
            fig.update_layout(title="👤 Yaş Dağılımı",height=340,**PL(),bargap=.04)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ds2 = df.groupby('sehir').size().nlargest(10).reset_index(name='n')
            fig2 = go.Figure(go.Bar(x=ds2['sehir'],y=ds2['n'],
                marker=dict(color=ds2['n'],colorscale=[[0,'rgba(201,168,76,.2)'],[1,GOLD]])))
            fig2.update_layout(title="🏙️ Şehir Bazlı",height=340,**PL())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            smp = df.sample(min(400,len(df)),random_state=42)
            fig = px.scatter(smp,x='islem_sayisi',y='toplam_harcama',
                size='risk_skoru',color='segment',size_max=22,
                color_discrete_map={'Bronze':'#5A6A7A','Silver':CYAN,'Gold':GOLD,'Platinum':RED})
            fig.update_layout(title="🔵 İşlem × Harcama × Risk",height=400,**PL())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            aktif = pd.cut(df['aktif_ay'],bins=[0,6,12,24,36],labels=['0-6 ay','7-12 ay','13-24 ay','25+ ay'])
            df2   = df.copy(); df2['ag']=aktif
            ad    = df2.groupby('ag',observed=True)['toplam_harcama'].mean().reset_index()
            fig2  = go.Figure(go.Bar(x=ad['ag'].astype(str),y=ad['toplam_harcama'],
                marker=dict(color=[GREEN,CYAN,GOLD,RED]),
                text=[f"${v/1000:.0f}K" for v in ad['toplam_harcama']],textposition='outside',
                textfont=dict(family='DM Mono',size=9)))
            fig2.update_layout(title="📅 Aktiflik × Harcama",height=400,**PL())
            st.plotly_chart(fig2, use_container_width=True)


# ── RİSK & FRAUD ──
elif sayfa == "⚠️ Risk & Fraud":
    st.markdown(section_header("Risk & Fraud","Gerçek zamanlı tehdit tespiti",None,RED), unsafe_allow_html=True)
    fraud_rate = (df['fraud_tahmini']=='Şüpheli').mean()*100
    r1,r2,r3,r4 = st.columns(4)
    with r1: st.metric("🚨 Fraud Oranı",f"{fraud_rate:.2f}%")
    with r2: st.metric("⚡ Ort. Risk",f"{df['risk_skoru'].mean():.1f}")
    with r3: st.metric("🔴 Yüksek Risk",f"{(df['risk_seviyesi']=='Yüksek Risk').sum():,}")
    with r4: st.metric("🟡 Orta Risk",f"{(df['risk_seviyesi']=='Orta Risk').sum():,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Dağılım","🚨 Şüpheli Liste","📈 Trend"])

    with tab1:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns([2,1])
        with c1:
            fig = go.Figure()
            for lvl,col_c in [('Düşük Risk',GREEN),('Orta Risk',ORANGE),('Yüksek Risk',RED)]:
                sub = df[df['risk_seviyesi']==lvl]['risk_skoru']
                if len(sub)>0:
                    fig.add_trace(go.Histogram(x=sub,name=lvl,marker_color=col_c,opacity=.75,nbinsx=30))
            fig.update_layout(title="📊 Risk Skoru Dağılımı",barmode='overlay',height=380,**PL())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fc = df['fraud_tahmini'].value_counts()
            fig2 = go.Figure(go.Pie(labels=list(fc.index),values=list(fc.values),hole=.62,
                marker=dict(colors=[GREEN,RED],line=dict(color=T['bg_base'],width=2))))
            fig2.add_annotation(text=f"<b>{fraud_rate:.1f}%</b>",x=.5,y=.5,showarrow=False,
                font=dict(family='Syne',size=16,color=RED))
            fig2.update_layout(title="🎯 Dağılım",height=380,**PL())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        yr = df[df['risk_skoru']>df['risk_skoru'].quantile(.93)].sort_values('risk_skoru',ascending=False).head(50)
        fig = go.Figure(go.Bar(x=yr['client_id'].astype(str).head(20),y=yr['risk_skoru'].head(20),
            marker=dict(color=yr['risk_skoru'].head(20),colorscale=[[0,ORANGE],[1,RED]])))
        fig.update_layout(title="🚨 En Riskli 20 Müşteri",height=340,**PL())
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.dataframe(yr[['client_id','risk_skoru','risk_seviyesi','toplam_harcama','fraud_tahmini','sehir']],
            use_container_width=True,height=360)

    with tab3:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        donem = pd.date_range('2022-01-01',periods=24,freq='ME')
        rt = pd.DataFrame({'donem':donem,
            'ort_risk':np.random.normal(10,2,24).cumsum()/10+8,
            'fraud_oran':np.random.normal(5,1,24)})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rt['donem'],y=rt['ort_risk'],
            line=dict(color=GOLD,width=2),name='Ort. Risk'))
        fig.add_trace(go.Scatter(x=rt['donem'],y=rt['fraud_oran'],
            line=dict(color=RED,width=2,dash='dot'),name='Fraud %'))
        fig.update_layout(title="📈 Risk Trendi",height=400,**PL())
        st.plotly_chart(fig, use_container_width=True)


# ── COĞRAFİ ──
elif sayfa == "🗺️ Coğrafi Analiz":
    st.markdown(section_header("Coğrafi Analiz","Bölgesel dağılım",None,PURPLE), unsafe_allow_html=True)
    ds = df.groupby('sehir').agg(
        musteri_sayisi=('client_id','count'), ort_risk=('risk_skoru','mean'),
        toplam_hacim=('toplam_harcama','sum'), ort_harcama=('toplam_harcama','mean')
    ).reset_index()
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("🏙️ Şehir",f"{len(ds):,}")
    with k2: st.metric("👥 En Büyük",ds.nlargest(1,'musteri_sayisi')['sehir'].values[0])
    with k3: st.metric("⚠️ En Riskli",ds.nlargest(1,'ort_risk')['sehir'].values[0])
    with k4: st.metric("📊 Ort./Şehir",f"{int(ds['musteri_sayisi'].mean()):,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📊 Sıralama","🗺️ Harita","📈 Karşılaştırma"])

    with tab1:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(ds.sort_values('musteri_sayisi',ascending=True).tail(10),
                x='musteri_sayisi',y='sehir',orientation='h',color='ort_risk',
                color_continuous_scale=[[0,GREEN],[.5,ORANGE],[1,RED]])
            fig.update_layout(title="🏙️ Müşteri Sıralaması",height=400,**PL())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(ds.sort_values('ort_harcama',ascending=True).tail(10),
                x='ort_harcama',y='sehir',orientation='h',color='ort_harcama',
                color_continuous_scale=[[0,'rgba(201,168,76,.3)'],[1,GOLD]])
            fig2.update_layout(title="💰 Ort. Harcama",height=400,**PL())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        fig = px.treemap(ds,path=['sehir'],values='toplam_hacim',color='ort_risk',
            color_continuous_scale=[[0,'rgba(0,227,150,.8)'],[.5,'rgba(255,107,53,.8)'],[1,'rgba(255,69,96,.8)']])
        fig.update_layout(title="🗺️ Toplam Hacim",paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans',color=T['text_primary']),
            margin=dict(l=10,r=10,t=50,b=10),height=520)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        secili = st.multiselect("Şehir seçin",ds['sehir'].tolist(),default=ds['sehir'].tolist()[:5])
        if secili:
            dss = ds[ds['sehir'].isin(secili)]
            fig = go.Figure()
            for m,label,col_c in [('musteri_sayisi','Müşteri',GOLD),
                                   ('ort_risk','Ort. Risk',RED),('ort_harcama','Ort. Harcama',CYAN)]:
                vals = dss[m]/1000 if m=='ort_harcama' else dss[m]
                fig.add_trace(go.Bar(name=label,x=dss['sehir'],y=vals,marker_color=col_c))
            fig.update_layout(title="📈 Şehir Karşılaştırması",barmode='group',height=420,**PL())
            st.plotly_chart(fig, use_container_width=True)


# ── AI İÇGÖRÜLERİ ──
elif sayfa == "🤖 AI İçgörüleri":
    st.markdown(section_header("İçgörüler","Fraud tespiti ve model analizi",None,CYAN), unsafe_allow_html=True)
    tab1,tab2 = st.tabs(["🚨 Şüpheli Tespitler","📊 Model Performansı"])

    with tab1:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        api_supheli = api_get("/fraud/supheli",{"limit":8})
        anomaliler  = api_supheli or []
        if not anomaliler:
            rows = df[df['risk_skoru']>df['risk_skoru'].quantile(.97)].head(8)
            anomaliler = [{"client_id":int(r['client_id']),
                           "fraud_skoru":r['risk_skoru']*1.5,
                           "sehir":r['sehir'],
                           "toplam_harcama":r['toplam_harcama']} for _,r in rows.iterrows()]
        ca,cb = st.columns(2)
        for i,row in enumerate(anomaliler):
            fs  = float(row.get('fraud_skoru',0))
            bc  = RED if fs>70 else ORANGE
            sev = "🔴 Kritik" if fs>70 else "🟡 Şüpheli"
            card = f"""
            <div style='background:linear-gradient(135deg,rgba(255,69,96,.05),{T['bg_card']});
                        border-left:3px solid {bc};border-radius:10px;
                        padding:14px 18px;margin:8px 0;border:1px solid rgba(255,69,96,.1);'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='font-family:Syne;font-weight:700;color:{T['text_primary']};'>
                        Müşteri #{int(row['client_id'])}
                    </span>
                    <span style='font-family:DM Mono;font-size:.68rem;color:{bc};
                                 background:{bc}15;padding:2px 10px;border-radius:20px;
                                 border:1px solid {bc}40;'>{sev}</span>
                </div>
                <div style='font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};line-height:1.8;'>
                    Skor: <span style='color:{bc};font-weight:700;'>{fs:.1f}</span>
                    {f" | Şehir: {row['sehir']}" if 'sehir' in row else ""}
                    {f" | ${float(row['toplam_harcama']):,.0f}" if 'toplam_harcama' in row else ""}
                </div>
            </div>"""
            with (ca if i%2==0 else cb):
                st.markdown(card, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        api_metrics = api_get("/model/metrics")
        if api_metrics:
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("🎯 AUC-ROC",f"{api_metrics.get('auc_roc',0):.4f}")
            with m2: st.metric("⚡ F1",f"{api_metrics.get('f1_skoru',0):.4f}")
            with m3: st.metric("🔍 Precision",f"{api_metrics.get('precision',0):.4f}")
            with m4: st.metric("📡 Recall",f"{api_metrics.get('recall',0):.4f}")
        else:
            st.info("Model metrikleri için analiz modülünü çalıştırın.")
            fpr = np.linspace(0,1,100)
            tpr = np.sqrt(fpr)*.92+fpr*.08
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr,y=tpr,fill='tozeroy',
                fillcolor='rgba(201,168,76,.07)',line=dict(color=GOLD,width=2),name='ROC'))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],
                line=dict(color=T['text_muted'],width=1,dash='dash'),name='Rastgele'))
            fig.update_xaxes(title='False Positive Rate')
            fig.update_yaxes(title='True Positive Rate')
            fig.update_layout(title="📊 ROC Eğrisi (Demo)",height=400,**PL())
            st.plotly_chart(fig, use_container_width=True)


# ── MÜŞTERİ DETAY ──
elif sayfa == "👤 Müşteri Detay":
    st.markdown(section_header("Müşteri Detay","360° müşteri profili"), unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner=False)
    def load_detail():
        import sqlite3 as sql3
        conn = sql3.connect("C:/financeai/data/financeai.db")
        try:    ml_df   = pd.read_sql("SELECT * FROM client_ml", conn)
        except: ml_df   = pd.DataFrame()
        try:    risk_df = pd.read_sql("SELECT * FROM client_risk", conn)
        except: risk_df = pd.DataFrame()
        conn.close()
        try:
            users_df = pd.read_csv("C:/financeai/data/users_data.csv")
            cards_df = pd.read_csv("C:/financeai/data/cards_data.csv")
            for c in ['per_capita_income','yearly_income','total_debt']:
                if c in users_df.columns:
                    users_df[c] = users_df[c].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)
            cards_df['credit_limit'] = cards_df['credit_limit'].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).astype(float)
        except:
            users_df = pd.DataFrame(); cards_df = pd.DataFrame()
        return ml_df, risk_df, users_df, cards_df

    ml_df, risk_df, users_df, cards_df = load_detail()
    id_list = sorted(ml_df['client_id'].astype(int).tolist()) if len(ml_df)>0 else \
              sorted(risk_df['client_id'].astype(int).tolist()) if len(risk_df)>0 else []

    if id_list:
        col_s,_ = st.columns([2,3])
        with col_s:
            secili = st.selectbox("Müşteri",id_list,format_func=lambda x:f"Müşteri #{x}")
        rr = risk_df[risk_df['client_id']==secili].iloc[0] if len(risk_df)>0 and secili in risk_df['client_id'].values else None
        ur = users_df[users_df['id']==secili].iloc[0]     if len(users_df)>0 and secili in users_df['id'].values else None
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        with k1: st.metric("💰 Harcama",f"${rr['toplam']:,.0f}" if rr is not None else "—")
        with k2: st.metric("📊 İşlem",f"{int(rr['islem']):,}" if rr is not None else "—")
        with k3: st.metric("⚠️ Risk",f"{rr['risk_skoru']:.1f}" if rr is not None else "—")
        with k4: st.metric("🏦 Kredi",f"{int(ur['credit_score'])}" if ur is not None else "—")
        with k5: st.metric("👤 Yaş",f"{int(ur['current_age'])}" if ur is not None else "—")
    else:
        st.info("📡 Müşteri detay verisi için veritabanı bağlantısını kontrol edin.")


# ═══════════════════════════════════════════
# YÖNETİCİ PANELİ
# ═══════════════════════════════════════════

elif sayfa == "⚙️ Yönetici":
    if current_role != "admin":
        st.error("⛔ Bu sayfaya erişim yetkiniz yok.")
        st.stop()

    st.markdown(section_header("Yönetici Paneli","Kullanıcı yönetimi ve sistem kontrolü"), unsafe_allow_html=True)

    login_stats = get_login_stats()
    all_users   = get_all_users()
    pending     = [u for u in all_users if u['status']=='pending']
    active      = [u for u in all_users if u['status']=='active']
    rejected    = [u for u in all_users if u['status']=='rejected']

    s1,s2,s3,s4,s5 = st.columns(5)
    with s1: st.metric("👥 Toplam",       f"{len(all_users):,}")
    with s2: st.metric("✅ Aktif",        f"{len(active):,}")
    with s3: st.metric("⏳ Onay Bekliyor",f"{len(pending):,}")
    with s4: st.metric("❌ Reddedilen",   f"{len(rejected):,}")
    with s5: st.metric("🔑 Top. Giriş",  f"{login_stats['toplam']:,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["⏳ Onay Bekleyenler","👥 Tüm Kullanıcılar","🔧 Sistem"])

    with tab1:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        if pending:
            st.markdown(f"""
            <div style='background:rgba(255,107,53,.07);border:1px solid rgba(255,107,53,.2);
                        border-radius:12px;padding:14px 18px;margin-bottom:18px;'>
                <div style='font-family:Syne;font-size:.9rem;font-weight:700;color:{ORANGE};margin-bottom:4px;'>
                    ⏳ {len(pending)} kullanıcı onay bekliyor
                </div>
                <div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};'>
                    Rol belirleyip Onayla veya Reddet'e tıklayın
                </div>
            </div>
            """, unsafe_allow_html=True)

            for u in pending:
                pc1,pc2,pc3,pc4,pc5 = st.columns([2.2,1.5,1.2,.9,.9])
                with pc1:
                    st.markdown(f"""
                    <div style='padding:8px 0;'>
                        <div style='font-family:Syne;font-size:.88rem;font-weight:700;color:{T['text_primary']};'>
                            {u['username']}
                        </div>
                        <div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};'>
                            {u['email']}
                        </div>
                        <div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};'>
                            {u['display_name']} · {(u.get('created_at','')[:10])}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with pc2:
                    st.markdown(f"<div style='font-family:DM Mono;font-size:.6rem;color:{ORANGE};padding-top:12px;'>⏳ Bekliyor</div>", unsafe_allow_html=True)
                with pc3:
                    new_role = st.selectbox("",["viewer","analyst","admin"],
                        index=["viewer","analyst","admin"].index(u['role']) if u['role'] in ["viewer","analyst","admin"] else 0,
                        key=f"rp_{u['id']}", label_visibility="collapsed")
                with pc4:
                    st.markdown('<div class="btn-green">', unsafe_allow_html=True)
                    if st.button("✅ Onayla", key=f"ap_{u['id']}", use_container_width=True):
                        update_user_role(u['id'], new_role)
                        approve_user(u['id'], st.session_state.username)
                        st.success(f"✅ {u['username']} onaylandı ({new_role})!")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with pc5:
                    st.markdown('<div class="btn-red">', unsafe_allow_html=True)
                    if st.button("❌ Reddet", key=f"rj_{u['id']}", use_container_width=True):
                        reject_user(u['id'])
                        st.warning(f"❌ {u['username']} reddedildi.")
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f"<div style='height:1px;background:{T['border']};margin:4px 0;'></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:rgba(0,227,150,.05);border:1px solid rgba(0,227,150,.18);
                        border-radius:12px;padding:30px;text-align:center;'>
                <div style='font-size:2rem;margin-bottom:8px;'>✅</div>
                <div style='font-family:Syne;font-size:.9rem;font-weight:700;color:{GREEN};'>
                    Onay bekleyen kullanıcı yok
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        for u in all_users:
            sc = {"active":GREEN,"pending":ORANGE,"rejected":RED}.get(u['status'],T['text_muted'])
            sl = {"active":"✅ Aktif","pending":"⏳ Bekliyor","rejected":"❌ Reddedildi"}.get(u['status'],"—")
            av = {"admin":"👨‍💼","analyst":"📊","viewer":"👁️"}.get(u['role'],"👤")

            uc1,uc2,uc3,uc4,uc5,uc6,uc7 = st.columns([.4,2,1.5,1.2,.9,1.2,.7])
            with uc1:
                st.markdown(f"<div style='font-size:1.2rem;padding-top:8px;text-align:center;'>{av}</div>", unsafe_allow_html=True)
            with uc2:
                st.markdown(f"""
                <div style='padding:5px 0;'>
                    <div style='font-family:Syne;font-size:.82rem;font-weight:700;color:{T['text_primary']};'>{u['username']}</div>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};'>{u['email']}</div>
                </div>""", unsafe_allow_html=True)
            with uc3:
                st.markdown(f"<div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};padding-top:9px;'>{u['display_name'] or '—'}</div>", unsafe_allow_html=True)
            with uc4:
                new_r = st.selectbox("",["viewer","analyst","admin"],
                    index=["viewer","analyst","admin"].index(u['role']) if u['role'] in ["viewer","analyst","admin"] else 0,
                    key=f"cr_{u['id']}", label_visibility="collapsed")
            with uc5:
                if new_r != u['role']:
                    st.markdown('<div class="btn-purple">', unsafe_allow_html=True)
                    if st.button("💾",key=f"sr_{u['id']}",use_container_width=True):
                        update_user_role(u['id'],new_r)
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            with uc6:
                st.markdown(f"<div style='font-family:DM Mono;font-size:.62rem;padding-top:10px;'><span style='color:{sc};'>{sl}</span></div>", unsafe_allow_html=True)
            with uc7:
                if u['username'] != st.session_state.username:
                    st.markdown('<div class="btn-red">', unsafe_allow_html=True)
                    if st.button("🗑️",key=f"dl_{u['id']}",use_container_width=True):
                        delete_user(u['id'])
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='height:1px;background:{T['border']};margin:2px 0;'></div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-family:Syne;font-size:1rem;font-weight:700;color:{T['text_primary']};margin-bottom:14px;'>🔑 Şifre Değiştir</div>", unsafe_allow_html=True)
        pw1,pw2,pw3 = st.columns(3)
        with pw1: old_pw  = st.text_input("Mevcut Şifre",    type="password", key="old_pw")
        with pw2: new_pw  = st.text_input("Yeni Şifre",      type="password", key="new_pw")
        with pw3: new_pw2 = st.text_input("Yeni Şifre Onayı",type="password", key="new_pw2")
        if st.button("🔑 Güncelle"):
            if new_pw != new_pw2: st.error("Şifreler eşleşmiyor!")
            else:
                r = change_password(st.session_state.username, old_pw, new_pw)
                st.success(r["message"]) if r["success"] else st.error(r["message"])

        st.markdown(hr(), unsafe_allow_html=True)

        ls1,ls2,ls3 = st.columns(3)
        with ls1: st.metric("🔑 Toplam Deneme",f"{login_stats['toplam']:,}")
        with ls2: st.metric("✅ Başarılı",f"{login_stats['basarili']:,}")
        with ls3: st.metric("❌ Başarısız",f"{login_stats['basarisiz']:,}")

        st.markdown(hr(), unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            ep = st.selectbox("API Endpoint",["/health","/stats","/stats/fraud","/model/metrics"])
            if st.button("▶️ Test Et", use_container_width=True):
                res = api_get(ep)
                if res: st.success("✅ Başarılı!"); st.json(res if isinstance(res,dict) else res[:2])
                else:   st.error("❌ Bağlantı başarısız.")
        with c2:
            st.download_button("⬇️ Tüm Müşteri Verisi",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="musteri_tam.csv",mime="text/csv",use_container_width=True)
            if st.button("🔄 Cache Temizle",use_container_width=True):
                st.cache_data.clear(); st.rerun()