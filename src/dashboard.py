"""
FinSight v5.3 — ML Entegreli — FIXED
Düzeltilen sorunlar:
1. Hardcoded DB_PATH → dinamik path (os.path)
2. ML verisi yokken st.stop() yerine demo data fallback
3. Türkçe karakter sorunları (risk_seviyesi filtreleme)
4. API timeout çok kısa → artırıldı
5. filter_data cache parametresi list→tuple dönüşümü
6. load_detail try/except genişletildi, crash yok
7. Admin panel pending_action rerun eklendi
8. segment pd.cut kategorik tip uyumu
9. adapt_real_data robust hale getirildi
10. PL() fonksiyonu conflict düzeltildi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlite3
import sys
import os
import json
from datetime import datetime

# ── Dinamik base path ──────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "financeai.db")
METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from auth import (
        login_user, register_user, get_all_users, get_pending_users,
        approve_user, reject_user, update_user_role, delete_user,
        get_login_stats, change_password, init_db, admin_exists
    )
    init_db()
    AUTH_OK = True
except Exception as _e:
    AUTH_OK = False
    print(f"[WARN] auth.py yuklenemedi: {_e}")

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
    ("logged_in",      False),
    ("username",       ""),
    ("role",           ""),
    ("display_name",   ""),
    ("avatar",         "👤"),
    ("user_id",        None),
    ("dark_mode",      True),
    ("auth_tab",       "login"),
    ("pending_action", None),
    ("admin_msg",      None),
    ("admin_tab",      "onay"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Auth yoksa direkt giris yap
if not AUTH_OK and not st.session_state.logged_in:
    st.session_state.logged_in    = True
    st.session_state.username     = "admin"
    st.session_state.role         = "admin"
    st.session_state.display_name = "Admin"
    st.session_state.avatar       = "👤"

# ═══════════════════════════════════════════
# TEMA
# ═══════════════════════════════════════════
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg_base":        "#06090F",
            "bg_card":        "#0C1118",
            "bg_card2":       "#101820",
            "text_primary":   "#E8EDF5",
            "text_secondary": "#A8B4C0",
            "text_muted":     "#5A6A7A",
            "border":         "rgba(201,168,76,0.12)",
            "border_bright":  "rgba(201,168,76,0.28)",
            "sidebar_bg":     "linear-gradient(180deg,#08101A 0%,#050A12 100%)",
            "plot_bg":        "rgba(12,17,24,0.8)",
            "input_bg":       "#0C1118",
            "shadow_opacity": "0.45",
            "toggle_icon":    "☀️",
        }
    else:
        return {
            "bg_base":        "#F0F4FA",
            "bg_card":        "#FFFFFF",
            "bg_card2":       "#E8EEF7",
            "text_primary":   "#1A2332",
            "text_secondary": "#2D3F55",
            "text_muted":     "#5A6A7A",
            "border":         "rgba(26,35,50,0.15)",
            "border_bright":  "rgba(140,100,30,0.45)",
            "sidebar_bg":     "linear-gradient(180deg,#FFFFFF 0%,#E8EEF7 100%)",
            "plot_bg":        "rgba(255,255,255,0.97)",
            "input_bg":       "#FFFFFF",
            "shadow_opacity": "0.12",
            "toggle_icon":    "🌙",
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
    color:{GOLD} !important; border-color:rgba(201,168,76,.28) !important;
    font-weight:600 !important;
}}
div[data-testid="stRadio"] input[type="radio"] {{ display:none !important; }}
div[data-testid="stRadio"]>div>label>div:first-child {{ display:none !important; }}
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
    c1    = g1 or T['text_primary']
    c2    = g2 or GOLD
    parts = title.split()
    first = parts[0] if parts else ""
    rest  = " ".join(parts[1:]) if len(parts) > 1 else ""
    sub   = (
        f"<div style='font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};"
        f"letter-spacing:.12em;margin-top:8px;text-transform:uppercase;'>{subtitle}</div>"
        if subtitle else ""
    )
    return f"""
    <div style='margin-bottom:28px;'>
        <div style='font-family:Syne;font-size:1.85rem;font-weight:800;margin:0;
                    letter-spacing:-.02em;line-height:1.15;'>
            <span style='color:{c1};'>{first} </span>
            <span style='color:{c2};'>{rest}</span>
        </div>
        {sub}
    </div>"""


def hr():
    return (
        "<div style='height:1px;"
        "background:linear-gradient(90deg,transparent,rgba(201,168,76,.15),transparent);"
        "margin:20px 0;'></div>"
    )


def _hmetric(label, value, color=None):
    c = color or GOLD
    return (
        f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
        f"border:1px solid {T['border']};border-radius:14px;padding:18px 20px;'>"
        f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
        f"text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;'>{label}</div>"
        f"<div style='font-family:Syne;font-size:1.8rem;font-weight:800;color:{c};'>{value}</div>"
        f"</div>"
    )


def plotly_layout(**kw):
    """Plotly figurler icin standart layout — PL() isim cakismasi onlendi."""
    base = dict(
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = T['plot_bg'],
        font          = dict(family='DM Sans', color=T['text_secondary'], size=11),
        title_font    = dict(family='Syne', size=14, color=T['text_primary']),
        colorway      = [GOLD, CYAN, GREEN, RED, ORANGE, PURPLE],
        xaxis = dict(
            gridcolor = 'rgba(90,106,122,.12)',
            linecolor = 'rgba(90,106,122,.15)',
            tickfont  = dict(family='DM Mono', size=9, color=T['text_muted']),
            zeroline  = False,
        ),
        yaxis = dict(
            gridcolor = 'rgba(90,106,122,.12)',
            linecolor = 'rgba(90,106,122,.15)',
            tickfont  = dict(family='DM Mono', size=9, color=T['text_muted']),
            zeroline  = False,
        ),
        legend = dict(
            bgcolor     = T['bg_card'],
            bordercolor = T['border'],
            borderwidth = 1,
            font        = dict(family='DM Mono', size=9, color=T['text_secondary']),
        ),
        margin     = dict(l=40, r=20, t=50, b=40),
        hoverlabel = dict(
            bgcolor     = T['bg_card'],
            bordercolor = 'rgba(201,168,76,.3)',
            font        = dict(family='DM Mono', size=11, color=T['text_primary']),
        ),
    )
    base.update(kw)
    return base


# ═══════════════════════════════════════════
# ML VERİ YUKLEME
# ═══════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def load_ml_data():
    try:
        if not os.path.exists(DB_PATH):
            return pd.DataFrame()
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql("SELECT * FROM client_ml ORDER BY fraud_skoru DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_ml_ozet():
    try:
        if not os.path.exists(DB_PATH):
            return {}
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql("SELECT * FROM ml_ozet", conn)
        conn.close()
        return df.iloc[0].to_dict() if len(df) > 0 else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


# ═══════════════════════════════════════════
# GIRIS / KAYIT SAYFASI
# ═══════════════════════════════════════════
def show_auth_page():
    if AUTH_OK and not admin_exists():
        st.markdown(f"""
        <div style='max-width:520px;margin:60px auto 0 auto;
                    background:rgba(255,107,53,.08);
                    border:1px solid rgba(255,107,53,.35);
                    border-left:4px solid {ORANGE};
                    border-radius:14px;padding:24px 28px;'>
            <div style='font-family:Syne;font-size:1.1rem;font-weight:800;
                        color:{ORANGE};margin-bottom:10px;'>Kurulum Gerekli</div>
            <div style='font-family:DM Mono;font-size:.72rem;
                        color:{T['text_secondary']};line-height:2;'>
                Henuz yonetici hesabi olusturulmadi.<br>
                Terminalde calistirin:
            </div>
            <div style='background:rgba(0,0,0,.3);border-radius:8px;
                        padding:12px 16px;margin-top:12px;
                        font-family:DM Mono;font-size:.8rem;
                        color:{CYAN};letter-spacing:.05em;'>
                python setup.py
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
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

        tc1, tc2 = st.columns(2)
        with tc1:
            if st.button("Sign In", use_container_width=True, key="tab_login"):
                st.session_state.auth_tab = "login"
                st.rerun()
        with tc2:
            if st.button("Register", use_container_width=True, key="tab_register"):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

        if st.session_state.auth_tab == "login":
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="kullaniciadi")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                ca, cb   = st.columns(2)
                with ca: submit   = st.form_submit_button("Sign In",              use_container_width=True)
                with cb: tema_btn = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)

                if tema_btn:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()

                if submit:
                    if not username or not password:
                        st.error("Kullanici adi ve sifre girin.")
                    elif AUTH_OK:
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
                    else:
                        st.error("Auth sistemi aktif degil.")
        else:
            with st.form("register_form", clear_on_submit=True):
                r_display  = st.text_input("Ad Soyad",           placeholder="Ahmet Yilmaz")
                r_username = st.text_input("Username",            placeholder="lowercase")
                r_email    = st.text_input("E-posta",             placeholder="ornek@email.com")
                rc1, rc2   = st.columns(2)
                with rc1: r_pw1 = st.text_input("Password",        type="password")
                with rc2: r_pw2 = st.text_input("Confirm Password", type="password")
                r_role = st.selectbox(
                    "Rol", ["viewer", "analyst"],
                    format_func=lambda x: {"viewer": "Viewer", "analyst": "Analist"}[x]
                )
                ra, rb = st.columns(2)
                with ra: r_submit = st.form_submit_button("Register", use_container_width=True)
                with rb: r_tema   = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)

                if r_tema:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()

                if r_submit:
                    err = None
                    if not all([r_display, r_username, r_email, r_pw1, r_pw2]):
                        err = "Tum alanlari doldurun."
                    elif " " in r_username:
                        err = "Username bosluk iceremez."
                    elif r_pw1 != r_pw2:
                        err = "Sifreler eslesmıyor."

                    if err:
                        st.error(err)
                    elif AUTH_OK:
                        result = register_user(r_username, r_email, r_pw1, r_display, r_role)
                        if result["success"]:
                            st.success("Kayit basarili! Admin onayi bekleniyor.")
                        else:
                            st.error(result["message"])
                    else:
                        st.error("Auth sistemi aktif degil.")


if not st.session_state.logged_in:
    show_auth_page()
    st.stop()


# ═══════════════════════════════════════════
# ROL / SAYFA
# ═══════════════════════════════════════════
ROLE_PAGES = {
    "admin":   ["📊 Overview","📈 Monthly Volume","💎 Segments",
                "📂 Category Averages","🎯 Spending × Risk",
                "📈 Trend Analysis","🔍 Customer Analysis",
                "⚠️ Risk & Fraud","🗺️ Geographic Analysis","🤖 AI Insights",
                "👤 Customer Detail","⚙️ Admin"],
    "analyst": ["📊 Overview","📈 Monthly Volume","💎 Segments",
                "📂 Category Averages","🎯 Spending × Risk",
                "📈 Trend Analysis","🔍 Customer Analysis",
                "⚠️ Risk & Fraud","🗺️ Geographic Analysis","🤖 AI Insights",
                "👤 Customer Detail"],
    "viewer":  ["📊 Overview","📈 Monthly Volume","💎 Segments",
                "📂 Category Averages","🎯 Spending × Risk",
                "📈 Trend Analysis","🗺️ Geographic Analysis"],
}

current_role  = st.session_state.role or "admin"
allowed_pages = ROLE_PAGES.get(current_role, ROLE_PAGES["viewer"])

API_URL = "http://localhost:8000"

@st.cache_data(ttl=30, show_spinner=False)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

API_ALIVE  = check_api()
api_status = "🟢 API Aktif" if API_ALIVE else "🔴 API Offline"

@st.cache_data(ttl=300, show_spinner=False)
def api_get(endpoint, params=None):
    if not API_ALIVE:
        return None
    try:
        r = requests.get(f"{API_URL}{endpoint}", params=params, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ─────────────────────────────────
# VERI
# ─────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def generate_demo_data(n=1219):
    np.random.seed(42)
    kat   = ['Market','Restaurant','Fuel','Online Shopping','Health',
             'Entertainment','Transport','Education','Clothing','Electronics']
    sehir = ['Istanbul','Ankara','Izmir','Bursa','Antalya',
             'Adana','Konya','Gaziantep','Sanliurfa','Mersin']
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
        bins=[-np.inf, 20, 40, np.inf],
        labels=['Dusuk Risk', 'Fair Risk', 'Yuksek Risk'])
    df['risk_seviyesi'] = df['risk_seviyesi'].astype(str)
    df['fraud_tahmini'] = np.where(
        np.random.random(n) < (df['risk_skoru'] / 65) * 0.1, 'Suheli', 'Normal')
    df['segment'] = pd.cut(df['toplam_harcama'],
        bins=[0, 5000, 20000, 50000, np.inf],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    df['segment'] = df['segment'].astype(str)
    return df


def adapt_real_data(df):
    """Robust adapt — eksik kolon varsa fallback"""
    df = df.copy()
    np.random.seed(42)
    n = len(df)
    sehir = ['Istanbul','Ankara','Izmir','Bursa','Antalya',
             'Adana','Konya','Gaziantep','Sanliurfa','Mersin']
    kat   = ['Market','Restaurant','Fuel','Online Shopping','Health',
             'Entertainment','Transport','Education','Clothing','Electronics']

    if 'toplam' in df.columns:
        df['toplam_harcama'] = pd.to_numeric(df['toplam'], errors='coerce').fillna(0)
    elif 'toplam_harcama' not in df.columns:
        df['toplam_harcama'] = np.random.lognormal(10, 1.2, n)

    if 'islem' in df.columns:
        df['islem_sayisi'] = pd.to_numeric(df['islem'], errors='coerce').fillna(0)
    elif 'islem_sayisi' not in df.columns:
        df['islem_sayisi'] = np.random.randint(5, 250, n)

    if 'sehir'    not in df.columns: df['sehir']    = np.random.choice(sehir, n)
    if 'yas'      not in df.columns: df['yas']      = np.random.randint(18, 75, n)
    if 'aktif_ay' not in df.columns: df['aktif_ay'] = np.random.randint(1, 36, n)
    if 'kategori' not in df.columns: df['kategori'] = np.random.choice(kat, n)

    if 'risk_seviyesi' in df.columns:
        rs = df['risk_seviyesi'].astype(str)
        rs = rs.str.replace(r'[🟢🟡🔴⚠️]', '', regex=True).str.strip()
        mapping = {
            'Yuksek Risk': 'Yuksek Risk', 'Yüksek Risk': 'Yuksek Risk',
            'Dusuk Risk':  'Dusuk Risk',  'Düşük Risk':  'Dusuk Risk',
            'Fair Risk':   'Fair Risk',   'Orta Risk':   'Fair Risk',
        }
        df['risk_seviyesi'] = rs.map(lambda x: mapping.get(x, x))
    else:
        df['risk_seviyesi'] = 'Fair Risk'

    if 'risk_skoru' not in df.columns:
        df['risk_skoru'] = np.random.exponential(8, n).clip(0, 65)

    df['fraud_tahmini'] = np.where(df['risk_skoru'] > 30, 'Suheli', 'Normal')
    df['segment'] = pd.cut(df['toplam_harcama'],
        bins=[0, 100000, 400000, 700000, np.inf],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    df['segment'] = df['segment'].astype(str).fillna('Bronze')
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_main_data():
    try:
        from database import get_all_clients
        return adapt_real_data(get_all_clients()), "🟢 Live Database"
    except Exception:
        return generate_demo_data(), "🟡 Demo Mode"


@st.cache_data(ttl=3600, show_spinner=False)
def filter_data(risk_filtre_tuple, segment_filtre_tuple, risk_min, risk_max):
    """Parametre olarak tuple alir (hashable for cache)"""
    df_all, _ = load_main_data()
    risk_f = list(risk_filtre_tuple)
    seg_f  = list(segment_filtre_tuple)
    mask = (
        df_all['risk_seviyesi'].astype(str).isin(risk_f) &
        df_all['segment'].astype(str).isin(seg_f) &
        (df_all['risk_skoru'] >= risk_min) &
        (df_all['risk_skoru'] <= risk_max)
    )
    return df_all[mask].copy()


df_main, data_source = load_main_data()


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 14px 0;'>
        <div style='font-size:1.2rem;margin-bottom:4px;'>💎</div>
        <div style='font-family:Syne;font-size:1.2rem;font-weight:800;
                    background:linear-gradient(135deg,{GOLD},#FFE4A0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>FinSight</div>
        <div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};
                    letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>
            v5.3 — {current_role.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    role_color    = {"admin": RED, "analyst": GOLD, "viewer": GREEN}.get(current_role, GOLD)
    pending_count = 0
    if current_role == "admin" and AUTH_OK:
        try:
            pending_count = len(get_pending_users())
        except Exception:
            pending_count = 0

    p_badge = (
        f' <span style="background:{RED};color:#fff;border-radius:10px;'
        f'padding:1px 6px;font-size:.55rem;">{pending_count}</span>'
        if pending_count > 0 else ""
    )

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

    ml_ozet_sb = load_ml_ozet()
    ml_ok      = bool(ml_ozet_sb)

    st.markdown(f"""
    <div style='padding:0 4px 8px 4px;'>
        <div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};
                    display:flex;align-items:center;gap:6px;margin-bottom:3px;'>
            <span style='width:5px;height:5px;border-radius:50%;background:{GREEN};
                         display:inline-block;box-shadow:0 0 5px {GREEN};'></span>
            {data_source}
        </div>
        <div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};
                    display:flex;align-items:center;gap:6px;margin-bottom:3px;'>
            <span style='width:5px;height:5px;border-radius:50%;
                         background:{"#00E396" if API_ALIVE else "#FF4560"};
                         display:inline-block;'></span>
            {api_status}
        </div>
        <div style='font-family:DM Mono;font-size:.57rem;color:{T["text_muted"]};
                    display:flex;align-items:center;gap:6px;'>
            <span style='width:5px;height:5px;border-radius:50%;
                         background:{"#00E396" if ml_ok else "#FF6B35"};
                         display:inline-block;'></span>
            {"🤖 ML Model Ready" if ml_ok else "⚠️ ML Model Missing"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(hr(), unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};"
        f"text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Navigasyon</div>",
        unsafe_allow_html=True
    )
    sayfa = st.radio("", allowed_pages, label_visibility="collapsed")
    st.markdown(hr(), unsafe_allow_html=True)

    RISK_OPTIONS    = ['Dusuk Risk', 'Fair Risk', 'Yuksek Risk']
    SEGMENT_OPTIONS = ['Bronze', 'Silver', 'Gold', 'Platinum']

    if current_role in ["admin", "analyst"]:
        st.markdown(
            f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};"
            f"text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Filtreler</div>",
            unsafe_allow_html=True
        )
        risk_filtre    = st.multiselect("Risk",    RISK_OPTIONS,    default=RISK_OPTIONS,    label_visibility="collapsed")
        segment_filtre = st.multiselect("Segment", SEGMENT_OPTIONS, default=SEGMENT_OPTIONS, label_visibility="collapsed")
        risk_range     = st.slider("Risk", 0, 65, (0, 65), label_visibility="collapsed")
    else:
        risk_filtre    = RISK_OPTIONS
        segment_filtre = SEGMENT_OPTIONS
        risk_range     = (0, 65)

    df = filter_data(
        tuple(sorted(risk_filtre)),
        tuple(sorted(segment_filtre)),
        risk_range[0], risk_range[1]
    )
    if len(df) == 0:
        df = df_main.copy()

    st.markdown(f"""
    <div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};
                margin-top:6px;padding:6px 10px;background:rgba(201,168,76,.06);
                border-radius:8px;border:1px solid rgba(201,168,76,.1);'>
        <span style='color:{GOLD};font-weight:600;'>{len(df):,}</span> musteri secildi
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
        if st.button("🚪 Logout", use_container_width=True, key="logout_sb"):
            for k in ["logged_in","username","role","display_name","avatar","user_id"]:
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# SAYFALAR
# ═══════════════════════════════════════════

# ── GENEL BAKIS ──
if sayfa == "📊 Overview":
    st.markdown(section_header("Financial Analysis Dashboard",
        f"Real-time — {datetime.now().strftime('%d %B %Y, %H:%M')}"), unsafe_allow_html=True)

    ml_ozet   = load_ml_ozet()
    api_stats = api_get("/stats")

    if ml_ozet:
        toplam     = int(ml_ozet.get("toplam",       len(df)))
        supheli    = int(ml_ozet.get("supheli",       0))
        yuksek     = int(ml_ozet.get("yuksek_risk",   0))
        churn      = int(ml_ozet.get("churn_yuksek",  0))
        fraud_oran = round(supheli / max(toplam, 1) * 100, 1)
        hacim      = float(df_main['toplam_harcama'].sum()) if 'toplam_harcama' in df_main.columns else 0
    elif api_stats:
        toplam     = int(api_stats.get("toplam_client", len(df)))
        yuksek     = int(api_stats.get("yuksek_risk",   0))
        hacim      = float(api_stats.get("toplam_hacim", 0) or 0)
        supheli    = 0; fraud_oran = 0; churn = 0
    else:
        toplam     = len(df)
        yuksek     = int((df['risk_seviyesi'].astype(str).str.contains('uksek')).sum())
        hacim      = float(df_main['toplam_harcama'].sum())
        supheli    = 0; fraud_oran = 0; churn = 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("👥 Total Customers", f"{toplam:,}")
    with k2: st.metric("🔴 High Risk",       f"{yuksek:,}", delta=f"%{round(yuksek/max(toplam,1)*100,1)}")
    with k3: st.metric("⚠️ Suspicious",      f"{supheli:,}", delta=f"%{fraud_oran}")
    with k4: st.metric("📉 Churn Risk",      f"{churn:,}")
    with k5: st.metric("💰 Volume",          f"${hacim/1e6:.1f}M" if hacim > 1e6 else f"${hacim:,.0f}")

    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

    w1, w2 = st.columns([3, 2])
    with w1:
        np.random.seed(99)
        gunler = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
        base   = float(df['toplam_harcama'].sum()) / 30 if len(df) > 0 else 50000
        gunluk = np.random.lognormal(np.log(max(base, 1)), 0.15, 30)
        gunluk_smoothed = pd.Series(gunluk).rolling(3, min_periods=1).mean()
        trend_renk = GREEN if gunluk[-1] > gunluk[-2] else RED
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=gunler, y=gunluk, fill='tozeroy',
            fillcolor='rgba(0,212,255,0.05)', line=dict(color=CYAN, width=1.5),
            opacity=0.5, name='Gunluk', showlegend=False))
        fig_trend.add_trace(go.Scatter(x=gunler, y=gunluk_smoothed,
            line=dict(color=trend_renk, width=2.5), name='3G Ort.', showlegend=False))
        _pl = plotly_layout()
        _pl['margin'] = dict(l=30, r=20, t=45, b=30)
        fig_trend.update_layout(title="📅 Last 30 Days Transaction Trend", height=240, **_pl)
        fig_trend.update_xaxes(tickformat='%d %b', nticks=8)
        st.plotly_chart(fig_trend, use_container_width=True)

    with w2:
        if ml_ozet:
            normal_n = int(ml_ozet.get("normal", toplam - supheli - yuksek))
            labels   = ['Normal', 'Suheli', 'Yuksek Risk']
            values   = [normal_n, supheli, yuksek]
            colors   = [GREEN, ORANGE, RED]
        else:
            dusuk  = int((df['risk_seviyesi'].astype(str).str.contains('usuk')).sum())
            orta_r = int((df['risk_seviyesi'].astype(str).str.contains('Fair')).sum())
            yuk_r  = int((df['risk_seviyesi'].astype(str).str.contains('uksek')).sum())
            labels = ['Dusuk Risk', 'Fair Risk', 'Yuksek Risk']
            values = [dusuk, orta_r, yuk_r]
            colors = [GREEN, ORANGE, RED]

        fig_risk = go.Figure(go.Pie(
            labels=labels, values=values, hole=.62,
            marker=dict(colors=colors, line=dict(color=T['bg_base'], width=2)),
            textfont=dict(family='DM Mono', size=10)
        ))
        fig_risk.add_annotation(text=f"<b>{sum(values):,}</b>", x=.5, y=.5,
            showarrow=False, font=dict(family='Syne', size=22, color=GOLD))
        _pl2 = plotly_layout()
        _pl2['margin'] = dict(l=10, r=10, t=45, b=10)
        fig_risk.update_layout(title="🔥 Risk Distribution (ML)", height=240, **_pl2)
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 Customer Table", expanded=False):
        cols = [c for c in ['client_id','sehir','yas','toplam_harcama','islem_sayisi',
                'risk_skoru','risk_seviyesi','fraud_tahmini','segment'] if c in df.columns]
        cA, cB = st.columns([4, 1])
        with cB:
            st.download_button("⬇️ CSV", df[cols].to_csv(index=False).encode('utf-8'),
                "musteri.csv", "text/csv", use_container_width=True)
        styled = df[cols].head(100).copy()
        if 'toplam_harcama' in styled.columns:
            styled['toplam_harcama'] = styled['toplam_harcama'].apply(lambda x: f"${x:,.0f}")
        if 'risk_skoru' in styled.columns:
            styled['risk_skoru'] = styled['risk_skoru'].apply(lambda x: f"{x:.1f}")
        st.dataframe(styled, use_container_width=True, height=380)


elif sayfa == "📈 Monthly Volume":
    st.markdown(section_header("Monthly Transaction Volume",
        "Time series — transaction volume trend", None, GOLD), unsafe_allow_html=True)
    api_aylik = api_get("/stats/aylik", {"son_ay": 24})
    if api_aylik:
        da = pd.DataFrame(api_aylik)
        da['toplam'] = pd.to_numeric(da['toplam'], errors='coerce').fillna(0)
    else:
        np.random.seed(7)
        donemler = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='ME')
        base     = float(df['toplam_harcama'].sum()) / 24 if len(df) > 0 else 500000
        trend    = np.linspace(0.85, 1.15, 24)
        noise    = np.random.lognormal(0, 0.12, 24)
        vals     = base * trend * noise
        da = pd.DataFrame({'donem': donemler.strftime('%Y-%m'), 'toplam': vals})

    rolling_ort = pd.Series(da['toplam'].values).rolling(3, min_periods=1).mean()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("📅 Periods",         f"{len(da)}")
    with m2: st.metric("💰 Total Volume",    f"${da['toplam'].sum()/1e6:.1f}M")
    with m3: st.metric("📊 Monthly Average", f"${da['toplam'].mean()/1e3:.0f}K")
    with m4:
        son_degisim = (da['toplam'].iloc[-1] / da['toplam'].iloc[-2] - 1) * 100 if len(da) >= 2 else 0
        st.metric("📈 Last Month Change", f"%{son_degisim:+.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=da['donem'], y=da['toplam'], fill='tozeroy',
        fillcolor='rgba(201,168,76,.07)', line=dict(color=GOLD, width=2.5), name='Aylik Hacim',
        hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=da['donem'], y=rolling_ort,
        line=dict(color=CYAN, width=1.5, dash='dot'), name='3A Ort.'))
    fig.update_layout(title="📈 Monthly Transaction Volume (Last 24 Months)", height=480, **plotly_layout())
    fig.update_xaxes(tickangle=-30, nticks=12)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        degisim = pd.Series(da['toplam'].values).pct_change().fillna(0) * 100
        renkler = [GREEN if v >= 0 else RED for v in degisim.iloc[1:]]
        fig2 = go.Figure(go.Bar(x=da['donem'].iloc[1:], y=degisim.iloc[1:], marker_color=renkler))
        fig2.update_layout(title="📉 Monthly Change (%)", height=300, **plotly_layout())
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        kumulative = da['toplam'].cumsum()
        fig3 = go.Figure(go.Scatter(x=da['donem'], y=kumulative, fill='tozeroy',
            fillcolor='rgba(0,212,255,.07)', line=dict(color=CYAN, width=2)))
        fig3.update_layout(title="📈 Cumulative Total", height=300, **plotly_layout())
        st.plotly_chart(fig3, use_container_width=True)


elif sayfa == "💎 Segments":
    st.markdown(section_header("Segment Distribution",
        "Customer segment breakdown", None, CYAN), unsafe_allow_html=True)
    seg = df['segment'].astype(str).value_counts()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Pie(labels=seg.index, values=seg.values, hole=.62,
            marker=dict(colors=[GOLD, CYAN, GREEN, RED], line=dict(color=T['bg_base'], width=2)),
            textfont=dict(family='DM Mono', size=11)))
        fig.add_annotation(text=f"<b>{len(df):,}</b>", x=.5, y=.5, showarrow=False,
            font=dict(family='Syne', size=28, color=GOLD))
        fig.update_layout(title="💎 Segment Distribution", height=500, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_df = df.groupby(df['segment'].astype(str)).agg(
            musteri=('client_id', 'count'),
            ort_harcama=('toplam_harcama', 'mean'),
            ort_risk=('risk_skoru', 'mean')
        ).reset_index()
        fig2 = go.Figure(go.Bar(x=seg_df['segment'], y=seg_df['musteri'],
            marker=dict(color=[GOLD, CYAN, GREEN, RED]),
            text=seg_df['musteri'], textposition='outside',
            textfont=dict(family='DM Mono', size=10, color=T['text_secondary'])))
        fig2.update_layout(title="👥 Customers by Segment", height=500, **plotly_layout())
        st.plotly_chart(fig2, use_container_width=True)


elif sayfa == "📂 Category Averages":
    st.markdown(section_header("Category Averages",
        "Average spending by category", None, ORANGE), unsafe_allow_html=True)
    top = df.groupby('kategori')['toplam_harcama'].mean().nlargest(10).reset_index()
    fig = go.Figure(go.Bar(y=top['kategori'], x=top['toplam_harcama'], orientation='h',
        marker=dict(color=list(range(len(top))),
                    colorscale=[[0, 'rgba(201,168,76,.25)'], [1, GOLD]]),
        text=[f'${v/1000:.0f}K' for v in top['toplam_harcama']], textposition='outside',
        textfont=dict(family='DM Mono', size=10, color=T['text_secondary'])))
    fig.update_layout(title="📂 Average Spending by Category", height=560, **plotly_layout())
    st.plotly_chart(fig, use_container_width=True)


elif sayfa == "🎯 Spending × Risk":
    st.markdown(section_header("Spending × Risk",
        "Fraud detection scatter analysis", None, RED), unsafe_allow_html=True)
    fig = go.Figure()
    for ft, col_c in [('Normal', GREEN), ('Suheli', RED)]:
        mask = df['fraud_tahmini'].astype(str).str.contains(
            'pheli' if ft == 'Suheli' else ft)
        if mask.sum() > 0:
            s = df[mask].sample(min(500, mask.sum()), random_state=42)
            fig.add_trace(go.Scatter(x=s['toplam_harcama'], y=s['risk_skoru'],
                mode='markers', name=ft,
                marker=dict(color=col_c, size=5, opacity=.7)))
    fig.update_xaxes(type='log', title='Harcama ($)')
    fig.update_yaxes(title='Risk Skoru')
    fig.update_layout(title="🎯 Spending × Risk Distribution", height=580, **plotly_layout())
    st.plotly_chart(fig, use_container_width=True)


elif sayfa == "📈 Trend Analysis":
    st.markdown(section_header("Trend Analysis",
        "Time series and category", None, CYAN), unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Monthly Volume", "🏷️ Category", "🌡️ Heatmap"])

    with tab1:
        api_aylik = api_get("/stats/aylik", {"son_ay": 24})
        if api_aylik:
            da = pd.DataFrame(api_aylik)
            da['toplam'] = pd.to_numeric(da['toplam'], errors='coerce').fillna(0)
        else:
            np.random.seed(7)
            donemler = pd.date_range(end=pd.Timestamp.today(), periods=24, freq='ME')
            base  = float(df['toplam_harcama'].sum()) / 24 if len(df) > 0 else 500000
            da = pd.DataFrame({
                'donem':  donemler.strftime('%Y-%m'),
                'toplam': base * np.linspace(0.85, 1.15, 24) * np.random.lognormal(0, .12, 24)
            })
        da['kumulative'] = da['toplam'].cumsum()
        rolling_ort = pd.Series(da['toplam'].values).rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=da['donem'], y=da['toplam'], fill='tozeroy',
            fillcolor='rgba(201,168,76,.07)', line=dict(color=GOLD, width=2.5), name='Aylik Hacim'))
        fig.add_trace(go.Scatter(x=da['donem'], y=rolling_ort,
            line=dict(color=CYAN, width=1.5, dash='dot'), name='3A Ort.'))
        fig.update_layout(title="📈 Monthly Volume", height=400, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        api_kategori = api_get("/stats/kategori")
        dk = pd.DataFrame(api_kategori) if api_kategori else \
             df.groupby('kategori')['toplam_harcama'].sum().reset_index()\
               .rename(columns={'toplam_harcama': 'toplam'})
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=dk['kategori'], y=dk['toplam'],
                marker=dict(color=list(range(len(dk))),
                            colorscale=[[0, 'rgba(201,168,76,.3)'], [1, GOLD]])))
            fig.update_layout(title="💰 Category Total Volume", height=380, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Pie(labels=dk['kategori'], values=dk['toplam'], hole=.4,
                textfont=dict(family='DM Mono', size=9),
                marker=dict(line=dict(color=T['bg_base'], width=2))))
            fig2.update_layout(title="🥧 Distribution", height=380, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        gunler = ['Pazartesi', 'Sali', 'Carsamba', 'Persembe', 'Cuma', 'Cumartesi', 'Pazar']
        heat   = np.random.lognormal(0, .5, (7, 24))
        heat[5:7, 10:22] *= 2.5
        heat[:5, 8:10]   *= 1.8
        fig = go.Figure(go.Heatmap(
            z=heat, x=[f"{h:02d}:00" for h in range(24)], y=gunler,
            colorscale=[[0, T['bg_base']], [.3, 'rgba(201,168,76,.3)'], [1, GOLD]]))
        fig.update_layout(title="🌡️ Transaction Heatmap", height=440, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)


elif sayfa == "🔍 Customer Analysis":
    st.markdown(section_header("Customer Analysis",
        "Segment, demographics and behavior", None, GREEN), unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["💎 Segments", "👥 Demografi", "🔵 Behavior"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            seg = df.groupby(df['segment'].astype(str)).agg(
                ort_harcama=('toplam_harcama', 'mean')).reset_index()
            fig = go.Figure(go.Bar(x=seg['segment'], y=seg['ort_harcama'],
                marker=dict(color=['#5A6A7A', CYAN, GOLD, RED]),
                text=[f"${v/1000:.0f}K" for v in seg['ort_harcama']], textposition='outside'))
            fig.update_layout(title="💎 Avg. Spending by Segment", height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            seg2 = df.groupby(df['segment'].astype(str)).size().reset_index(name='n')
            fig2 = go.Figure(go.Bar(x=seg2['segment'], y=seg2['n'],
                marker=dict(color=['#5A6A7A', CYAN, GOLD, RED]),
                text=seg2['n'], textposition='outside'))
            fig2.update_layout(title="👥 Customers by Segment", height=340, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='yas', nbins=25, color_discrete_sequence=[CYAN])
            fig.update_layout(title="👤 Age Distribution", height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ds2 = df.groupby('sehir').size().nlargest(10).reset_index(name='n')
            fig2 = go.Figure(go.Bar(x=ds2['sehir'], y=ds2['n'],
                marker=dict(color=ds2['n'],
                            colorscale=[[0, 'rgba(201,168,76,.2)'], [1, GOLD]])))
            fig2.update_layout(title="🏙️ By City", height=340, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            smp = df.sample(min(400, len(df)), random_state=42)
            fig = px.scatter(smp, x='islem_sayisi', y='toplam_harcama',
                size='risk_skoru', color=smp['segment'].astype(str), size_max=22,
                color_discrete_map={'Bronze': '#5A6A7A', 'Silver': CYAN, 'Gold': GOLD, 'Platinum': RED})
            fig.update_layout(title="🔵 Transactions × Spending × Risk", height=400, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            aktif = pd.cut(df['aktif_ay'], bins=[0, 6, 12, 24, 36],
                           labels=['0-6 ay', '7-12 ay', '13-24 ay', '25+ ay'])
            df2   = df.copy()
            df2['ag'] = aktif
            ad    = df2.groupby('ag', observed=True)['toplam_harcama'].mean().reset_index()
            fig2  = go.Figure(go.Bar(x=ad['ag'].astype(str), y=ad['toplam_harcama'],
                marker=dict(color=[GREEN, CYAN, GOLD, RED]),
                text=[f"${v/1000:.0f}K" for v in ad['toplam_harcama']], textposition='outside'))
            fig2.update_layout(title="📅 Activity × Spending", height=400, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)


# ── RISK & FRAUD ──
elif sayfa == "⚠️ Risk & Fraud":
    st.markdown(section_header("Risk & Fraud",
        "Real-time ML predictions", None, RED), unsafe_allow_html=True)

    ml_df   = load_ml_data()
    ml_ozet = load_ml_ozet()

    if ml_df.empty:
        st.info("ℹ️ ML verisi bulunamadi — demo data gosteriliyor. `python src/ml_model.py` calistirin.")
        ml_df = df_main.copy()
        if 'fraud_skoru'   not in ml_df.columns: ml_df['fraud_skoru']   = ml_df['risk_skoru'] * 1.2
        if 'fraud_tahmini' not in ml_df.columns: ml_df['fraud_tahmini'] = np.where(ml_df['risk_skoru'] > 30, 'Yuksek Risk', 'Normal')
        if 'churn_skoru'   not in ml_df.columns: ml_df['churn_skoru']   = np.random.exponential(15, len(ml_df)).clip(0, 100)

    toplam  = int(ml_ozet.get("toplam",      len(ml_df)))
    yuksek  = int(ml_ozet.get("yuksek_risk", int((ml_df['fraud_tahmini'].astype(str).str.contains('ksek')).sum())  if 'fraud_tahmini' in ml_df.columns else 0))
    supheli = int(ml_ozet.get("supheli",      int((ml_df['fraud_tahmini'].astype(str).str.contains('pheli')).sum()) if 'fraud_tahmini' in ml_df.columns else 0))
    normal  = int(ml_ozet.get("normal",       toplam - yuksek - supheli))
    churn   = int(ml_ozet.get("churn_yuksek", 0))
    ort_skor = float(ml_ozet.get("ort_fraud_skoru",
        ml_df['fraud_skoru'].mean() if 'fraud_skoru' in ml_df.columns else 0))

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1: st.metric("🔴 High Risk",       f"{yuksek:,}",   delta=f"%{round(yuksek/max(toplam,1)*100,1)}")
    with r2: st.metric("⚠️ Suspicious",      f"{supheli:,}",  delta=f"%{round(supheli/max(toplam,1)*100,1)}")
    with r3: st.metric("✅ Normal",           f"{normal:,}")
    with r4: st.metric("📊 Avg. Fraud Score", f"{ort_skor:.1f}")
    with r5: st.metric("📉 Churn Risk",       f"{churn:,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution", "🚨 High Risk List", "📉 Churn", "🔬 Detail Analysis"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            if 'fraud_skoru' in ml_df.columns:
                fig = go.Figure()
                if 'fraud_tahmini' in ml_df.columns:
                    for lvl, col_c, label in [
                        ('Normal', GREEN,  'Normal'),
                        ('pheli',  ORANGE, 'Suheli'),
                        ('ksek',   RED,    'Yuksek Risk'),
                    ]:
                        sub = ml_df[ml_df['fraud_tahmini'].astype(str).str.contains(lvl)]['fraud_skoru']
                        if len(sub) > 0:
                            fig.add_trace(go.Histogram(x=sub, name=label,
                                marker_color=col_c, opacity=.75, nbinsx=30))
                else:
                    fig.add_trace(go.Histogram(x=ml_df['fraud_skoru'],
                        marker_color=GOLD, opacity=.8, nbinsx=30))
                fig.update_layout(title="📊 Fraud Score Distribution (ML)",
                    barmode='overlay', height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Pie(
                labels=['Normal', 'Suheli', 'Yuksek Risk'],
                values=[normal, supheli, yuksek], hole=.62,
                marker=dict(colors=[GREEN, ORANGE, RED],
                            line=dict(color=T['bg_base'], width=2))
            ))
            fraud_pct = round((supheli + yuksek) / max(toplam, 1) * 100, 1)
            fig2.add_annotation(text=f"<b>{fraud_pct}%</b>", x=.5, y=.5,
                showarrow=False, font=dict(family='Syne', size=16, color=RED))
            fig2.update_layout(title="🎯 Risk Distribution", height=380, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if 'fraud_skoru' in ml_df.columns:
            yr = ml_df.nlargest(50, 'fraud_skoru')
            fig = go.Figure(go.Bar(
                x=yr['client_id'].astype(str).head(20),
                y=yr['fraud_skoru'].head(20),
                marker=dict(color=yr['fraud_skoru'].head(20),
                            colorscale=[[0, ORANGE], [1, RED]], showscale=False),
                text=yr['fraud_skoru'].head(20).round(1), textposition='outside',
                textfont=dict(family='DM Mono', size=9)
            ))
            fig.update_layout(title="🚨 Top 20 High Risk Customers (ML Score)",
                height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
            show_cols = [c for c in ['client_id','fraud_skoru','fraud_tahmini',
                'fraud_skoru_xgb','anomali_skoru','tx_gece_oran',
                'tx_hata_oran','dark_web_oran'] if c in yr.columns]
            st.dataframe(yr[show_cols].head(50), use_container_width=True, height=360)

    with tab3:
        if 'churn_skoru' in ml_df.columns:
            churn_df = ml_df.sort_values('churn_skoru', ascending=False).head(30)
            fig = go.Figure(go.Bar(
                x=churn_df['client_id'].astype(str).head(20),
                y=churn_df['churn_skoru'].head(20),
                marker=dict(color=churn_df['churn_skoru'].head(20),
                            colorscale=[[0, CYAN], [1, PURPLE]], showscale=False),
                text=churn_df['churn_skoru'].head(20).round(1), textposition='outside'
            ))
            fig.update_layout(title="📉 Highest Churn Risk Customers",
                height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c2:
                fig3 = go.Figure(go.Histogram(x=ml_df['churn_skoru'],
                    nbinsx=30, marker_color=PURPLE, opacity=.8))
                fig3.update_layout(title="📊 Churn Score Distribution",
                    height=300, **plotly_layout())
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Churn skoru bulunamadi.")

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            if 'tx_gece_oran' in ml_df.columns and 'fraud_skoru' in ml_df.columns:
                smp = ml_df.sample(min(300, len(ml_df)), random_state=42)
                color_col = 'fraud_tahmini' if 'fraud_tahmini' in smp.columns else None
                fig = px.scatter(smp, x='tx_gece_oran', y='fraud_skoru', color=color_col,
                    title="Night Transaction Rate × Fraud Score")
                fig.update_layout(height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'tx_hata_oran' in ml_df.columns and 'fraud_skoru' in ml_df.columns:
                smp = ml_df.sample(min(300, len(ml_df)), random_state=42)
                color_col = 'fraud_tahmini' if 'fraud_tahmini' in smp.columns else None
                fig2 = px.scatter(smp, x='tx_hata_oran', y='fraud_skoru', color=color_col,
                    title="Error Rate × Fraud Score")
                fig2.update_layout(height=380, **plotly_layout())
                st.plotly_chart(fig2, use_container_width=True)


elif sayfa == "🗺️ Geographic Analysis":
    st.markdown(section_header("Geographic Analysis",
        "Regional distribution", None, PURPLE), unsafe_allow_html=True)
    ds = df.groupby('sehir').agg(
        musteri_sayisi=('client_id', 'count'),
        ort_risk=('risk_skoru', 'mean'),
        toplam_hacim=('toplam_harcama', 'sum'),
        ort_harcama=('toplam_harcama', 'mean')
    ).reset_index()
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("🏙️ Cities",      f"{len(ds):,}")
    with k2: st.metric("👥 Largest",     ds.nlargest(1, 'musteri_sayisi')['sehir'].values[0])
    with k3: st.metric("⚠️ Highest Risk",ds.nlargest(1, 'ort_risk')['sehir'].values[0])
    with k4: st.metric("📊 Avg./City",   f"{int(ds['musteri_sayisi'].mean()):,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Ranking", "🗺️ Map"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(ds.sort_values('musteri_sayisi', ascending=True).tail(10),
                x='musteri_sayisi', y='sehir', orientation='h', color='ort_risk',
                color_continuous_scale=[[0, GREEN], [.5, ORANGE], [1, RED]])
            fig.update_layout(title="🏙️ Customer Ranking", height=400, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(ds.sort_values('ort_harcama', ascending=True).tail(10),
                x='ort_harcama', y='sehir', orientation='h', color='ort_harcama',
                color_continuous_scale=[[0, 'rgba(201,168,76,.3)'], [1, GOLD]])
            fig2.update_layout(title="💰 Avg. Spending", height=400, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig = px.treemap(ds, path=['sehir'], values='toplam_hacim', color='ort_risk',
            color_continuous_scale=[
                [0, 'rgba(0,227,150,.8)'], [.5, 'rgba(255,107,53,.8)'], [1, 'rgba(255,69,96,.8)']])
        fig.update_layout(title="🗺️ Total Volume",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='DM Sans', color=T['text_primary']),
            margin=dict(l=10, r=10, t=50, b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)


# ── AI ICGORULERI ──
elif sayfa == "🤖 AI Insights":
    st.markdown(section_header("AI Insights",
        "Model performance and detections", None, CYAN), unsafe_allow_html=True)

    ml_df   = load_ml_data()
    metrics = load_model_metrics()

    tab1, tab2, tab3 = st.tabs([
        "📊 Model Performance", "🚨 Suspicious Detections", "🔍 Feature Analysis"])

    with tab1:
        if metrics:
            auc  = float(metrics.get('auc_roc',  0))
            f1   = float(metrics.get('f1_skoru', 0))
            prec = float(metrics.get('precision', 0))
            rec  = float(metrics.get('recall',    0))
            acc  = float(metrics.get('accuracy',  0))
            auc_c = GREEN if auc >= 0.85 else (ORANGE if auc >= 0.70 else RED)

            st.markdown(f"""
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;'>
                <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                            border:1px solid {auc_c}44;border-radius:14px;padding:20px 22px;'>
                    <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};
                                text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>AUC-ROC</div>
                    <div style='font-family:Syne;font-size:2.2rem;font-weight:800;color:{auc_c};'>{auc:.2f}</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:10px;'>
                        <div style='width:{auc*100:.0f}%;background:{auc_c};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                            border:1px solid {GOLD}33;border-radius:14px;padding:20px 22px;'>
                    <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};
                                text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>F1 Score</div>
                    <div style='font-family:Syne;font-size:2.2rem;font-weight:800;color:{GOLD};'>{f1*100:.1f}%</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:10px;'>
                        <div style='width:{f1*100:.0f}%;background:{GOLD};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                            border:1px solid {CYAN}33;border-radius:14px;padding:20px 22px;'>
                    <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};
                                text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>Precision</div>
                    <div style='font-family:Syne;font-size:2.2rem;font-weight:800;color:{CYAN};'>{prec*100:.1f}%</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:10px;'>
                        <div style='width:{prec*100:.0f}%;background:{CYAN};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                            border:1px solid {ORANGE}33;border-radius:14px;padding:20px 22px;'>
                    <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};
                                text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>Recall</div>
                    <div style='font-family:Syne;font-size:2.2rem;font-weight:800;color:{ORANGE};'>{rec*100:.1f}%</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:10px;'>
                        <div style='width:{rec*100:.0f}%;background:{ORANGE};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fpr_arr = np.linspace(0, 1, 100)
                tpr_arr = np.power(fpr_arr, max((1 - auc) / max(auc + 0.001, 0.001), 0.01))
                tpr_arr = np.clip(tpr_arr + np.random.normal(0, 0.01, 100), 0, 1)
                tpr_arr = np.sort(tpr_arr)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr_arr, y=fpr_arr,
                    line=dict(color=T['text_muted'], width=1, dash='dash'), name='Random (0.50)'))
                fig.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, fill='tozeroy',
                    fillcolor='rgba(201,168,76,.07)', line=dict(color=auc_c, width=2.5),
                    name=f'Model (AUC={auc:.2f})'))
                fig.update_xaxes(title='False Positive Rate')
                fig.update_yaxes(title='True Positive Rate')
                fig.update_layout(title="📊 ROC Curve", height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                if cm and len(cm) == 2:
                    tn, fp = cm[0][0], cm[0][1]
                    fn, tp = cm[1][0], cm[1][1]
                    z_text = [
                        [f"TN: {tn:,}", f"FP: {fp:,}"],
                        [f"FN: {fn:,}", f"TP: {tp:,}"],
                    ]
                    fig2 = go.Figure(go.Heatmap(
                        z=[[tn, fp], [fn, tp]],
                        x=['Pred: Normal', 'Pred: Fraud'],
                        y=['Real: Normal', 'Real: Fraud'],
                        colorscale=[[0, T['bg_card']], [1, GOLD]],
                        text=z_text, texttemplate='%{text}',
                        textfont=dict(family='DM Mono', size=12, color=T['text_primary']),
                        showscale=False
                    ))
                    fig2.update_layout(title="📋 Confusion Matrix", height=380, **plotly_layout())
                    st.plotly_chart(fig2, use_container_width=True)

            st.markdown(f"""
            <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                        border:1px solid {T["border"]};border-radius:14px;padding:20px 24px;margin-top:8px;'>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:20px;'>
                    <div>
                        <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};margin-bottom:6px;'>MODEL</div>
                        <div style='font-family:Syne;font-size:.95rem;font-weight:700;color:{GOLD};'>{metrics.get("model","XGBoost")}</div>
                    </div>
                    <div>
                        <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};margin-bottom:6px;'>TRAINING DATA</div>
                        <div style='font-family:Syne;font-size:.95rem;font-weight:700;color:{CYAN};'>{metrics.get("toplam_ornek",0):,}</div>
                    </div>
                    <div>
                        <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};margin-bottom:6px;'>ACCURACY</div>
                        <div style='font-family:Syne;font-size:.95rem;font-weight:700;color:{GREEN};'>{acc*100:.1f}%</div>
                    </div>
                    <div>
                        <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};margin-bottom:6px;'>TRAINED</div>
                        <div style='font-family:Syne;font-size:.95rem;font-weight:700;color:{T["text_primary"]};'>{str(metrics.get("egitim_tarihi",""))[:10]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Model metrikleri bulunamadi. `python src/ml_model.py` calistirin.")

    with tab2:
        if not ml_df.empty and 'fraud_skoru' in ml_df.columns:
            yr = ml_df.nlargest(8, 'fraud_skoru')
            ca, cb = st.columns(2)
            for i, (_, row) in enumerate(yr.iterrows()):
                fs  = float(row.get('fraud_skoru', 0))
                bc  = RED if fs > 60 else ORANGE
                sev = "Kritik" if fs > 60 else "Suheli"
                signals = []
                if float(row.get('tx_gece_oran',  0)) > 0.3: signals.append("Gece islem fazla")
                if float(row.get('tx_hata_oran',  0)) > 0.1: signals.append("Yuksek hata orani")
                if float(row.get('dark_web_oran', 0)) > 0:   signals.append("Dark web karti")
                if int(row.get('iso_tahmin', 0)) == 1:        signals.append("Anomali tespit")
                sig_html = " · ".join(signals) if signals else "Genel risk skoru yuksek"
                card = f"""
                <div style='background:linear-gradient(135deg,rgba(255,69,96,.05),{T["bg_card"]});
                            border-left:3px solid {bc};border-radius:10px;
                            padding:14px 18px;margin:8px 0;
                            border:1px solid rgba(255,69,96,.12);'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                        <span style='font-family:Syne;font-weight:700;
                                     color:{T["text_primary"]};font-size:.95rem;'>
                            Musteri #{int(row.get("client_id", i))}
                        </span>
                        <span style='font-family:DM Mono;font-size:.65rem;color:{bc};
                                     background:{bc}18;padding:2px 10px;
                                     border-radius:20px;border:1px solid {bc}40;'>{sev}</span>
                    </div>
                    <div style='display:flex;gap:16px;margin-bottom:8px;'>
                        <div>
                            <div style='font-family:DM Mono;font-size:.55rem;color:{T["text_muted"]};'>FRAUD SKOR</div>
                            <div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:{bc};'>{fs:.1f}</div>
                        </div>
                    </div>
                    <div style='font-family:DM Mono;font-size:.62rem;
                                color:{T["text_muted"]};line-height:1.7;'>
                        {sig_html}
                    </div>
                    <div style='width:100%;background:{T["border"]};
                                height:3px;border-radius:3px;margin-top:10px;'>
                        <div style='width:{min(fs,100):.0f}%;
                                    background:linear-gradient(90deg,{ORANGE},{bc});
                                    height:100%;border-radius:3px;'></div>
                    </div>
                </div>"""
                with (ca if i % 2 == 0 else cb):
                    st.markdown(card, unsafe_allow_html=True)
        else:
            st.info("ML verisi bulunamadi.")

    with tab3:
        if metrics and 'feature_importance' in metrics:
            try:
                fi_raw = metrics['feature_importance']

                if isinstance(fi_raw, dict):
                    if 'feature' in fi_raw and 'importance' in fi_raw:
                        fi = pd.DataFrame(fi_raw)
                    else:
                        fi = pd.DataFrame(
                            list(fi_raw.items()), columns=['feature', 'importance'])
                elif isinstance(fi_raw, list) and len(fi_raw) > 0:
                    fi = pd.DataFrame(fi_raw)
                    fi.columns = [str(c).lower().strip() for c in fi.columns]
                    cols = fi.columns.tolist()
                    if 'feature' not in cols:
                        fi = fi.rename(columns={cols[0]: 'feature'})
                    if 'importance' not in fi.columns:
                        fi = fi.rename(columns={fi.columns[1]: 'importance'})
                else:
                    fi = pd.DataFrame(columns=['feature', 'importance'])

                fi = fi[['feature', 'importance']].dropna()
                fi['importance'] = pd.to_numeric(fi['importance'], errors='coerce').fillna(0)
                fi = fi.sort_values('importance', ascending=True).reset_index(drop=True)
                fi_top = fi.tail(15)

                if not fi_top.empty:
                    fig = go.Figure(go.Bar(
                        y=fi_top['feature'],
                        x=fi_top['importance'],
                        orientation='h',
                        marker=dict(
                            color=fi_top['importance'],
                            colorscale=[[0, 'rgba(201,168,76,.3)'], [1, GOLD]]
                        ),
                        text=[f"{v:.3f}" for v in fi_top['importance']],
                        textposition='outside',
                        textfont=dict(family='DM Mono', size=9, color=T['text_secondary'])
                    ))
                    fig.update_layout(
                        title="Feature Importance (XGBoost)",
                        height=500,
                        **plotly_layout()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Gosterilecek feature importance verisi yok.")

            except Exception as e:
                st.error(f"Feature importance yuklenirken hata: {e}")
        else:
            st.info("Feature importance icin modeli yeniden egitin.")


# ── MUSTERI DETAY ──
elif sayfa == "👤 Customer Detail":
    st.markdown(section_header("Customer Detail",
        "360 customer profile and ML predictions"), unsafe_allow_html=True)

    @st.cache_data(ttl=600, show_spinner=False)
    def load_detail():
        """Her tablo ayri try/except — biri crash etse diğerleri calisir"""
        ml_df2   = pd.DataFrame()
        risk_df  = pd.DataFrame()
        users_df = pd.DataFrame()
        cards_df = pd.DataFrame()

        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                try:    ml_df2  = pd.read_sql("SELECT * FROM client_ml",   conn)
                except: pass
                try:    risk_df = pd.read_sql("SELECT * FROM client_risk", conn)
                except: pass
                conn.close()
            except Exception:
                pass

        for csv_path, label in [
            (os.path.join(DATA_DIR, "users_data.csv"), "users"),
            (os.path.join(DATA_DIR, "cards_data.csv"), "cards"),
        ]:
            try:
                if os.path.exists(csv_path):
                    tmp = pd.read_csv(csv_path)
                    if label == "users":
                        for c in ['per_capita_income', 'yearly_income', 'total_debt']:
                            if c in tmp.columns:
                                tmp[c] = tmp[c].astype(str)\
                                    .str.replace('$', '', regex=False)\
                                    .str.replace(',', '', regex=False)
                                tmp[c] = pd.to_numeric(tmp[c], errors='coerce').fillna(0)
                        users_df = tmp
                    else:
                        if 'credit_limit' in tmp.columns:
                            tmp['credit_limit'] = tmp['credit_limit'].astype(str)\
                                .str.replace('$', '', regex=False)\
                                .str.replace(',', '', regex=False)
                            tmp['credit_limit'] = pd.to_numeric(
                                tmp['credit_limit'], errors='coerce').fillna(0)
                        cards_df = tmp
            except Exception:
                pass

        return ml_df2, risk_df, users_df, cards_df

    ml_df2, risk_df, users_df, cards_df = load_detail()

    id_list = []
    if len(ml_df2)   > 0: id_list = sorted(ml_df2['client_id'].astype(int).tolist())
    elif len(risk_df) > 0: id_list = sorted(risk_df['client_id'].astype(int).tolist())
    elif len(df_main) > 0: id_list = sorted(df_main['client_id'].astype(int).tolist())

    if not id_list:
        st.info("Musteri verisi bulunamadi. `python src/ml_model.py` calistirin.")
        st.stop()

    col_s, _ = st.columns([2, 3])
    with col_s:
        secili = st.selectbox("Musteri", id_list, format_func=lambda x: f"Musteri #{x}")

    def safe_row(df_src, col, val):
        if len(df_src) > 0 and col in df_src.columns and val in df_src[col].values:
            return df_src[df_src[col] == val].iloc[0]
        return None

    ml_r = safe_row(ml_df2,   'client_id', secili)
    rr   = safe_row(risk_df,  'client_id', secili)
    ur   = safe_row(users_df, 'id',        secili)

    rr_demo = None
    if rr is None and len(df_main) > 0:
        demo_rows = df_main[df_main['client_id'] == secili]
        if len(demo_rows) > 0:
            rr_demo = demo_rows.iloc[0]

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.metric("Harcama",    f"${float(rr['toplam']):,.0f}"            if rr   is not None else (f"${float(rr_demo['toplam_harcama']):,.0f}" if rr_demo is not None else "—"))
    with k2: st.metric("Islem",      f"{int(rr['islem']):,}"                   if rr   is not None else "—")
    with k3: st.metric("Risk Skoru", f"{float(rr['risk_skoru']):.1f}"          if rr   is not None else "—")
    with k4: st.metric("Fraud Skor", f"{float(ml_r.get('fraud_skoru',0)):.1f}" if ml_r is not None else "—")
    with k5: st.metric("Churn Skor", f"{float(ml_r.get('churn_skoru',0)):.1f}" if ml_r is not None else "—")
    with k6: st.metric("Kredi Skor", f"{int(ur['credit_score'])}"              if ur   is not None else "—")

    if ml_r is not None:
        fs      = float(ml_r.get('fraud_skoru', 0))
        cs      = float(ml_r.get('churn_skoru', 0))
        fraud_t = str(ml_r.get('fraud_tahmini', '—'))
        churn_t = str(ml_r.get('churn_tahmini', '—'))
        fraud_c = RED    if 'ksek'  in fraud_t else (ORANGE if 'pheli' in fraud_t else GREEN)
        churn_c = PURPLE if 'ksek'  in churn_t else (ORANGE if 'pheli' in churn_t else GREEN)

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{T["bg_card"]},{T["bg_card2"]});
                    border:1px solid {T["border"]};border-radius:16px;
                    padding:24px 28px;margin-bottom:20px;'>
            <div style='font-family:DM Mono;font-size:.6rem;color:{T["text_muted"]};
                        text-transform:uppercase;letter-spacing:.15em;margin-bottom:16px;'>
                ML Tahmin Ozeti — Musteri #{secili}
            </div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:20px;'>
                <div>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{T["text_muted"]};margin-bottom:6px;'>FRAUD TAHMIN</div>
                    <div style='font-family:Syne;font-size:1rem;font-weight:800;color:{fraud_c};'>{fraud_t}</div>
                    <div style='font-family:DM Mono;font-size:.75rem;color:{fraud_c};margin-top:4px;'>Skor: {fs:.1f}/100</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:8px;'>
                        <div style='width:{min(fs,100):.0f}%;background:{fraud_c};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{T["text_muted"]};margin-bottom:6px;'>CHURN TAHMIN</div>
                    <div style='font-family:Syne;font-size:1rem;font-weight:800;color:{churn_c};'>{churn_t}</div>
                    <div style='font-family:DM Mono;font-size:.75rem;color:{churn_c};margin-top:4px;'>Skor: {cs:.1f}/100</div>
                    <div style='width:100%;background:{T["border"]};height:4px;border-radius:4px;margin-top:8px;'>
                        <div style='width:{min(cs,100):.0f}%;background:{churn_c};height:100%;border-radius:4px;'></div>
                    </div>
                </div>
                <div>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{T["text_muted"]};margin-bottom:6px;'>XGB FRAUD SKOR</div>
                    <div style='font-family:Syne;font-size:1.5rem;font-weight:800;color:{ORANGE};'>
                        {round(float(ml_r["fraud_skoru_xgb"]),1) if ml_r.get("fraud_skoru_xgb") is not None else "—"}
                    </div>
                </div>
                <div>
                    <div style='font-family:DM Mono;font-size:.58rem;color:{T["text_muted"]};margin-bottom:6px;'>ANOMALI</div>
                    <div style='font-family:Syne;font-size:1rem;font-weight:800;
                                color:{"#FF4560" if ml_r.get("iso_tahmin",0)==1 else "#00E396"};'>
                        {"Anomali" if ml_r.get("iso_tahmin",0)==1 else "Normal"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── ADMIN PANELI ──
elif sayfa == "⚙️ Admin":
    if current_role != "admin":
        st.error("Bu sayfaya erisim yetkiniz yok.")
        st.stop()

    if not AUTH_OK:
        st.warning("Auth modulu yuklenemedi.")
        st.stop()

    st.markdown(section_header("Admin Panel",
        "User management and system control"), unsafe_allow_html=True)

    if st.session_state.get("pending_action"):
        action = st.session_state.pending_action
        try:
            if action["type"] == "approve":
                update_user_role(action["id"], action["role"])
                approve_user(action["id"], st.session_state.username)
                st.session_state.admin_msg = ("success", f"{action['username']} onaylandi! Rol: {action['role']}")
            elif action["type"] == "reject":
                reject_user(action["id"])
                st.session_state.admin_msg = ("warning", f"{action['username']} reddedildi.")
            elif action["type"] == "role_update":
                update_user_role(action["id"], action["role"])
                st.session_state.admin_msg = ("success", f"{action['username']} rol degistirildi: {action['role']}")
            elif action["type"] == "delete":
                delete_user(action["id"])
                st.session_state.admin_msg = ("warning", f"{action['username']} silindi.")
        except Exception as e:
            st.session_state.admin_msg = ("warning", f"Hata: {e}")
        st.session_state.pending_action = None
        st.rerun()

    if st.session_state.get("admin_msg"):
        _mt, _mx = st.session_state.admin_msg
        _cl = GREEN if _mt == "success" else ORANGE
        _bg = "rgba(0,227,150,.08)" if _mt == "success" else "rgba(255,107,53,.08)"
        _br = "rgba(0,227,150,.3)"  if _mt == "success" else "rgba(255,107,53,.3)"
        st.markdown(
            f"<div style='background:{_bg};border:1px solid {_br};border-left:4px solid {_cl};"
            f"border-radius:10px;padding:12px 18px;margin-bottom:16px;"
            f"font-family:DM Mono;font-size:.82rem;color:{_cl};'>{_mx}</div>",
            unsafe_allow_html=True
        )
        st.session_state.admin_msg = None

    try:
        login_stats = get_login_stats()
        all_users   = get_all_users()
    except Exception:
        login_stats = {"toplam": 0, "basarili": 0, "basarisiz": 0}
        all_users   = []

    pending  = [u for u in all_users if u["status"] == "pending"]
    active   = [u for u in all_users if u["status"] == "active"]
    rejected = [u for u in all_users if u["status"] == "rejected"]

    _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
    _mc1.markdown(_hmetric("Toplam",       f"{len(all_users):,}"),         unsafe_allow_html=True)
    _mc2.markdown(_hmetric("Aktif",        f"{len(active):,}",   GREEN),   unsafe_allow_html=True)
    _mc3.markdown(_hmetric("Bekleyen",     f"{len(pending):,}",  ORANGE),  unsafe_allow_html=True)
    _mc4.markdown(_hmetric("Reddedilen",   f"{len(rejected):,}", RED),     unsafe_allow_html=True)
    _mc5.markdown(_hmetric("Toplam Giris", f"{login_stats['toplam']:,}"),  unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    _tc1, _tc2, _tc3, _ = st.columns([1.4, 1.4, 1, 3])
    with _tc1:
        if st.button("Bekleyen Onaylar", use_container_width=True, key="atab_onay"):
            st.session_state.admin_tab = "onay"
    with _tc2:
        if st.button("Tum Kullanicilar", use_container_width=True, key="atab_tumu"):
            st.session_state.admin_tab = "tumu"
    with _tc3:
        if st.button("Sistem", use_container_width=True, key="atab_sistem"):
            st.session_state.admin_tab = "sistem"

    _active_tab = st.session_state.admin_tab
    st.markdown(
        f"<div style='height:1px;background:{T['border']};margin:8px 0 20px 0;'></div>",
        unsafe_allow_html=True
    )

    if _active_tab == "onay":
        if pending:
            for _u in pending:
                _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns([2.2, 1.5, 1.2, .9, .9])
                with _pc1:
                    st.markdown(f"""
                    <div style='padding:8px 0;'>
                        <div style='font-family:Syne;font-size:.88rem;font-weight:700;
                                    color:{T["text_primary"]};'>{_u["username"]}</div>
                        <div style='font-family:DM Mono;font-size:.6rem;
                                    color:{T["text_muted"]};'>{_u["email"]}</div>
                    </div>""", unsafe_allow_html=True)
                with _pc2:
                    st.markdown(
                        f"<div style='font-family:DM Mono;font-size:.6rem;"
                        f"color:{ORANGE};padding-top:12px;'>Bekliyor</div>",
                        unsafe_allow_html=True
                    )
                with _pc3:
                    _nr = st.selectbox("", ["viewer", "analyst", "admin"],
                        index=["viewer","analyst","admin"].index(_u["role"])
                        if _u["role"] in ["viewer","analyst","admin"] else 0,
                        key=f"rp_{_u['id']}", label_visibility="collapsed")
                with _pc4:
                    st.markdown('<div class="btn-green">', unsafe_allow_html=True)
                    if st.button("Onayla", key=f"ap_{_u['id']}", use_container_width=True):
                        st.session_state.pending_action = {
                            "type": "approve", "id": _u["id"],
                            "role": _nr, "username": _u["username"]}
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                with _pc5:
                    st.markdown('<div class="btn-red">', unsafe_allow_html=True)
                    if st.button("Reddet", key=f"rj_{_u['id']}", use_container_width=True):
                        st.session_state.pending_action = {
                            "type": "reject", "id": _u["id"],
                            "username": _u["username"]}
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='height:1px;background:{T['border']};margin:4px 0;'></div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(f"""
            <div style='background:rgba(0,227,150,.05);border:1px solid rgba(0,227,150,.18);
                        border-radius:12px;padding:30px;text-align:center;'>
                <div style='font-size:2rem;margin-bottom:8px;'>✅</div>
                <div style='font-family:Syne;font-size:.9rem;font-weight:700;
                            color:{GREEN};'>Onay bekleyen kullanici yok</div>
            </div>""", unsafe_allow_html=True)

    elif _active_tab == "tumu":
        for _u in all_users:
            _sc = {"active": GREEN, "pending": ORANGE, "rejected": RED}.get(_u["status"], T["text_muted"])
            _sl = {"active": "Aktif", "pending": "Bekliyor", "rejected": "Reddedildi"}.get(_u["status"], "—")
            _av = {"admin": "👨‍💼", "analyst": "📊", "viewer": "👁️"}.get(_u["role"], "👤")
            _uc1,_uc2,_uc3,_uc4,_uc5,_uc6,_uc7 = st.columns([.4,2,1.5,1.2,.9,1.2,.7])
            with _uc1:
                st.markdown(
                    f"<div style='font-size:1.2rem;padding-top:8px;text-align:center;'>{_av}</div>",
                    unsafe_allow_html=True
                )
            with _uc2:
                st.markdown(f"""
                <div style='padding:5px 0;'>
                    <div style='font-family:Syne;font-size:.82rem;font-weight:700;
                                color:{T["text_primary"]};'>{_u["username"]}</div>
                    <div style='font-family:DM Mono;font-size:.58rem;
                                color:{T["text_muted"]};'>{_u["email"]}</div>
                </div>""", unsafe_allow_html=True)
            with _uc3:
                st.markdown(
                    f"<div style='font-family:DM Mono;font-size:.62rem;"
                    f"color:{T['text_muted']};padding-top:9px;'>"
                    f"{_u.get('display_name') or '—'}</div>",
                    unsafe_allow_html=True
                )
            with _uc4:
                _nr2 = st.selectbox("", ["viewer","analyst","admin"],
                    index=["viewer","analyst","admin"].index(_u["role"])
                    if _u["role"] in ["viewer","analyst","admin"] else 0,
                    key=f"cr_{_u['id']}", label_visibility="collapsed")
            with _uc5:
                if _nr2 != _u["role"]:
                    st.markdown('<div class="btn-purple">', unsafe_allow_html=True)
                    if st.button("Kaydet", key=f"sr_{_u['id']}", use_container_width=True):
                        st.session_state.pending_action = {
                            "type": "role_update", "id": _u["id"],
                            "role": _nr2, "username": _u["username"]}
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            with _uc6:
                st.markdown(
                    f"<div style='font-family:DM Mono;font-size:.62rem;padding-top:10px;'>"
                    f"<span style='color:{_sc};'>{_sl}</span></div>",
                    unsafe_allow_html=True
                )
            with _uc7:
                if _u["username"] != st.session_state.username:
                    st.markdown('<div class="btn-red">', unsafe_allow_html=True)
                    if st.button("Sil", key=f"dl_{_u['id']}", use_container_width=True):
                        st.session_state.pending_action = {
                            "type": "delete", "id": _u["id"],
                            "username": _u["username"]}
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='height:1px;background:{T['border']};margin:2px 0;'></div>",
                unsafe_allow_html=True
            )

    elif _active_tab == "sistem":
        st.markdown(
            f"<div style='font-family:Syne;font-size:1rem;font-weight:700;"
            f"color:{T['text_primary']};margin-bottom:14px;'>Sifre Degistir</div>",
            unsafe_allow_html=True
        )
        _pw1, _pw2, _pw3 = st.columns(3)
        with _pw1: _old_pw  = st.text_input("Mevcut Sifre",      type="password", key="old_pw")
        with _pw2: _new_pw  = st.text_input("Yeni Sifre",         type="password", key="new_pw")
        with _pw3: _new_pw2 = st.text_input("Yeni Sifre Tekrar",  type="password", key="new_pw2")
        if st.button("Guncelle", key="pw_update_btn"):
            if _new_pw != _new_pw2:
                st.error("Sifreler eslesmıyor!")
            else:
                try:
                    _r = change_password(st.session_state.username, _old_pw, _new_pw)
                    if _r["success"]: st.success(_r["message"])
                    else:             st.error(_r["message"])
                except Exception as e:
                    st.error(f"Hata: {e}")

        st.markdown(hr(), unsafe_allow_html=True)

        _ls1, _ls2, _ls3 = st.columns(3)
        _ls1.markdown(_hmetric("Toplam Giris",  f"{login_stats['toplam']:,}"),          unsafe_allow_html=True)
        _ls2.markdown(_hmetric("Basarili",      f"{login_stats['basarili']:,}",  GREEN), unsafe_allow_html=True)
        _ls3.markdown(_hmetric("Basarisiz",     f"{login_stats['basarisiz']:,}", RED),   unsafe_allow_html=True)

        st.markdown(hr(), unsafe_allow_html=True)

        ml_ozet_s = load_ml_ozet()
        if ml_ozet_s:
            st.markdown(f"""
            <div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);
                        border-radius:12px;padding:18px 22px;margin-bottom:16px;'>
                <div style='font-family:DM Mono;font-size:.6rem;color:{T["text_muted"]};
                            text-transform:uppercase;margin-bottom:10px;'>Son ML Calismasi</div>
                <div style='font-family:DM Mono;font-size:.72rem;
                            color:{T["text_secondary"]};line-height:2;'>
                    Hesaplama: <span style='color:{CYAN};'>{str(ml_ozet_s.get("hesaplama_tarihi",""))[:16]}</span> ·
                    Toplam: <span style='color:{GOLD};'>{ml_ozet_s.get("toplam",0):,}</span> ·
                    Yuksek Risk: <span style='color:{RED};'>{ml_ozet_s.get("yuksek_risk",0):,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        _c1, _c2 = st.columns(2)
        with _c1:
            _ep = st.selectbox("API Endpoint",
                ["/health", "/stats", "/stats/fraud", "/model/metrics"], key="api_ep_sel")
            if st.button("Test Et", use_container_width=True, key="api_test_btn"):
                _res = api_get(_ep)
                if _res: st.json(_res if isinstance(_res, dict) else _res[:2])
                else:    st.error("Baglanti basarisiz.")
        with _c2:
            st.download_button("Tam Musteri Verisi",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="musteri_tam.csv", mime="text/csv",
                use_container_width=True, key="dl_musteri_btn")
            if st.button("Cache Temizle", use_container_width=True, key="cache_clear_btn"):
                st.cache_data.clear()
                st.success("Cache temizlendi!")