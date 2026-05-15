"""
FinSight v6.1 — Profesyonel & Hatasız
Düzeltmeler:
  1. call_claude_for_customer — anthropic-version + x-api-key header eklendi
  2. filter_data — hashable tuple cache key düzeltildi
  3. api_get_raw — timeout + exception handling güçlendirildi
  4. Admin tab — radio/button state karmaşası giderildi
  5. load_detail_data — cards_df join koşulu düzeltildi
  6. CSS btn-* wrapper — Streamlit render sorunu giderildi
  7. _hmetric — tüm kullanım noktalarında unsafe_allow_html=True eklendi
  8. Tüm f-string içi çift tırnak → kaçış / değişken ile çözüldü
  9. Bölüm başlıklarına tutarlı font/renk uygulandı
 10. st.stop() çağrıları doğru yerlerde bırakıldı
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
import random
from datetime import datetime

# ── Dinamik base path ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "financeai.db")
METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from auth import (
        login_user, register_user, get_all_users, get_pending_users,
        approve_user, reject_user, update_user_role, delete_user,
        get_login_stats, change_password, init_db, admin_exists,
    )
    init_db()
    AUTH_OK = True
except Exception:
    AUTH_OK = False

# ── Sayfa Konfigürasyonu ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "logged_in":      False,
    "username":       "",
    "role":           "",
    "display_name":   "",
    "avatar":         "👤",
    "user_id":        None,
    "dark_mode":      True,
    "auth_tab":       "login",
    "pending_action": None,
    "admin_msg":      None,
    "admin_tab":      "onay",
    "api_token":      None,
    "detail_tab":     "profile",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# AUTH yoksa geliştirici modu
if not AUTH_OK and not st.session_state.logged_in:
    st.session_state.update({
        "logged_in": True, "username": "admin",
        "role": "admin", "display_name": "Admin", "avatar": "👤",
    })

# ═══════════════════════════════════════════════════════════════════════════════
# TEMA SİSTEMİ
# ═══════════════════════════════════════════════════════════════════════════════
def get_theme() -> dict:
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
            "metric_gold":    "#C9A84C",
        }
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
        "metric_gold":    "#8B6820",
    }


T = get_theme()

# Renk sabitleri
GOLD   = "#C9A84C"
CYAN   = "#00D4FF"
RED    = "#FF4560"
GREEN  = "#00E396"
ORANGE = "#FF6B35"
PURPLE = "#8B5CF6"

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
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
    font-weight:800 !important; color:{T['metric_gold']} !important;
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
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stTextArea>div>div>textarea {{
    background:var(--input-bg) !important; border:1px solid var(--border) !important;
    color:var(--text-primary) !important; border-radius:8px !important;
    transition:border-color .2s,box-shadow .2s !important;
}}
.stTextInput>div>div>input:focus {{
    border-color:rgba(201,168,76,.5) !important;
    box-shadow:0 0 0 3px rgba(201,168,76,.08) !important;
}}
.stSelectbox>div>div, .stMultiSelect>div>div {{
    background:var(--input-bg) !important; border:1px solid var(--border) !important;
    border-radius:8px !important;
}}
.stSlider>div>div>div>div {{ background:{GOLD} !important; }}
.stDataFrame {{ border:1px solid var(--border) !important; border-radius:12px !important; }}
.streamlit-expanderHeader {{
    background:var(--bg-card) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; color:var(--text-primary) !important;
}}
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:var(--bg-base); }}
::-webkit-scrollbar-thumb {{ background:rgba(201,168,76,.2); border-radius:10px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(201,168,76,.4); }}
hr {{ border-color:var(--border) !important; margin:1.5rem 0 !important; }}
@keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
.fade-in {{ animation:fadeIn .3s ease forwards; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def section_header(title: str, subtitle: str = "", g1=None, g2=None) -> str:
    c1    = g1 or T["text_primary"]
    c2    = g2 or GOLD
    parts = title.split()
    first = parts[0] if parts else ""
    rest  = " ".join(parts[1:]) if len(parts) > 1 else ""
    sub   = (
        f"<div style='font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};"
        f"letter-spacing:.12em;margin-top:8px;text-transform:uppercase;'>{subtitle}</div>"
        if subtitle else ""
    )
    return (
        f"<div style='margin-bottom:28px;'>"
        f"<div style='font-family:Syne;font-size:1.85rem;font-weight:800;margin:0;"
        f"letter-spacing:-.02em;line-height:1.15;'>"
        f"<span style='color:{c1};'>{first} </span>"
        f"<span style='color:{c2};'>{rest}</span>"
        f"</div>{sub}</div>"
    )


def hr_line() -> str:
    return (
        "<div style='height:1px;"
        "background:linear-gradient(90deg,transparent,rgba(201,168,76,.15),transparent);"
        "margin:20px 0;'></div>"
    )


def hmetric(label: str, value: str, color: str = "") -> str:
    c = color or GOLD
    bg   = T["bg_card"]
    bg2  = T["bg_card2"]
    bdr  = T["border"]
    muted= T["text_muted"]
    return (
        f"<div style='background:linear-gradient(135deg,{bg},{bg2});"
        f"border:1px solid {bdr};border-radius:14px;padding:18px 20px;'>"
        f"<div style='font-family:DM Mono;font-size:.58rem;color:{muted};"
        f"text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;'>{label}</div>"
        f"<div style='font-family:Syne;font-size:1.8rem;font-weight:800;color:{c};'>{value}</div>"
        f"</div>"
    )


def score_bar(score: float, color: str, height: int = 6) -> str:
    pct = min(max(score, 0), 100)
    bdr = T["border"]
    return (
        f"<div style='width:100%;background:{bdr};height:{height}px;"
        f"border-radius:{height}px;margin-top:8px;overflow:hidden;'>"
        f"<div style='width:{pct:.0f}%;height:100%;"
        f"background:linear-gradient(90deg,{color}99,{color});border-radius:{height}px;"
        f"transition:width .6s ease;'></div></div>"
    )


def status_badge(status: str) -> str:
    cfg = {
        "Normal":     (GREEN,  "✓ Normal"),
        "Suspicious": (ORANGE, "⚠ Şüpheli"),
        "Fraud Risk": (RED,    "🚨 Fraud Risk"),
    }
    color, label = cfg.get(status, (T["text_muted"], status))
    return (
        f"<span style='font-family:DM Mono;font-size:.6rem;color:{color};"
        f"background:{color}18;padding:3px 10px;border-radius:20px;"
        f"border:1px solid {color}40;white-space:nowrap;'>{label}</span>"
    )


def plotly_layout(**kw) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(family="DM Sans", color=T["text_secondary"], size=11),
        title_font=dict(family="Syne", size=14, color=T["text_primary"]),
        colorway=[GOLD, CYAN, GREEN, RED, ORANGE, PURPLE],
        xaxis=dict(
            gridcolor="rgba(90,106,122,.12)", linecolor="rgba(90,106,122,.15)",
            tickfont=dict(family="DM Mono", size=9, color=T["text_muted"]), zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(90,106,122,.12)", linecolor="rgba(90,106,122,.15)",
            tickfont=dict(family="DM Mono", size=9, color=T["text_muted"]), zeroline=False,
        ),
        legend=dict(
            bgcolor=T["bg_card"], bordercolor=T["border"], borderwidth=1,
            font=dict(family="DM Mono", size=9, color=T["text_secondary"]),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor=T["bg_card"], bordercolor="rgba(201,168,76,.3)",
            font=dict(family="DM Mono", size=11, color=T["text_primary"]),
        ),
    )
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# API KATMANI
# ═══════════════════════════════════════════════════════════════════════════════
API_URL = "http://localhost:8000"


@st.cache_data(ttl=15, show_spinner=False)
def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


API_ALIVE = check_api()
api_status = "🟢 API Aktif" if API_ALIVE else "🔴 API Offline"


def _auth_headers() -> dict:
    token = st.session_state.get("api_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(endpoint: str, params: dict = None, timeout: int = 6):
    if not API_ALIVE:
        return None
    try:
        r = requests.get(
            f"{API_URL}{endpoint}", params=params,
            headers=_auth_headers(), timeout=timeout,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(endpoint: str, payload: dict, timeout: int = 8):
    if not API_ALIVE:
        return None
    try:
        r = requests.post(
            f"{API_URL}{endpoint}", json=payload,
            headers=_auth_headers(), timeout=timeout,
        )
        return r.json() if r.status_code in (200, 201) else None
    except Exception:
        return None


def api_get_raw(endpoint: str, params: dict = None, timeout: int = 6) -> tuple:
    """(status_code, response_dict) döner. Hata durumunda (0, {error: ...})"""
    try:
        r = requests.get(
            f"{API_URL}{endpoint}", params=params,
            headers=_auth_headers(), timeout=timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {}
        return r.status_code, body
    except requests.exceptions.ConnectionError:
        return 0, {"error": "Bağlantı reddedildi — API çalışıyor mu?"}
    except requests.exceptions.Timeout:
        return 0, {"error": f"Zaman aşımı ({timeout}s)"}
    except Exception as exc:
        return 0, {"error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_ml_data() -> pd.DataFrame:
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
def load_ml_ozet() -> dict:
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
def load_model_metrics() -> dict:
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE AI — MÜŞTERİ RİSK YORUMU
# ═══════════════════════════════════════════════════════════════════════════════

def call_claude_for_customer(client_data: dict) -> str:
    """
    Müşteri verisini Claude API'ye gönderir, risk yorumu alır.
    API erişilemezse kural tabanlı yorum döner.
    DÜZELTİLDİ: anthropic-version + x-api-key header eklendi.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        try:
            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Sen bir finansal risk analisti yapay zekasısın. "
                            "Aşağıdaki müşteri verilerini analiz et ve Türkçe, "
                            "kısa ve net bir risk yorumu yaz. "
                            "Maksimum 3 cümle, madde işareti kullanma.\n\n"
                            f"Müşteri #{client_data.get('client_id', '?')} Verileri:\n"
                            f"- Fraud Skoru: {client_data.get('fraud_skoru', 0):.1f}/100\n"
                            f"- Churn Skoru: {client_data.get('churn_skoru', 0):.1f}/100\n"
                            f"- Anomali Skoru: {client_data.get('anomali_skoru', 0):.2f}\n"
                            f"- Dark Web Oranı: {float(client_data.get('dark_web_oran', 0)) * 100:.1f}%\n"
                            f"- Gece İşlem Oranı: {float(client_data.get('tx_gece_oran', 0)) * 100:.1f}%\n"
                            f"- İşlem Hata Oranı: {float(client_data.get('tx_hata_oran', 0)) * 100:.1f}%\n"
                            f"- Fraud Tahmini: {client_data.get('fraud_tahmini', '—')}\n"
                            f"- İşlem Sayısı: {client_data.get('tx_islem_sayisi', 0)}\n\n"
                            "Kısa, profesyonel bir risk yorumu yaz:"
                        ),
                    }
                ],
            }
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["content"][0]["text"].strip()
        except Exception:
            pass

    # ── Kural tabanlı fallback ──────────────────────────────────────────────
    fs = float(client_data.get("fraud_skoru", 0))
    dw = float(client_data.get("dark_web_oran", 0))
    gn = float(client_data.get("tx_gece_oran", 0))

    if fs >= 60:
        yorum = f"Bu müşteri %{fs:.0f} fraud skoruyla yüksek risk kategorisindedir."
        if dw > 0.1:
            yorum += " Dark web işlem geçmişi ciddi bir güvenlik riski oluşturmaktadır."
        yorum += " Hesabın acilen incelenmesi ve gerekirse geçici kısıtlama uygulanması önerilir."
    elif fs >= 30:
        yorum = f"Müşteri %{fs:.0f} fraud skoru ile orta düzey risk taşımaktadır."
        if gn > 0.3:
            yorum += f" Gece işlem oranının yüksekliği (%{gn * 100:.0f}) dikkat çekmektedir."
        yorum += " Ek doğrulama ve yakın takip önerilir."
    else:
        yorum = (
            f"Müşteri %{fs:.0f} fraud skoru ile düşük risk profiline sahiptir. "
            "İşlem davranışları beklenen aralıkta seyrediyor. Standart izleme yeterlidir."
        )
    return yorum


# ═══════════════════════════════════════════════════════════════════════════════
# GİRİŞ SAYFASI
# ═══════════════════════════════════════════════════════════════════════════════

def show_auth_page():
    if AUTH_OK and not admin_exists():
        st.markdown(
            f"<div style='max-width:520px;margin:60px auto 0;background:rgba(255,107,53,.08);"
            f"border:1px solid rgba(255,107,53,.35);border-left:4px solid {ORANGE};"
            f"border-radius:14px;padding:24px 28px;'>"
            f"<div style='font-family:Syne;font-size:1.1rem;font-weight:800;color:{ORANGE};margin-bottom:10px;'>"
            f"Kurulum Gerekli</div>"
            f"<div style='font-family:DM Mono;font-size:.72rem;color:{T['text_secondary']};line-height:2;'>"
            f"Henüz yönetici hesabı oluşturulmadı.<br>Terminalde çalıştırın:</div>"
            f"<div style='background:rgba(0,0,0,.3);border-radius:8px;padding:12px 16px;"
            f"margin-top:12px;font-family:DM Mono;font-size:.8rem;color:{CYAN};'>python setup.py</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown(
            f"<div style='text-align:center;padding:36px 0 24px;'>"
            f"<div style='font-size:2.2rem;margin-bottom:8px;'>💎</div>"
            f"<div style='font-family:Syne;font-size:2rem;font-weight:800;"
            f"background:linear-gradient(135deg,{GOLD},#FFE4A0);"
            f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>FinSight</div>"
            f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
            f"letter-spacing:.22em;text-transform:uppercase;margin-top:4px;'>Analiz Sistemi</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

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
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="kullaniciadi")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                ca, cb = st.columns(2)
                with ca:
                    submit = st.form_submit_button("Sign In", use_container_width=True)
                with cb:
                    tema_btn = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)
                if tema_btn:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()
                if submit:
                    if not username or not password:
                        st.error("Kullanıcı adı ve şifre girin.")
                    elif AUTH_OK:
                        result = login_user(username, password)
                        if result["success"]:
                            u = result["user"]
                            st.session_state.update({
                                "logged_in": True,
                                "username": u["username"],
                                "role": u["role"],
                                "display_name": u.get("display_name") or u["username"],
                                "avatar": u.get("avatar") or "👤",
                                "user_id": u["id"],
                            })
                            try:
                                tr = requests.post(
                                    f"{API_URL}/auth/login",
                                    json={"username": username, "password": password},
                                    timeout=4,
                                )
                                if tr.status_code == 200:
                                    st.session_state.api_token = tr.json().get("token")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(result["message"])
                    else:
                        st.error("Auth sistemi aktif değil.")
        else:
            with st.form("register_form", clear_on_submit=True):
                r_display  = st.text_input("Ad Soyad",           placeholder="Ahmet Yilmaz")
                r_username = st.text_input("Username",            placeholder="lowercase")
                r_email    = st.text_input("E-posta",             placeholder="ornek@email.com")
                rc1, rc2   = st.columns(2)
                with rc1:
                    r_pw1 = st.text_input("Password", type="password")
                with rc2:
                    r_pw2 = st.text_input("Confirm Password", type="password")
                r_role = st.selectbox(
                    "Rol", ["viewer", "analyst"],
                    format_func=lambda x: {"viewer": "Viewer", "analyst": "Analist"}[x],
                )
                ra, rb = st.columns(2)
                with ra:
                    r_submit = st.form_submit_button("Register", use_container_width=True)
                with rb:
                    r_tema = st.form_submit_button(f"{T['toggle_icon']} Tema", use_container_width=True)
                if r_tema:
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()
                if r_submit:
                    if not all([r_display, r_username, r_email, r_pw1, r_pw2]):
                        st.error("Tüm alanları doldurun.")
                    elif " " in r_username:
                        st.error("Username boşluk içeremez.")
                    elif r_pw1 != r_pw2:
                        st.error("Şifreler eşleşmiyor.")
                    elif AUTH_OK:
                        result = register_user(r_username, r_email, r_pw1, r_display, r_role)
                        if result["success"]:
                            st.success("Kayıt başarılı! Admin onayı bekleniyor.")
                        else:
                            st.error(result["message"])
                    else:
                        st.error("Auth sistemi aktif değil.")


if not st.session_state.logged_in:
    show_auth_page()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ROL / SAYFA TANIMLAMALARI
# ═══════════════════════════════════════════════════════════════════════════════
ROLE_PAGES = {
    "admin": [
        "📊 Overview", "📈 Monthly Volume", "💎 Segments",
        "📂 Category Averages", "🎯 Spending × Risk",
        "📈 Trend Analysis", "🔍 Customer Analysis",
        "⚠️ Risk & Fraud", "🗺️ Geographic Analysis", "🤖 AI Insights",
        "👤 Customer Detail", "🧠 New Prediction", "⚙️ Admin",
    ],
    "analyst": [
        "📊 Overview", "📈 Monthly Volume", "💎 Segments",
        "📂 Category Averages", "🎯 Spending × Risk",
        "📈 Trend Analysis", "🔍 Customer Analysis",
        "⚠️ Risk & Fraud", "🗺️ Geographic Analysis", "🤖 AI Insights",
        "👤 Customer Detail", "🧠 New Prediction",
    ],
    "viewer": [
        "📊 Overview", "📈 Monthly Volume", "💎 Segments",
        "📂 Category Averages", "🎯 Spending × Risk",
        "📈 Trend Analysis", "🗺️ Geographic Analysis",
    ],
}

current_role  = st.session_state.role or "admin"
allowed_pages = ROLE_PAGES.get(current_role, ROLE_PAGES["viewer"])


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / GERÇEK VERİ
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def generate_demo_data(n: int = 1219) -> pd.DataFrame:
    np.random.seed(42)
    cats  = ["Market","Restaurant","Fuel","Online Shopping","Health",
             "Entertainment","Transport","Education","Clothing","Electronics"]
    cities = ["Istanbul","Ankara","Izmir","Bursa","Antalya",
              "Adana","Konya","Gaziantep","Sanliurfa","Mersin"]
    df = pd.DataFrame({
        "client_id":      range(1, n + 1),
        "sehir":          np.random.choice(cities, n),
        "yas":            np.random.randint(18, 75, n),
        "toplam_harcama": np.random.lognormal(10, 1.2, n),
        "islem_sayisi":   np.random.randint(5, 250, n),
        "risk_skoru":     np.random.exponential(8, n).clip(0, 65),
        "kategori":       np.random.choice(cats, n),
        "aktif_ay":       np.random.randint(1, 36, n),
    })
    df["risk_seviyesi"] = pd.cut(
        df["risk_skoru"],
        bins=[-np.inf, 20, 40, np.inf],
        labels=["Dusuk Risk", "Fair Risk", "Yuksek Risk"],
    ).astype(str)
    df["fraud_tahmini"] = np.where(
        np.random.random(n) < (df["risk_skoru"] / 65) * 0.1, "Suheli", "Normal"
    )
    df["segment"] = pd.cut(
        df["toplam_harcama"],
        bins=[0, 5000, 20000, 50000, np.inf],
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    ).astype(str)
    return df


def adapt_real_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    np.random.seed(42)
    n = len(df)
    cities = ["Istanbul","Ankara","Izmir","Bursa","Antalya",
              "Adana","Konya","Gaziantep","Sanliurfa","Mersin"]
    cats = ["Market","Restaurant","Fuel","Online Shopping","Health",
            "Entertainment","Transport","Education","Clothing","Electronics"]

    for src_col, dst_col, fallback in [
        ("toplam", "toplam_harcama", lambda: np.random.lognormal(10, 1.2, n)),
        ("islem",  "islem_sayisi",   lambda: np.random.randint(5, 250, n)),
    ]:
        if src_col in df.columns:
            df[dst_col] = pd.to_numeric(df[src_col], errors="coerce").fillna(0)
        elif dst_col not in df.columns:
            df[dst_col] = fallback()

    for col, fallback in [
        ("sehir",    lambda: np.random.choice(cities, n)),
        ("yas",      lambda: np.random.randint(18, 75, n)),
        ("aktif_ay", lambda: np.random.randint(1, 36, n)),
        ("kategori", lambda: np.random.choice(cats, n)),
    ]:
        if col not in df.columns:
            df[col] = fallback()

    if "risk_seviyesi" in df.columns:
        mapping = {
            "Yuksek Risk": "Yuksek Risk", "Yüksek Risk": "Yuksek Risk",
            "Dusuk Risk":  "Dusuk Risk",  "Düşük Risk":  "Dusuk Risk",
            "Fair Risk":   "Fair Risk",   "Orta Risk":   "Fair Risk",
        }
        df["risk_seviyesi"] = (
            df["risk_seviyesi"]
            .astype(str)
            .str.replace(r"[🟢🟡🔴⚠️]", "", regex=True)
            .str.strip()
            .map(lambda x: mapping.get(x, x))
        )
    else:
        df["risk_seviyesi"] = "Fair Risk"

    if "risk_skoru" not in df.columns:
        df["risk_skoru"] = np.random.exponential(8, n).clip(0, 65)

    df["fraud_tahmini"] = np.where(df["risk_skoru"] > 30, "Suheli", "Normal")
    df["segment"] = pd.cut(
        df["toplam_harcama"],
        bins=[0, 100000, 400000, 700000, np.inf],
        labels=["Bronze", "Silver", "Gold", "Platinum"],
    ).astype(str).fillna("Bronze")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_main_data():
    try:
        from database import get_all_clients
        return adapt_real_data(get_all_clients()), "🟢 Live Database"
    except Exception:
        return generate_demo_data(), "🟡 Demo Mode"


@st.cache_data(ttl=3600, show_spinner=False)
def filter_data(
    risk_filtre_tuple: tuple,
    segment_filtre_tuple: tuple,
    risk_min: float,
    risk_max: float,
) -> pd.DataFrame:
    """
    DÜZELTİLDİ: Argümanlar hashable tuple olarak alınıyor.
    Çağırırken: filter_data(tuple(sorted(risk)), tuple(sorted(seg)), rmin, rmax)
    """
    df_all, _ = load_main_data()
    mask = (
        df_all["risk_seviyesi"].astype(str).isin(list(risk_filtre_tuple))
        & df_all["segment"].astype(str).isin(list(segment_filtre_tuple))
        & (df_all["risk_skoru"] >= risk_min)
        & (df_all["risk_skoru"] <= risk_max)
    )
    return df_all[mask].copy()


df_main, data_source = load_main_data()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center;padding:16px 0 14px;'>"
        f"<div style='font-size:1.2rem;margin-bottom:4px;'>💎</div>"
        f"<div style='font-family:Syne;font-size:1.2rem;font-weight:800;"
        f"background:linear-gradient(135deg,{GOLD},#FFE4A0);"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>FinSight</div>"
        f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};"
        f"letter-spacing:.2em;text-transform:uppercase;margin-top:2px;'>"
        f"v6.1 — {current_role.upper()}</div></div>",
        unsafe_allow_html=True,
    )

    role_color = {"admin": RED, "analyst": GOLD, "viewer": GREEN}.get(current_role, GOLD)
    pending_count = 0
    if current_role == "admin" and AUTH_OK:
        try:
            pending_count = len(get_pending_users())
        except Exception:
            pass

    p_badge = (
        f' <span style="background:{RED};color:#fff;border-radius:10px;'
        f'padding:1px 6px;font-size:.55rem;">{pending_count}</span>'
        if pending_count > 0 else ""
    )

    ml_ozet_sb = load_ml_ozet()
    ml_ok      = bool(ml_ozet_sb)

    st.markdown(
        f"<div style='background:linear-gradient(135deg,rgba(201,168,76,.08),rgba(201,168,76,.03));"
        f"border:1px solid rgba(201,168,76,.18);border-radius:12px;"
        f"padding:10px 13px;margin:0 4px 10px 4px;'>"
        f"<div style='display:flex;align-items:center;gap:9px;'>"
        f"<div style='font-size:1.2rem;'>{st.session_state.avatar}</div>"
        f"<div><div style='font-family:Syne;font-size:.82rem;font-weight:700;"
        f"color:{T['text_primary']};'>{st.session_state.display_name}</div>"
        f"<div style='font-family:DM Mono;font-size:.55rem;color:{role_color};"
        f"text-transform:uppercase;letter-spacing:.12em;'>{current_role}{p_badge}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    api_dot  = "#00E396" if API_ALIVE else "#FF4560"
    ml_dot   = "#00E396" if ml_ok else "#FF6B35"
    ml_label = "🤖 ML Model Ready" if ml_ok else "⚠️ ML Model Missing"

    st.markdown(
        f"<div style='padding:0 4px 8px;'>"
        f"<div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};"
        f"display:flex;align-items:center;gap:6px;margin-bottom:3px;'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{GREEN};"
        f"display:inline-block;box-shadow:0 0 5px {GREEN};'></span>"
        f"{data_source}</div>"
        f"<div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};"
        f"display:flex;align-items:center;gap:6px;margin-bottom:3px;'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{api_dot};"
        f"display:inline-block;'></span>{api_status}</div>"
        f"<div style='font-family:DM Mono;font-size:.57rem;color:{T['text_muted']};"
        f"display:flex;align-items:center;gap:6px;'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{ml_dot};"
        f"display:inline-block;'></span>{ml_label}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(hr_line(), unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};"
        f"text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Navigasyon</div>",
        unsafe_allow_html=True,
    )
    sayfa = st.radio("", allowed_pages, label_visibility="collapsed")
    st.markdown(hr_line(), unsafe_allow_html=True)

    RISK_OPTIONS    = ["Dusuk Risk", "Fair Risk", "Yuksek Risk"]
    SEGMENT_OPTIONS = ["Bronze", "Silver", "Gold", "Platinum"]

    if current_role in ("admin", "analyst"):
        st.markdown(
            f"<div style='font-family:DM Mono;font-size:.5rem;color:{T['text_muted']};"
            f"text-transform:uppercase;letter-spacing:.2em;margin-bottom:6px;padding:0 4px;'>Filtreler</div>",
            unsafe_allow_html=True,
        )
        risk_filtre    = st.multiselect("Risk",    RISK_OPTIONS,    default=RISK_OPTIONS,    label_visibility="collapsed")
        segment_filtre = st.multiselect("Segment", SEGMENT_OPTIONS, default=SEGMENT_OPTIONS, label_visibility="collapsed")
        risk_range     = st.slider("Risk Aralığı", 0, 65, (0, 65), label_visibility="collapsed")
    else:
        risk_filtre    = RISK_OPTIONS
        segment_filtre = SEGMENT_OPTIONS
        risk_range     = (0, 65)

    df = filter_data(
        tuple(sorted(risk_filtre)),
        tuple(sorted(segment_filtre)),
        risk_range[0],
        risk_range[1],
    )
    if len(df) == 0:
        df = df_main.copy()

    st.markdown(
        f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
        f"margin-top:6px;padding:6px 10px;background:rgba(201,168,76,.06);"
        f"border-radius:8px;border:1px solid rgba(201,168,76,.1);'>"
        f"<span style='color:{GOLD};font-weight:600;'>{len(df):,}</span> müşteri seçildi</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    tc, cc = st.columns(2)
    with tc:
        if st.button(f"{T['toggle_icon']} Tema", use_container_width=True, key="tema_sb"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with cc:
        if st.button("🚪 Logout", use_container_width=True, key="logout_sb"):
            for k in ("logged_in", "username", "role", "display_name", "avatar", "user_id", "api_token"):
                st.session_state[k] = False if k == "logged_in" else ""
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ───────────────────────────── SAYFALAR ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── OVERVIEW ───────────────────────────────────────────────────────────────────
if sayfa == "📊 Overview":
    st.markdown(
        section_header("Financial Analysis Dashboard",
                        f"Real-time — {datetime.now().strftime('%d %B %Y, %H:%M')}"),
        unsafe_allow_html=True,
    )

    ml_ozet   = load_ml_ozet()
    api_stats = api_get("/stats")

    if ml_ozet:
        toplam     = int(ml_ozet.get("toplam", len(df)))
        supheli    = int(ml_ozet.get("supheli", 0))
        yuksek     = int(ml_ozet.get("yuksek_risk", 0))
        churn      = int(ml_ozet.get("churn_yuksek", 0))
        fraud_oran = round(supheli / max(toplam, 1) * 100, 1)
        hacim      = float(df_main["toplam_harcama"].sum()) if "toplam_harcama" in df_main.columns else 0
    elif api_stats:
        toplam     = int(api_stats.get("toplam", len(df)))
        yuksek     = int(api_stats.get("yuksek_risk", 0))
        supheli    = int(api_stats.get("supheli", 0))
        fraud_oran = round(supheli / max(toplam, 1) * 100, 1)
        hacim      = float(df_main["toplam_harcama"].sum()) if "toplam_harcama" in df_main.columns else 0
        churn      = 0
    else:
        toplam     = len(df)
        yuksek     = int(df["risk_seviyesi"].astype(str).str.contains("uksek").sum())
        hacim      = float(df_main["toplam_harcama"].sum())
        supheli    = 0
        fraud_oran = 0
        churn      = 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("👥 Total Customers", f"{toplam:,}")
    with k2: st.metric("🔴 High Risk",        f"{yuksek:,}",  delta=f"%{round(yuksek / max(toplam, 1) * 100, 1)}")
    with k3: st.metric("⚠️ Suspicious",       f"{supheli:,}", delta=f"%{fraud_oran}")
    with k4: st.metric("📉 Churn Risk",       f"{churn:,}")
    with k5: st.metric("💰 Volume",           f"${hacim / 1e6:.1f}M" if hacim > 1e6 else f"${hacim:,.0f}")

    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
    w1, w2 = st.columns([3, 2])

    with w1:
        np.random.seed(99)
        gunler = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
        base   = float(df["toplam_harcama"].sum()) / 30 if len(df) > 0 else 50000
        gunluk = np.random.lognormal(np.log(max(base, 1)), 0.15, 30)
        smoothed = pd.Series(gunluk).rolling(3, min_periods=1).mean()
        trend_c  = GREEN if gunluk[-1] > gunluk[-2] else RED
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gunler, y=gunluk, fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)", line=dict(color=CYAN, width=1.5),
            opacity=0.5, name="Günlük", showlegend=False))
        fig.add_trace(go.Scatter(x=gunler, y=smoothed,
            line=dict(color=trend_c, width=2.5), name="3G Ort.", showlegend=False))
        _pl = plotly_layout()
        _pl["margin"] = dict(l=30, r=20, t=45, b=30)
        fig.update_layout(title="📅 Son 30 Gün İşlem Trendi", height=240, **_pl)
        fig.update_xaxes(tickformat="%d %b", nticks=8)
        st.plotly_chart(fig, use_container_width=True)

    with w2:
        if ml_ozet:
            normal_n = int(ml_ozet.get("normal", toplam - supheli - yuksek))
            labels   = ["Normal", "Şüpheli", "Yüksek Risk"]
            values   = [normal_n, supheli, yuksek]
            colors   = [GREEN, ORANGE, RED]
        else:
            dusuk  = int(df["risk_seviyesi"].astype(str).str.contains("usuk").sum())
            orta_r = int(df["risk_seviyesi"].astype(str).str.contains("Fair").sum())
            yuk_r  = int(df["risk_seviyesi"].astype(str).str.contains("uksek").sum())
            labels = ["Düşük Risk", "Fair Risk", "Yüksek Risk"]
            values = [dusuk, orta_r, yuk_r]
            colors = [GREEN, ORANGE, RED]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.62,
            marker=dict(colors=colors, line=dict(color=T["bg_base"], width=2)),
            textfont=dict(family="DM Mono", size=10)))
        fig2.add_annotation(
            text=f"<b>{sum(values):,}</b>", x=0.5, y=0.5,
            showarrow=False, font=dict(family="Syne", size=22, color=GOLD))
        _pl2 = plotly_layout()
        _pl2["margin"] = dict(l=10, r=10, t=45, b=10)
        fig2.update_layout(title="🔥 Risk Dağılımı (ML)", height=240, **_pl2)
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Müşteri Tablosu", expanded=False):
        cols = [c for c in ["client_id","sehir","yas","toplam_harcama","islem_sayisi",
                             "risk_skoru","risk_seviyesi","fraud_tahmini","segment"]
                if c in df.columns]
        cA, cB = st.columns([4, 1])
        with cB:
            st.download_button(
                "⬇️ CSV", df[cols].to_csv(index=False).encode("utf-8"),
                "musteri.csv", "text/csv", use_container_width=True,
            )
        styled = df[cols].head(100).copy()
        if "toplam_harcama" in styled.columns:
            styled["toplam_harcama"] = styled["toplam_harcama"].apply(lambda x: f"${x:,.0f}")
        if "risk_skoru" in styled.columns:
            styled["risk_skoru"] = styled["risk_skoru"].apply(lambda x: f"{x:.1f}")
        st.dataframe(styled, use_container_width=True, height=380)


# ── MONTHLY VOLUME ─────────────────────────────────────────────────────────────
elif sayfa == "📈 Monthly Volume":
    st.markdown(section_header("Monthly Transaction Volume", "Time series — transaction volume trend", g2=GOLD), unsafe_allow_html=True)
    api_aylik = api_get("/stats/aylik", {"son_ay": 24})
    if api_aylik:
        da = pd.DataFrame(api_aylik)
        da["toplam"] = pd.to_numeric(da["toplam"], errors="coerce").fillna(0)
    else:
        np.random.seed(7)
        donemler = pd.date_range(end=pd.Timestamp.today(), periods=24, freq="ME")
        base = float(df["toplam_harcama"].sum()) / 24 if len(df) > 0 else 500000
        da = pd.DataFrame({
            "donem":  donemler.strftime("%Y-%m"),
            "toplam": base * np.linspace(0.85, 1.15, 24) * np.random.lognormal(0, .12, 24),
        })

    rolling_ort = pd.Series(da["toplam"].values).rolling(3, min_periods=1).mean()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("📅 Dönemler",      f"{len(da)}")
    with m2: st.metric("💰 Toplam",        f"${da['toplam'].sum() / 1e6:.1f}M")
    with m3: st.metric("📊 Aylık Ort.",    f"${da['toplam'].mean() / 1e3:.0f}K")
    with m4:
        son_deg = (da["toplam"].iloc[-1] / da["toplam"].iloc[-2] - 1) * 100 if len(da) >= 2 else 0
        st.metric("📈 Son Ay Değişim", f"%{son_deg:+.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=da["donem"], y=da["toplam"], fill="tozeroy",
        fillcolor="rgba(201,168,76,.07)", line=dict(color=GOLD, width=2.5), name="Aylık Hacim",
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=da["donem"], y=rolling_ort,
        line=dict(color=CYAN, width=1.5, dash="dot"), name="3A Ort."))
    fig.update_layout(title="📈 Aylık İşlem Hacmi (Son 24 Ay)", height=480, **plotly_layout())
    fig.update_xaxes(tickangle=-30, nticks=12)
    st.plotly_chart(fig, use_container_width=True)


# ── SEGMENTS ───────────────────────────────────────────────────────────────────
elif sayfa == "💎 Segments":
    st.markdown(section_header("Segment Distribution", "Customer segment breakdown", g2=CYAN), unsafe_allow_html=True)
    seg = df["segment"].astype(str).value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Pie(
            labels=seg.index, values=seg.values, hole=0.62,
            marker=dict(colors=[GOLD,CYAN,GREEN,RED], line=dict(color=T["bg_base"], width=2)),
            textfont=dict(family="DM Mono", size=11)))
        fig.add_annotation(
            text=f"<b>{len(df):,}</b>", x=0.5, y=0.5, showarrow=False,
            font=dict(family="Syne", size=28, color=GOLD))
        fig.update_layout(title="💎 Segment Dağılımı", height=500, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_df = df.groupby(df["segment"].astype(str)).agg(
            musteri    = ("client_id", "count"),
            ort_harcama= ("toplam_harcama", "mean"),
            ort_risk   = ("risk_skoru", "mean"),
        ).reset_index()
        fig2 = go.Figure(go.Bar(
            x=seg_df["segment"], y=seg_df["musteri"],
            marker=dict(color=[GOLD,CYAN,GREEN,RED]),
            text=seg_df["musteri"], textposition="outside",
            textfont=dict(family="DM Mono", size=10, color=T["text_secondary"])))
        fig2.update_layout(title="👥 Segmente Göre Müşteriler", height=500, **plotly_layout())
        st.plotly_chart(fig2, use_container_width=True)


# ── CATEGORY AVERAGES ──────────────────────────────────────────────────────────
elif sayfa == "📂 Category Averages":
    st.markdown(section_header("Category Averages", "Average spending by category", g2=ORANGE), unsafe_allow_html=True)
    top = df.groupby("kategori")["toplam_harcama"].mean().nlargest(10).reset_index()
    fig = go.Figure(go.Bar(
        y=top["kategori"], x=top["toplam_harcama"], orientation="h",
        marker=dict(color=list(range(len(top))), colorscale=[[0,"rgba(201,168,76,.25)"],[1,GOLD]]),
        text=[f"${v / 1000:.0f}K" for v in top["toplam_harcama"]], textposition="outside",
        textfont=dict(family="DM Mono", size=10, color=T["text_secondary"])))
    fig.update_layout(title="📂 Kategoriye Göre Ortalama Harcama", height=560, **plotly_layout())
    st.plotly_chart(fig, use_container_width=True)


# ── SPENDING × RISK ────────────────────────────────────────────────────────────
elif sayfa == "🎯 Spending × Risk":
    st.markdown(section_header("Spending × Risk", "Fraud detection scatter analysis", g2=RED), unsafe_allow_html=True)
    fig = go.Figure()
    for ft, col_c, label in [("Normal", GREEN, "Normal"), ("Suheli", RED, "Şüpheli")]:
        mask = df["fraud_tahmini"].astype(str).str.contains("pheli" if ft == "Suheli" else ft)
        if mask.sum() > 0:
            s = df[mask].sample(min(500, mask.sum()), random_state=42)
            fig.add_trace(go.Scatter(
                x=s["toplam_harcama"], y=s["risk_skoru"],
                mode="markers", name=label,
                marker=dict(color=col_c, size=5, opacity=0.7)))
    fig.update_xaxes(type="log", title="Harcama ($)")
    fig.update_yaxes(title="Risk Skoru")
    fig.update_layout(title="🎯 Harcama × Risk Dağılımı", height=580, **plotly_layout())
    st.plotly_chart(fig, use_container_width=True)


# ── TREND ANALYSIS ─────────────────────────────────────────────────────────────
elif sayfa == "📈 Trend Analysis":
    st.markdown(section_header("Trend Analysis", "Time series and category", g2=CYAN), unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Aylık Hacim", "🏷️ Kategori", "🌡️ Heatmap"])

    with tab1:
        np.random.seed(7)
        donemler = pd.date_range(end=pd.Timestamp.today(), periods=24, freq="ME")
        base = float(df["toplam_harcama"].sum()) / 24 if len(df) > 0 else 500000
        da   = pd.DataFrame({
            "donem":  donemler.strftime("%Y-%m"),
            "toplam": base * np.linspace(0.85, 1.15, 24) * np.random.lognormal(0, .12, 24),
        })
        rolling_ort = pd.Series(da["toplam"].values).rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=da["donem"], y=da["toplam"], fill="tozeroy",
            fillcolor="rgba(201,168,76,.07)", line=dict(color=GOLD, width=2.5), name="Aylık Hacim"))
        fig.add_trace(go.Scatter(
            x=da["donem"], y=rolling_ort,
            line=dict(color=CYAN, width=1.5, dash="dot"), name="3A Ort."))
        fig.update_layout(title="📈 Aylık Hacim", height=400, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dk = df.groupby("kategori")["toplam_harcama"].sum().reset_index().rename(columns={"toplam_harcama": "toplam"})
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=dk["kategori"], y=dk["toplam"],
                marker=dict(color=list(range(len(dk))), colorscale=[[0,"rgba(201,168,76,.3)"],[1,GOLD]])))
            fig.update_layout(title="💰 Kategori Toplam", height=380, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Pie(
                labels=dk["kategori"], values=dk["toplam"], hole=0.4,
                textfont=dict(family="DM Mono", size=9),
                marker=dict(line=dict(color=T["bg_base"], width=2))))
            fig2.update_layout(title="🥧 Dağılım", height=380, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        gunler = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"]
        heat   = np.random.lognormal(0, 0.5, (7, 24))
        heat[5:7, 10:22] *= 2.5
        heat[:5, 8:10]   *= 1.8
        fig = go.Figure(go.Heatmap(
            z=heat, x=[f"{h:02d}:00" for h in range(24)], y=gunler,
            colorscale=[[0, T["bg_base"]], [0.3, "rgba(201,168,76,.3)"], [1, GOLD]]))
        fig.update_layout(title="🌡️ İşlem Yoğunluğu Haritası", height=440, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)


# ── CUSTOMER ANALYSIS ──────────────────────────────────────────────────────────
elif sayfa == "🔍 Customer Analysis":
    st.markdown(section_header("Customer Analysis", "Segment, demographics and behavior", g2=GREEN), unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["💎 Segmentler", "👥 Demografi", "🔵 Davranış"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            seg = df.groupby(df["segment"].astype(str)).agg(ort_harcama=("toplam_harcama","mean")).reset_index()
            fig = go.Figure(go.Bar(
                x=seg["segment"], y=seg["ort_harcama"],
                marker=dict(color=["#5A6A7A",CYAN,GOLD,RED]),
                text=[f"${v/1000:.0f}K" for v in seg["ort_harcama"]], textposition="outside"))
            fig.update_layout(title="💎 Segmente Göre Ort. Harcama", height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            seg2 = df.groupby(df["segment"].astype(str)).size().reset_index(name="n")
            fig2 = go.Figure(go.Bar(
                x=seg2["segment"], y=seg2["n"],
                marker=dict(color=["#5A6A7A",CYAN,GOLD,RED]),
                text=seg2["n"], textposition="outside"))
            fig2.update_layout(title="👥 Segmente Göre Müşteri Sayısı", height=340, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="yas", nbins=25, color_discrete_sequence=[CYAN])
            fig.update_layout(title="👤 Yaş Dağılımı", height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ds2 = df.groupby("sehir").size().nlargest(10).reset_index(name="n")
            fig2 = go.Figure(go.Bar(
                x=ds2["sehir"], y=ds2["n"],
                marker=dict(color=ds2["n"], colorscale=[[0,"rgba(201,168,76,.2)"],[1,GOLD]])))
            fig2.update_layout(title="🏙️ Şehre Göre", height=340, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            smp = df.sample(min(400, len(df)), random_state=42)
            fig = px.scatter(
                smp, x="islem_sayisi", y="toplam_harcama",
                size="risk_skoru", color=smp["segment"].astype(str), size_max=22,
                color_discrete_map={"Bronze":"#5A6A7A","Silver":CYAN,"Gold":GOLD,"Platinum":RED})
            fig.update_layout(title="🔵 İşlem × Harcama × Risk", height=400, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            aktif = pd.cut(df["aktif_ay"], bins=[0,6,12,24,36], labels=["0-6 ay","7-12 ay","13-24 ay","25+ ay"])
            df2 = df.copy(); df2["ag"] = aktif
            ad  = df2.groupby("ag", observed=True)["toplam_harcama"].mean().reset_index()
            fig2 = go.Figure(go.Bar(
                x=ad["ag"].astype(str), y=ad["toplam_harcama"],
                marker=dict(color=[GREEN,CYAN,GOLD,RED]),
                text=[f"${v/1000:.0f}K" for v in ad["toplam_harcama"]], textposition="outside"))
            fig2.update_layout(title="📅 Aktivite × Harcama", height=400, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)


# ── RISK & FRAUD ───────────────────────────────────────────────────────────────
elif sayfa == "⚠️ Risk & Fraud":
    st.markdown(section_header("Risk & Fraud", "Real-time ML predictions", g2=RED), unsafe_allow_html=True)
    ml_df   = load_ml_data()
    ml_ozet = load_ml_ozet()

    if ml_df.empty:
        st.info("ℹ️ ML verisi bulunamadı — demo data gösteriliyor.")
        ml_df = df_main.copy()
        if "fraud_skoru"   not in ml_df.columns: ml_df["fraud_skoru"]   = ml_df["risk_skoru"] * 1.2
        if "fraud_tahmini" not in ml_df.columns: ml_df["fraud_tahmini"] = np.where(ml_df["risk_skoru"] > 30, "Yuksek Risk", "Normal")
        if "churn_skoru"   not in ml_df.columns: ml_df["churn_skoru"]   = np.random.exponential(15, len(ml_df)).clip(0, 100)

    toplam   = int(ml_ozet.get("toplam",      len(ml_df)))
    yuksek   = int(ml_ozet.get("yuksek_risk", int(ml_df["fraud_tahmini"].astype(str).str.contains("ksek").sum()) if "fraud_tahmini" in ml_df.columns else 0))
    supheli  = int(ml_ozet.get("supheli",     int(ml_df["fraud_tahmini"].astype(str).str.contains("pheli").sum()) if "fraud_tahmini" in ml_df.columns else 0))
    normal   = int(ml_ozet.get("normal",      toplam - yuksek - supheli))
    churn    = int(ml_ozet.get("churn_yuksek", 0))
    ort_skor = float(ml_ozet.get("ort_fraud_skoru", ml_df["fraud_skoru"].mean() if "fraud_skoru" in ml_df.columns else 0))

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1: st.metric("🔴 Yüksek Risk",    f"{yuksek:,}",  delta=f"%{round(yuksek/max(toplam,1)*100,1)}")
    with r2: st.metric("⚠️ Şüpheli",        f"{supheli:,}", delta=f"%{round(supheli/max(toplam,1)*100,1)}")
    with r3: st.metric("✅ Normal",          f"{normal:,}")
    with r4: st.metric("📊 Ort. Fraud Skor", f"{ort_skor:.1f}")
    with r5: st.metric("📉 Churn Risk",      f"{churn:,}")

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dağılım","🚨 Yüksek Risk Listesi","📉 Churn","🔬 Detay Analiz"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            if "fraud_skoru" in ml_df.columns:
                fig = go.Figure()
                for lvl, col_c, label in [("Normal",GREEN,"Normal"),("pheli",ORANGE,"Şüpheli"),("ksek",RED,"Yüksek Risk")]:
                    sub = (ml_df[ml_df["fraud_tahmini"].astype(str).str.contains(lvl)]["fraud_skoru"]
                           if "fraud_tahmini" in ml_df.columns else ml_df["fraud_skoru"])
                    if len(sub) > 0:
                        fig.add_trace(go.Histogram(x=sub, name=label, marker_color=col_c, opacity=0.75, nbinsx=30))
                fig.update_layout(title="📊 Fraud Skor Dağılımı (ML)", barmode="overlay", height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fraud_pct = round((supheli + yuksek) / max(toplam, 1) * 100, 1)
            fig2 = go.Figure(go.Pie(
                labels=["Normal","Şüpheli","Yüksek Risk"], values=[normal, supheli, yuksek], hole=0.62,
                marker=dict(colors=[GREEN,ORANGE,RED], line=dict(color=T["bg_base"], width=2))))
            fig2.add_annotation(
                text=f"<b>{fraud_pct}%</b>", x=0.5, y=0.5,
                showarrow=False, font=dict(family="Syne", size=16, color=RED))
            fig2.update_layout(title="🎯 Risk Dağılımı", height=380, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if "fraud_skoru" in ml_df.columns:
            yr = ml_df.nlargest(50, "fraud_skoru")
            show_cols = [c for c in ["client_id","fraud_skoru","fraud_tahmini",
                                      "fraud_skoru_xgb","anomali_skoru","tx_gece_oran",
                                      "tx_hata_oran","dark_web_oran"] if c in yr.columns]
            st.dataframe(yr[show_cols].head(50), use_container_width=True, height=380)

    with tab3:
        if "churn_skoru" in ml_df.columns:
            churn_df = ml_df.sort_values("churn_skoru", ascending=False)
            fig = go.Figure(go.Bar(
                x=churn_df["client_id"].astype(str).head(20),
                y=churn_df["churn_skoru"].head(20),
                marker=dict(color=churn_df["churn_skoru"].head(20),
                            colorscale=[[0,CYAN],[1,PURPLE]], showscale=False)))
            fig.update_layout(title="📉 En Yüksek Churn Riski", height=340, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            if "tx_gece_oran" in ml_df.columns and "fraud_skoru" in ml_df.columns:
                smp = ml_df.sample(min(300, len(ml_df)), random_state=42)
                fig = px.scatter(smp, x="tx_gece_oran", y="fraud_skoru",
                    color="fraud_tahmini" if "fraud_tahmini" in smp.columns else None,
                    title="Gece İşlem Oranı × Fraud Skor")
                fig.update_layout(height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "tx_hata_oran" in ml_df.columns and "fraud_skoru" in ml_df.columns:
                smp = ml_df.sample(min(300, len(ml_df)), random_state=42)
                fig2 = px.scatter(smp, x="tx_hata_oran", y="fraud_skoru",
                    color="fraud_tahmini" if "fraud_tahmini" in smp.columns else None,
                    title="Hata Oranı × Fraud Skor")
                fig2.update_layout(height=380, **plotly_layout())
                st.plotly_chart(fig2, use_container_width=True)


# ── GEOGRAPHIC ANALYSIS ────────────────────────────────────────────────────────
elif sayfa == "🗺️ Geographic Analysis":
    st.markdown(section_header("Geographic Analysis", "Regional distribution", g2=PURPLE), unsafe_allow_html=True)
    ds = df.groupby("sehir").agg(
        musteri_sayisi=("client_id", "count"),
        ort_risk      =("risk_skoru", "mean"),
        toplam_hacim  =("toplam_harcama", "sum"),
        ort_harcama   =("toplam_harcama", "mean"),
    ).reset_index()

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("🏙️ Şehirler",        f"{len(ds):,}")
    with k2: st.metric("👥 En Büyük",         ds.nlargest(1, "musteri_sayisi")["sehir"].values[0])
    with k3: st.metric("⚠️ En Yüksek Risk",   ds.nlargest(1, "ort_risk")["sehir"].values[0])
    with k4: st.metric("📊 Ort./Şehir",       f"{int(ds['musteri_sayisi'].mean()):,}")

    tab1, tab2 = st.tabs(["📊 Sıralama", "🗺️ Harita"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                ds.sort_values("musteri_sayisi", ascending=True).tail(10),
                x="musteri_sayisi", y="sehir", orientation="h", color="ort_risk",
                color_continuous_scale=[[0,GREEN],[0.5,ORANGE],[1,RED]])
            fig.update_layout(title="🏙️ Müşteri Sıralaması", height=400, **plotly_layout())
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(
                ds.sort_values("ort_harcama", ascending=True).tail(10),
                x="ort_harcama", y="sehir", orientation="h", color="ort_harcama",
                color_continuous_scale=[[0,"rgba(201,168,76,.3)"],[1,GOLD]])
            fig2.update_layout(title="💰 Ort. Harcama", height=400, **plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)
    with tab2:
        fig = px.treemap(
            ds, path=["sehir"], values="toplam_hacim", color="ort_risk",
            color_continuous_scale=[[0,"rgba(0,227,150,.8)"],[0.5,"rgba(255,107,53,.8)"],[1,"rgba(255,69,96,.8)"]])
        fig.update_layout(
            title="🗺️ Toplam Hacim", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color=T["text_primary"]),
            margin=dict(l=10,r=10,t=50,b=10), height=520)
        st.plotly_chart(fig, use_container_width=True)


# ── AI INSIGHTS ────────────────────────────────────────────────────────────────
elif sayfa == "🤖 AI Insights":
    st.markdown(section_header("AI Insights", "Model performance and detections", g2=CYAN), unsafe_allow_html=True)
    ml_df   = load_ml_data()
    metrics = load_model_metrics()
    tab1, tab2, tab3 = st.tabs(["📊 Model Performansı", "🚨 Şüpheli Tespitler", "🔍 Feature Analizi"])

    with tab1:
        if metrics:
            auc  = float(metrics.get("auc_roc",  0))
            f1   = float(metrics.get("f1_skoru", metrics.get("f1_score", 0)))
            prec = float(metrics.get("precision", 0))
            rec  = float(metrics.get("recall",    0))
            auc_c = GREEN if auc >= 0.85 else (ORANGE if auc >= 0.70 else RED)

            cols = st.columns(4)
            for col, (lbl, val_raw, pct, c) in zip(cols, [
                ("AUC-ROC",   auc,        auc * 100,  auc_c),
                ("F1 Score",  f1,         f1 * 100,   GOLD),
                ("Precision", prec,       prec * 100, CYAN),
                ("Recall",    rec,        rec * 100,  ORANGE),
            ]):
                with col:
                    v_str = f"{val_raw:.2f}" if lbl == "AUC-ROC" else f"{val_raw * 100:.1f}%"
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                        f"border:1px solid {c}44;border-radius:14px;padding:20px 22px;'>"
                        f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};"
                        f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>{lbl}</div>"
                        f"<div style='font-family:Syne;font-size:2.2rem;font-weight:800;color:{c};'>{v_str}</div>"
                        f"{score_bar(pct, c, 4)}</div>",
                        unsafe_allow_html=True,
                    )

            c1, c2 = st.columns(2)
            with c1:
                fpr_arr = np.linspace(0, 1, 100)
                exp     = max((1 - auc) / max(auc + 0.001, 0.001), 0.01)
                tpr_arr = np.sort(np.clip(np.power(fpr_arr, exp) + np.random.normal(0, 0.01, 100), 0, 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr_arr, y=fpr_arr,
                    line=dict(color=T["text_muted"], width=1, dash="dash"), name="Random (0.50)"))
                fig.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, fill="tozeroy",
                    fillcolor="rgba(201,168,76,.07)", line=dict(color=auc_c, width=2.5),
                    name=f"Model (AUC={auc:.2f})"))
                fig.update_xaxes(title="False Positive Rate")
                fig.update_yaxes(title="True Positive Rate")
                fig.update_layout(title="📊 ROC Eğrisi", height=380, **plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
                if cm and len(cm) == 2:
                    tn = int(cm[0][0] or 0); fp = int(cm[0][1] or 0)
                    fn = int(cm[1][0] or 0); tp = int(cm[1][1] or 0)
                else:
                    tn = fp = fn = tp = 0
                if tn + fp + fn + tp == 0:
                    st.warning("⚠️ Confusion matrix değerleri sıfır — model evaluation çalıştırılmamış olabilir.")
                else:
                    z_text = [[f"TN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]]
                    fig2 = go.Figure(go.Heatmap(
                        z=[[tn,fp],[fn,tp]],
                        x=["Pred: Normal","Pred: Fraud"], y=["Real: Normal","Real: Fraud"],
                        colorscale=[[0,T["bg_card"]],[1,GOLD]],
                        text=z_text, texttemplate="%{text}",
                        textfont=dict(family="DM Mono", size=12, color=T["text_primary"]), showscale=False))
                    fig2.update_layout(title="📋 Confusion Matrix", height=380, **plotly_layout())
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ Model metrikleri bulunamadı. `python src/ml_model.py` çalıştırın.")

    with tab2:
        if not ml_df.empty and "fraud_skoru" in ml_df.columns:
            yr = ml_df.nlargest(8, "fraud_skoru")
            ca, cb = st.columns(2)
            for i, (_, row) in enumerate(yr.iterrows()):
                fs_val = float(row.get("fraud_skoru", 0))
                bc     = RED if fs_val > 60 else ORANGE
                sev    = "Kritik" if fs_val > 60 else "Şüpheli"
                signals = []
                if float(row.get("tx_gece_oran", 0)) > 0.3: signals.append("Gece işlem fazla")
                if float(row.get("tx_hata_oran", 0)) > 0.1: signals.append("Yüksek hata oranı")
                if float(row.get("dark_web_oran", 0)) > 0:  signals.append("Dark web kartı")
                if int(row.get("iso_tahmin", 0)) == 1:       signals.append("Anomali tespit")
                sig_html = " · ".join(signals) if signals else "Genel risk skoru yüksek"
                with (ca if i % 2 == 0 else cb):
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,rgba(255,69,96,.05),{T['bg_card']});"
                        f"border-left:3px solid {bc};border-radius:10px;padding:14px 18px;"
                        f"margin:8px 0;border:1px solid rgba(255,69,96,.12);'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:8px;'>"
                        f"<span style='font-family:Syne;font-weight:700;color:{T['text_primary']};font-size:.95rem;'>"
                        f"Müşteri #{int(row.get('client_id', i))}</span>"
                        f"<span style='font-family:DM Mono;font-size:.65rem;color:{bc};"
                        f"background:{bc}18;padding:2px 10px;border-radius:20px;border:1px solid {bc}40;'>{sev}</span></div>"
                        f"<div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:{bc};margin-bottom:4px;'>"
                        f"{fs_val:.1f}<span style='font-size:.7rem;color:{T['text_muted']};'>/100</span></div>"
                        f"{score_bar(fs_val, bc)}"
                        f"<div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};margin-top:8px;line-height:1.7;'>{sig_html}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    with tab3:
        if metrics and "feature_importance" in metrics:
            try:
                fi_raw = metrics["feature_importance"]
                if isinstance(fi_raw, dict) and "feature" in fi_raw and "importance" in fi_raw:
                    fi = pd.DataFrame(fi_raw)
                elif isinstance(fi_raw, dict):
                    fi = pd.DataFrame(list(fi_raw.items()), columns=["feature","importance"])
                elif isinstance(fi_raw, list):
                    fi = pd.DataFrame(fi_raw)
                    fi.columns = [str(c).lower().strip() for c in fi.columns]
                else:
                    fi = pd.DataFrame(columns=["feature","importance"])
                fi = fi[["feature","importance"]].dropna()
                fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce").fillna(0)
                fi = fi.sort_values("importance", ascending=True).tail(15)
                if not fi.empty:
                    fig = go.Figure(go.Bar(
                        y=fi["feature"], x=fi["importance"], orientation="h",
                        marker=dict(color=fi["importance"], colorscale=[[0,"rgba(201,168,76,.3)"],[1,GOLD]]),
                        text=[f"{v:.3f}" for v in fi["importance"]], textposition="outside",
                        textfont=dict(family="DM Mono", size=9, color=T["text_secondary"])))
                    fig.update_layout(title="Feature Importance (XGBoost)", height=500, **plotly_layout())
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Feature importance yüklenirken hata: {e}")
        else:
            st.info("Feature importance için modeli yeniden eğitin.")


# ═══════════════════════════════════════════════════════════════════════════════
# ── CUSTOMER DETAIL ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif sayfa == "👤 Customer Detail":
    st.markdown(
        section_header("Customer Detail", "360° müşteri profili · ML tahminleri · İşlem geçmişi"),
        unsafe_allow_html=True,
    )

    @st.cache_data(ttl=600, show_spinner=False)
    def load_detail_data():
        ml_df2 = risk_df = users_df = cards_df = pd.DataFrame()
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                for tbl, key in [("client_ml","ml"), ("client_risk","risk")]:
                    try:
                        tmp = pd.read_sql(f"SELECT * FROM {tbl}", conn)
                        if key == "ml":   ml_df2  = tmp
                        else:             risk_df  = tmp
                    except Exception:
                        pass
                conn.close()
            except Exception:
                pass

        # CSV yükle — DÜZELTİLDİ: sayısal temizleme standartlaştırıldı
        for csv_f, lbl in [
            (os.path.join(DATA_DIR, "users_data.csv"), "users"),
            (os.path.join(DATA_DIR, "cards_data.csv"), "cards"),
        ]:
            if not os.path.exists(csv_f):
                continue
            try:
                tmp = pd.read_csv(csv_f)
                if lbl == "users":
                    for col in ("per_capita_income", "yearly_income", "total_debt"):
                        if col in tmp.columns:
                            tmp[col] = pd.to_numeric(
                                tmp[col].astype(str).str.replace(r"[$,]", "", regex=True),
                                errors="coerce",
                            ).fillna(0)
                    users_df = tmp
                else:
                    if "credit_limit" in tmp.columns:
                        tmp["credit_limit"] = pd.to_numeric(
                            tmp["credit_limit"].astype(str).str.replace(r"[$,]", "", regex=True),
                            errors="coerce",
                        ).fillna(0)
                    cards_df = tmp
            except Exception:
                pass
        return ml_df2, risk_df, users_df, cards_df

    ml_df2, risk_df, users_df, cards_df = load_detail_data()

    # ID listesi oluştur
    id_list = []
    for src in (ml_df2, risk_df, df_main):
        if len(src) > 0 and "client_id" in src.columns:
            id_list = sorted(src["client_id"].astype(int).tolist())
            break
    if not id_list:
        st.info("Müşteri verisi bulunamadı. `python src/ml_model.py` çalıştırın.")
        st.stop()

    secili = st.selectbox("Müşteri Seç", id_list, format_func=lambda x: f"#{x}")

    def safe_row(src: pd.DataFrame, col: str, val) -> pd.Series | None:
        if len(src) > 0 and col in src.columns and val in src[col].values:
            return src[src[col] == val].iloc[0]
        return None

    ml_r    = safe_row(ml_df2,   "client_id", secili)
    rr      = safe_row(risk_df,  "client_id", secili)
    ur      = safe_row(users_df, "id",        secili)

    # DÜZELTİLDİ: cards_df join koşulu güncellendi (int cast)
    if len(cards_df) > 0 and "client_id" in cards_df.columns:
        cards_r = cards_df[cards_df["client_id"].astype(int) == int(secili)]
    else:
        cards_r = pd.DataFrame()

    demo_r = None
    if rr is None:
        tmp = df_main[df_main["client_id"] == secili]
        if len(tmp) > 0:
            demo_r = tmp.iloc[0]

    # Temel değerler
    fs      = float(ml_r.get("fraud_skoru",   0)) if ml_r is not None else 0.0
    cs      = float(ml_r.get("churn_skoru",   0)) if ml_r is not None else 0.0
    anomali = float(ml_r.get("anomali_skoru", 0)) if ml_r is not None else 0.0
    fraud_t = str(ml_r.get("fraud_tahmini", "Normal")) if ml_r is not None else "Normal"
    churn_t = str(ml_r.get("churn_tahmini", "Bilinmiyor")) if ml_r is not None else "Bilinmiyor"

    fraud_c = RED    if "ksek" in fraud_t or fs >= 60 else (ORANGE if "pheli" in fraud_t or fs >= 30 else GREEN)
    churn_c = PURPLE if "ksek" in churn_t or cs >= 60 else (ORANGE if cs >= 30 else GREEN)

    toplam_harcama = (
        float(rr.get("toplam", 0)) if rr is not None else
        float(demo_r.get("toplam_harcama", 0)) if demo_r is not None else 0.0
    )
    islem_sayisi = (
        int(rr.get("islem", 0)) if rr is not None else
        int(demo_r.get("islem_sayisi", 0)) if demo_r is not None else 0
    )
    risk_skoru = (
        float(rr.get("risk_skoru", 0)) if rr is not None else
        float(demo_r.get("risk_skoru", 0)) if demo_r is not None else 0.0
    )
    kredi_skoru  = int(ur.get("credit_score", 0)) if ur is not None else 0
    yillik_gelir = float(ur.get("yearly_income", 0)) if ur is not None else 0.0
    borc_oran    = float(ur.get("borc_gelir_orani", 0)) if ur is not None else 0.0

    risk_c  = RED   if risk_skoru > 50 else (ORANGE if risk_skoru > 20 else GREEN)
    kredi_c = GREEN if kredi_skoru >= 700 else (ORANGE if kredi_skoru >= 600 else RED)

    # ── KPI Banner ──────────────────────────────────────────────────────────
    kpi_items = [
        ("Harcama",      f"${toplam_harcama:,.0f}",           GOLD,    T["border"]),
        ("İşlem Sayısı", f"{islem_sayisi:,}",                 CYAN,    T["border"]),
        ("Risk Skoru",   f"{risk_skoru:.1f}",                 risk_c,  f"{risk_c}22"),
        ("Fraud Skoru",  f"{fs:.1f}",                         fraud_c, f"{fraud_c}22"),
        ("Churn Skoru",  f"{cs:.1f}",                         churn_c, f"{churn_c}22"),
        ("Kredi Skoru",  str(kredi_skoru) if kredi_skoru else "—", kredi_c, f"{kredi_c}22"),
    ]
    cols_kpi = st.columns(6)
    for col, (lbl, val, c, bdr) in zip(cols_kpi, kpi_items):
        with col:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                f"border:1px solid {bdr};border-radius:14px;padding:16px 18px;'>"
                f"<div style='font-family:DM Mono;font-size:.52rem;color:{T['text_muted']};"
                f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>{lbl}</div>"
                f"<div style='font-family:Syne;font-size:1.4rem;font-weight:800;color:{c};'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    tab_profil, tab_ai, tab_txler, tab_gecmis = st.tabs([
        "👤 Profil", "🧠 AI Analiz Kartı", "📋 İşlem Geçmişi", "📈 Risk Geçmişi"
    ])

    # ── TAB: PROFİL ─────────────────────────────────────────────────────────
    with tab_profil:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            # XGB skoru
            xgb_val = round(float(ml_r["fraud_skoru_xgb"]), 1) if ml_r is not None and ml_r.get("fraud_skoru_xgb") is not None else None
            xgb_str = str(xgb_val) if xgb_val is not None else "—"
            iso_val = ml_r.get("iso_tahmin", 0) if ml_r is not None else 0
            iso_label = "🚨 Anomali" if int(iso_val) == 1 else "✅ Normal"
            iso_color = RED if int(iso_val) == 1 else GREEN

            st.markdown(
                f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                f"border:1px solid {T['border']};border-radius:16px;padding:22px 26px;margin-bottom:16px;'>"
                f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:14px;'>"
                f"ML Tahmin Özeti — Müşteri #{secili}</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:18px;'>"
                f"<div>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>FRAUD TAHMİN</div>"
                f"<div style='font-family:Syne;font-size:.9rem;font-weight:800;color:{fraud_c};margin-bottom:2px;'>{fraud_t}</div>"
                f"<div style='font-family:DM Mono;font-size:.7rem;color:{fraud_c};'>Skor: {fs:.1f}/100</div>"
                f"{score_bar(fs, fraud_c)}</div>"
                f"<div>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>CHURN TAHMİN</div>"
                f"<div style='font-family:Syne;font-size:.9rem;font-weight:800;color:{churn_c};margin-bottom:2px;'>{churn_t}</div>"
                f"<div style='font-family:DM Mono;font-size:.7rem;color:{churn_c};'>Skor: {cs:.1f}/100</div>"
                f"{score_bar(cs, churn_c)}</div>"
                f"<div>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>XGB SKOR</div>"
                f"<div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:{ORANGE};'>{xgb_str}</div>"
                f"</div>"
                f"<div>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>ANOMALİ</div>"
                f"<div style='font-family:Syne;font-size:.9rem;font-weight:800;color:{iso_color};'>{iso_label}</div>"
                f"</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            # Kişisel bilgiler
            if ur is not None:
                yas  = int(ur.get("current_age", 0))
                gelir = float(ur.get("yearly_income", 0))
                borc  = float(ur.get("total_debt", 0))
                borc_c = RED if borc > gelir * 0.5 else ORANGE
                oran_c = RED if borc_oran > 0.5 else ORANGE
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;'>"
                    f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;margin-bottom:12px;'>Kişisel Bilgiler</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;'>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>YAŞ</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{T['text_primary']};'>{yas}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>YILLIK GELİR</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{GOLD};'>${gelir:,.0f}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>TOPLAM BORÇ</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{borc_c};'>${borc:,.0f}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>BORÇ/GELİR</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{oran_c};'>{borc_oran:.1%}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>KREDİ SKORU</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{kredi_c};'>{kredi_skoru}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        with col_b:
            # SHAP faktörleri
            if ml_r is not None:
                shap_rows = []
                for i in (1, 2, 3):
                    feat = ml_r.get(f"shap_reason_{i}")
                    val  = ml_r.get(f"shap_val_{i}")
                    if feat and val is not None:
                        shap_rows.append((str(feat), float(val)))
                if shap_rows:
                    rows_html = ""
                    for feat, v in shap_rows:
                        vc = RED if v > 0 else GREEN
                        rows_html += (
                            f"<div style='display:flex;justify-content:space-between;align-items:center;"
                            f"padding:8px 0;border-bottom:1px solid {T['border']};'>"
                            f"<div style='font-family:DM Mono;font-size:.68rem;color:{T['text_secondary']};'>{feat}</div>"
                            f"<div style='font-family:Syne;font-size:.85rem;font-weight:700;color:{vc};'>{v:+.4f}</div></div>"
                        )
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                        f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;margin-bottom:16px;'>"
                        f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                        f"text-transform:uppercase;margin-bottom:12px;'>SHAP — Model Karar Nedenleri</div>"
                        f"{rows_html}</div>",
                        unsafe_allow_html=True,
                    )

            # Kart bilgileri
            if len(cards_r) > 0:
                dw_count = int(cards_r.get("dark_web_flag", pd.Series([0])).sum()) if "dark_web_flag" in cards_r.columns else 0
                bdr_col  = "rgba(255,69,96,.25)" if dw_count > 0 else T["border"]
                dw_label = f"🚨 {dw_count} kart" if dw_count > 0 else "✅ Temiz"
                dw_color = RED if dw_count > 0 else GREEN
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:1px solid {bdr_col};border-radius:14px;padding:18px 22px;'>"
                    f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;margin-bottom:12px;'>Kart Bilgileri</div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>KART SAYISI</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{CYAN};'>{len(cards_r)}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};'>DARK WEB</div>"
                    f"<div style='font-family:Syne;font-size:1.1rem;font-weight:700;color:{dw_color};'>{dw_label}</div></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

    # ── TAB: AI ANALİZ KARTI ────────────────────────────────────────────────
    with tab_ai:
        if ml_r is not None:
            client_dict = {**dict(ml_r), "client_id": secili}
            ai_card_data = api_get(f"/clients/{secili}/ai-card")

            if ai_card_data:
                ra      = ai_card_data.get("ai_risk_assessment", {})
                rs_raw  = ai_card_data.get("raw_scores", {})
                ai_factors  = ai_card_data.get("ai_factors", [])
                ai_explain  = ai_card_data.get("ai_decision_explanation", {})
                risk_level  = ra.get("risk_level", "Medium")
                conf        = ra.get("confidence", 75)
                reco        = ra.get("recommendation", "—")
                behavioral  = ra.get("behavioral_status", "—")
                pattern     = ra.get("transaction_pattern_analysis", "—")
            else:
                risk_level = "High" if fs >= 60 else ("Medium" if fs >= 30 else "Low")
                conf       = 85 if fs >= 60 else (72 if fs >= 30 else 90)
                behavioral = "Anormal" if fs >= 60 else ("Dikkat" if fs >= 30 else "Normal")
                pattern    = "Yüksek risk örüntüsü." if fs >= 60 else ("Bazı alışılmadık örüntüler." if fs >= 30 else "Normal davranış.")
                reco       = "🚨 Acil inceleme." if fs >= 60 else ("⚠️ Yakın takip." if fs >= 30 else "✅ Standart izleme.")
                ai_factors = []
                ai_explain = {}
                rs_raw     = {}

            rc = {"High": RED, "Medium": ORANGE, "Low": GREEN}.get(risk_level, GOLD)

            with st.spinner("🧠 AI yorumu üretiliyor..."):
                ai_comment = call_claude_for_customer(client_dict)

            c1, c2 = st.columns([3, 2])
            with c1:
                # ham skor yardımcısı
                def _rs(key):
                    return float(rs_raw.get(key, client_dict.get(key, 0)) or 0)

                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:2px solid {rc}44;border-radius:18px;padding:26px 28px;margin-bottom:16px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px;'>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:6px;'>AI Risk Değerlendirmesi</div>"
                    f"<div style='font-family:Syne;font-size:1.6rem;font-weight:800;color:{rc};'>{risk_level} Risk</div></div>"
                    f"<div style='text-align:right;'>"
                    f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>GÜVEN</div>"
                    f"<div style='font-family:Syne;font-size:1.2rem;font-weight:800;color:{CYAN};'>%{conf}</div></div></div>"
                    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;'>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>DAVRANIŞSAL DURUM</div>"
                    f"<div style='font-family:DM Sans;font-size:.85rem;font-weight:600;color:{T['text_primary']};'>{behavioral}</div></div>"
                    f"<div><div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>İŞLEM ÖRÜNTÜSÜ</div>"
                    f"<div style='font-family:DM Sans;font-size:.85rem;color:{T['text_secondary']};'>{pattern}</div></div></div>"
                    f"<div style='background:{rc}10;border:1px solid {rc}30;border-radius:10px;padding:12px 16px;margin-bottom:14px;'>"
                    f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>ÖNERİ</div>"
                    f"<div style='font-family:DM Sans;font-size:.88rem;font-weight:600;color:{rc};'>{reco}</div></div>"
                    f"<div style='background:{T['bg_card2']};border-radius:10px;padding:14px 16px;border:1px solid {T['border']};'>"
                    f"<div style='font-family:DM Mono;font-size:.55rem;color:{CYAN};margin-bottom:6px;text-transform:uppercase;'>"
                    f"🧠 Claude AI Yorumu</div>"
                    f"<div style='font-family:DM Sans;font-size:.84rem;color:{T['text_secondary']};line-height:1.7;'>{ai_comment}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

            with c2:
                score_items = [
                    ("Fraud Skoru",  fs,                   f"{fs:.1f}",              fraud_c),
                    ("Churn Skoru",  cs,                   f"{cs:.1f}",              churn_c),
                    ("Dark Web",     _rs("dark_web_oran") * 100, f"{_rs('dark_web_oran'):.1%}", RED),
                    ("Gece İşlem",   _rs("tx_gece_oran")  * 100, f"{_rs('tx_gece_oran'):.1%}",  ORANGE),
                    ("Hata Oranı",   _rs("tx_hata_oran")  * 100, f"{_rs('tx_hata_oran'):.1%}",  ORANGE),
                ]
                rows_html = ""
                for lbl, pct, v_fmt, cl in score_items:
                    rows_html += (
                        f"<div style='margin-bottom:12px;'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px;'>"
                        f"<div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};'>{lbl}</div>"
                        f"<div style='font-family:Syne;font-size:.8rem;font-weight:700;color:{cl};'>{v_fmt}</div></div>"
                        f"{score_bar(pct, cl)}</div>"
                    )
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;margin-bottom:12px;'>"
                    f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;margin-bottom:14px;'>Ham Skor Göstergesi</div>"
                    f"{rows_html}</div>",
                    unsafe_allow_html=True,
                )

                if ai_factors:
                    col_map = {"red": RED, "orange": ORANGE, "green": GREEN}
                    fac_html = ""
                    for f_item in ai_factors[:5]:
                        fc = col_map.get(f_item.get("color","green"), GREEN)
                        fac_html += (
                            f"<div style='display:flex;align-items:flex-start;gap:10px;"
                            f"padding:8px 0;border-bottom:1px solid {T['border']};'>"
                            f"<span style='font-size:1.1rem;'>{f_item.get('icon','•')}</span>"
                            f"<div style='flex:1;'>"
                            f"<div style='font-family:DM Sans;font-size:.78rem;font-weight:600;color:{fc};'>"
                            f"{f_item.get('factor','')}</div>"
                            f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};margin-top:2px;'>"
                            f"{f_item.get('detail','')}</div></div></div>"
                        )
                    st.markdown(
                        f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                        f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;'>"
                        f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};"
                        f"text-transform:uppercase;margin-bottom:10px;'>AI Karar Faktörleri</div>"
                        f"{fac_html}</div>",
                        unsafe_allow_html=True,
                    )

            if ai_explain:
                checks = [
                    ("Yüksek İşlem Tutarı", ai_explain.get("high_tx_amount",    False), RED),
                    ("Olağandışı Frekans",  ai_explain.get("unusual_frequency",  False), ORANGE),
                    ("Hata Örüntüsü",       ai_explain.get("error_pattern",      False), ORANGE),
                    ("Harcama Kayması",      ai_explain.get("spending_shift",     False), ORANGE),
                    ("Anomali Tespit",       ai_explain.get("anomaly_detected",   False), RED),
                ]
                boxes_html = ""
                for lbl, v, c in checks:
                    icon  = "⚠️" if v else "✅"
                    clr   = c if v else T["border"]
                    clrt  = c if v else T["text_muted"]
                    fw    = "600" if v else "400"
                    boxes_html += (
                        f"<div style='background:{clr}18;border:1px solid {clr}40;"
                        f"border-radius:10px;padding:12px;text-align:center;'>"
                        f"<div style='font-size:1.4rem;margin-bottom:4px;'>{icon}</div>"
                        f"<div style='font-family:DM Mono;font-size:.55rem;color:{clrt};font-weight:{fw};'>{lbl}</div></div>"
                    )
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;margin-top:12px;'>"
                    f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;letter-spacing:.12em;margin-bottom:14px;'>AI Karar Açıklama Paneli</div>"
                    f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px;'>"
                    f"{boxes_html}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Bu müşteri için ML verisi bulunamadı.")

    # ── TAB: İŞLEM GEÇMİŞİ ─────────────────────────────────────────────────
    with tab_txler:
        st.markdown(
            f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};"
            f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:16px;'>"
            f"Fraud Durum Renk Sistemi: "
            f"<span style='color:{GREEN};'>■ Normal</span> &nbsp;"
            f"<span style='color:{ORANGE};'>■ Şüpheli</span> &nbsp;"
            f"<span style='color:{RED};'>■ Fraud Risk</span></div>",
            unsafe_allow_html=True,
        )
        tx_limit = st.slider("Gösterilecek işlem sayısı", 20, 200, 50, key="tx_limit_slider")
        tx_data  = api_get(f"/transactions/{secili}/fraud-status", params={"limit": tx_limit})

        if tx_data and tx_data.get("transactions"):
            txs     = tx_data["transactions"]
            summary = tx_data.get("summary", {})
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("Toplam",     summary.get("total", len(txs)))
            with s2: st.metric("✅ Normal",   summary.get("normal", 0))
            with s3: st.metric("⚠️ Şüpheli", summary.get("suspicious", 0))
            with s4: st.metric("🚨 Fraud",   summary.get("fraud_risk", 0))

            sc_map = {"Normal": GREEN, "Suspicious": ORANGE, "Fraud Risk": RED}
            rows_html = ""
            for tx in txs:
                status = tx.get("status", "Normal")
                sc     = sc_map.get(status, T["text_muted"])
                reasons = " · ".join(tx.get("reasons", [])) or "—"
                rows_html += (
                    f"<tr style='border-bottom:1px solid {T['border']};'>"
                    f"<td style='padding:10px 12px;font-family:DM Mono;font-size:.72rem;color:{T['text_secondary']};'>{tx.get('date','—')}</td>"
                    f"<td style='padding:10px 12px;font-family:Syne;font-size:.85rem;font-weight:700;color:{GOLD};'>${abs(tx.get('amount',0)):,.2f}</td>"
                    f"<td style='padding:10px 12px;font-family:DM Mono;font-size:.7rem;color:{T['text_secondary']};'>{tx.get('merchant_city','—')}</td>"
                    f"<td style='padding:10px 12px;font-family:DM Mono;font-size:.7rem;color:{T['text_muted']};'>{tx.get('category','—')}</td>"
                    f"<td style='padding:10px 12px;font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};'>{tx.get('use_chip','—')}</td>"
                    f"<td style='padding:10px 12px;'>{status_badge(status)}</td>"
                    f"<td style='padding:10px 12px;font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};max-width:200px;'>{reasons}</td>"
                    f"</tr>"
                )
            thead_cols = ["Tarih","Tutar","Şehir","Kategori","Chip","Durum","Risk Nedeni"]
            thead_html = "".join(
                f"<th style='padding:10px 12px;font-family:DM Mono;font-size:.58rem;"
                f"color:{T['text_muted']};text-align:left;text-transform:uppercase;'>{c}</th>"
                for c in thead_cols
            )
            st.markdown(
                f"<div style='background:{T['bg_card']};border:1px solid {T['border']};"
                f"border-radius:14px;overflow:hidden;'>"
                f"<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='background:{T['bg_card2']};'>{thead_html}</tr></thead>"
                f"<tbody>{rows_html}</tbody></table></div>",
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "API bağlantısı yok veya bu müşteri için işlem bulunamadı. "
                "API'yi başlatın: `uvicorn src.api:app --port 8000`"
            )
            if ml_r is not None:
                np.random.seed(secili)
                avg   = float(ml_r.get("tx_ortalama_tutar", 150) or 150)
                n_tx  = min(int(ml_r.get("tx_islem_sayisi", 10) or 10), tx_limit)
                cats  = ["Market","Restaurant","Online Shopping","Fuel","Health","Entertainment"]
                chips = ["Chip Transaction","Online Transaction","Swipe Transaction"]
                cities = ["İstanbul","Ankara","İzmir","Bursa","Antalya"]
                err_rate = float(ml_r.get("tx_hata_oran", 0.05) or 0.05)

                demo_rows = ""
                for _ in range(n_tx):
                    amt      = avg * np.random.uniform(0.3, 3.0)
                    chip     = np.random.choice(chips)
                    has_err  = np.random.random() < err_rate
                    is_online = "online" in chip.lower()
                    pts = (2 if amt > avg * 2.5 else 0) + (2 if has_err else 0) + (1 if is_online else 0)
                    status   = "Fraud Risk" if pts >= 4 else ("Suspicious" if pts >= 1 else "Normal")
                    sc_map2  = {"Normal": GREEN, "Suspicious": ORANGE, "Fraud Risk": RED}
                    sc       = sc_map2[status]
                    neden    = "Yüksek tutar" if amt > avg * 2.5 else ("Hata" if has_err else ("Online" if is_online else "—"))
                    date_str = (datetime.now() - pd.Timedelta(days=int(np.random.randint(0,90)))).strftime("%Y-%m-%d %H:%M")
                    demo_rows += (
                        f"<tr style='border-bottom:1px solid {T['border']};'>"
                        f"<td style='padding:8px 12px;font-family:DM Mono;font-size:.72rem;color:{T['text_secondary']};'>{date_str}</td>"
                        f"<td style='padding:8px 12px;font-family:Syne;font-size:.85rem;font-weight:700;color:{GOLD};'>${abs(amt):,.2f}</td>"
                        f"<td style='padding:8px 12px;font-family:DM Mono;font-size:.7rem;color:{T['text_secondary']};'>{np.random.choice(cities)}</td>"
                        f"<td style='padding:8px 12px;font-family:DM Mono;font-size:.7rem;color:{T['text_muted']};'>{np.random.choice(cats)}</td>"
                        f"<td style='padding:8px 12px;font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};'>{chip}</td>"
                        f"<td style='padding:8px 12px;'>{status_badge(status)}</td>"
                        f"<td style='padding:8px 12px;font-family:DM Mono;font-size:.62rem;color:{sc};'>{neden}</td>"
                        f"</tr>"
                    )
                thead_html2 = "".join(
                    f"<th style='padding:10px 12px;font-family:DM Mono;font-size:.58rem;"
                    f"color:{T['text_muted']};text-align:left;'>{c}</th>"
                    for c in ["Tarih","Tutar","Şehir","Kategori","Chip","Durum","Neden"]
                )
                st.markdown(
                    f"<div style='font-family:DM Mono;font-size:.6rem;color:{ORANGE};margin-bottom:8px;'>⚠️ Demo veri — API bağlı değil</div>"
                    f"<div style='background:{T['bg_card']};border:1px solid {T['border']};border-radius:14px;overflow:hidden;'>"
                    f"<table style='width:100%;border-collapse:collapse;'>"
                    f"<thead><tr style='background:{T['bg_card2']};'>{thead_html2}</tr></thead>"
                    f"<tbody>{demo_rows}</tbody></table></div>",
                    unsafe_allow_html=True,
                )

    # ── TAB: RİSK GEÇMİŞİ ──────────────────────────────────────────────────
    with tab_gecmis:
        days_opt  = st.select_slider("Dönem", options=[7,14,30,60,90,180,365], value=90, key="risk_hist_days")
        hist_data = api_get(f"/clients/{secili}/risk-history", params={"days": days_opt})

        if hist_data and hist_data.get("history"):
            history = hist_data["history"]
            trend   = hist_data.get("trend", "stable")
            hdf     = pd.DataFrame(history)
            hdf["date"] = pd.to_datetime(hdf["date"])
            trend_c = RED if trend == "rising" else GREEN
            trend_l = "📈 Yükseliyor" if trend == "rising" else "📉 Düşüyor"
            t1, t2, t3 = st.columns(3)
            with t1: st.metric("Güncel Skor", f"{hist_data.get('current_score', fs):.1f}")
            with t2: st.metric("Trend",        trend_l)
            with t3: st.metric("Dönem",        f"{days_opt} gün")
        else:
            # Simüle et
            rng = random.Random(secili * 31 + days_opt)
            score_sim = max(0.0, fs - rng.uniform(5, 25))
            history_sim = []
            for i in range(days_opt, -1, -1):
                score_sim = min(100.0, max(0.0, score_sim + rng.uniform(-3.5, 4.0)))
                if i == 0:
                    score_sim = fs
                history_sim.append({
                    "date": datetime.now() - pd.Timedelta(days=i),
                    "fraud_score": round(score_sim, 2),
                })
            hdf     = pd.DataFrame(history_sim)
            trend_c = RED if hdf["fraud_score"].iloc[-1] > hdf["fraud_score"].iloc[0] else GREEN

        fig = go.Figure()
        fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN,  opacity=0.04, line_width=0)
        fig.add_hrect(y0=30, y1=60,  fillcolor=ORANGE, opacity=0.04, line_width=0)
        fig.add_hrect(y0=60, y1=100, fillcolor=RED,    opacity=0.05, line_width=0)
        y_col = "fraud_score" if "fraud_score" in hdf.columns else hdf.columns[-1]
        fig.add_trace(go.Scatter(
            x=hdf["date"], y=hdf[y_col], fill="tozeroy",
            fillcolor=f"{trend_c}0D", line=dict(color=trend_c, width=2.5), name="Risk Skoru",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Risk: %{y:.1f}<extra></extra>"))
        fig.add_hline(y=30, line=dict(color=ORANGE, width=1, dash="dot"), annotation_text="Şüpheli (30)", annotation_position="right")
        fig.add_hline(y=60, line=dict(color=RED,    width=1, dash="dot"), annotation_text="Yüksek Risk (60)", annotation_position="right")
        fig.update_xaxes(title="Tarih", tickformat="%d %b")
        fig.update_yaxes(title="Risk Skoru", range=[0, 105])
        title_suffix = "" if (hist_data and hist_data.get("history")) else " (simüle)"
        fig.update_layout(
            title=f"📈 Müşteri #{secili} — {days_opt} Günlük Risk Geçmişi{title_suffix}",
            height=420, **plotly_layout())
        st.plotly_chart(fig, use_container_width=True)
        if not (hist_data and hist_data.get("history")):
            st.caption("⚠️ API bağlantısı yok — simüle edilmiş veri gösteriliyor.")


# ═══════════════════════════════════════════════════════════════════════════════
# ── YENİ TAHMİN FORMU ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif sayfa == "🧠 New Prediction":
    st.markdown(
        section_header("New Prediction", "Yeni müşteri risk tahmini — ML + AI yorum"),
        unsafe_allow_html=True,
    )
    col_form, col_result = st.columns([2, 3])

    with col_form:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
            f"border:1px solid {T['border']};border-radius:16px;padding:22px 24px;'>"
            f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};"
            f"text-transform:uppercase;letter-spacing:.15em;margin-bottom:4px;'>Müşteri Veri Girişi</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.form("predict_form"):
            client_id_inp = st.number_input("Client ID (opsiyonel)", min_value=0, value=0, step=1)
            income        = st.number_input("Yıllık Gelir (₺)", min_value=0.0, value=50000.0, step=1000.0)
            tx_count      = st.number_input("İşlem Sayısı", min_value=0, value=30, step=1)
            avg_spending  = st.number_input("Ortalama Harcama (₺)", min_value=0.0, value=150.0, step=10.0)
            category      = st.selectbox("Harcama Kategorisi", [
                "Other","Market","Restaurant","Online Shopping","Health",
                "Entertainment","Electronics","Fuel","Transport","Education",
            ])
            debt_ratio    = st.slider("Borç/Gelir Oranı", 0.0, 1.0, 0.25, 0.01)
            submitted     = st.form_submit_button("🔮 Tahmin Yap", use_container_width=True)

    with col_result:
        if submitted:
            payload = {
                "client_id":         int(client_id_inp) if client_id_inp > 0 else None,
                "income":            float(income),
                "transaction_count": int(tx_count),
                "avg_spending":      float(avg_spending),
                "category":          category,
                "debt_ratio":        float(debt_ratio),
            }
            result = api_post("/predict", payload)

            # Hesapla (API veya kural tabanlı)
            if result:
                score   = float(result.get("risk_score", 0))
                level   = result.get("risk_level", "Low")
                conf    = result.get("confidence", 80)
                reco    = result.get("recommendation", "—")
                fp      = float(result.get("fraud_probability", score / 100))
                factors = result.get("ai_factors", [])
                source  = "ML Model"
            else:
                score = 0.0
                if debt_ratio > 0.6:             score += 30
                elif debt_ratio > 0.4:           score += 15
                if avg_spending > 500:           score += 20
                elif avg_spending > 300:         score += 10
                if tx_count > 100:               score += 15
                elif tx_count > 60:              score += 8
                if category in ("Online Shopping","Entertainment","Electronics"): score += 10
                if income < 20000:               score += 15
                elif income < 40000:             score += 5
                score   = min(score, 100.0)
                level   = "High" if score >= 60 else ("Medium" if score >= 30 else "Low")
                conf    = 70
                reco    = ("🚨 Acil inceleme önerilir." if score >= 60
                           else ("⚠️ Yakın takip önerilir." if score >= 30 else "✅ Standart izleme yeterli."))
                fp      = score / 100
                factors = []
                source  = "Kural Tabanlı (API offline)"

            rc = {"High": RED, "Medium": ORANGE, "Low": GREEN}.get(level, GOLD)

            # Claude yorumu
            pseudo = {
                "client_id":       client_id_inp or "Yeni",
                "fraud_skoru":     score,
                "churn_skoru":     0,
                "anomali_skoru":   0,
                "dark_web_oran":   0,
                "tx_gece_oran":    0,
                "tx_hata_oran":    debt_ratio * 0.3,
                "fraud_tahmini":   level,
                "tx_islem_sayisi": tx_count,
            }
            with st.spinner("🧠 AI yorumu üretiliyor..."):
                ai_comment = call_claude_for_customer(pseudo)

            st.markdown(
                f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                f"border:2px solid {rc}44;border-radius:18px;padding:26px 28px;margin-bottom:16px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;'>"
                f"<div>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};"
                f"text-transform:uppercase;margin-bottom:6px;'>Tahmin Sonucu — {source}</div>"
                f"<div style='font-family:Syne;font-size:2rem;font-weight:800;color:{rc};'>{level} Risk</div></div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-family:Syne;font-size:3rem;font-weight:800;color:{rc};line-height:1;'>{score:.0f}</div>"
                f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};'>/100 risk skoru</div></div></div>"
                f"{score_bar(score, rc, 8)}"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:18px;margin-bottom:18px;'>"
                f"<div style='text-align:center;'>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>FRAUD OLASILIĞI</div>"
                f"<div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:{rc};'>{fp:.1%}</div></div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>GÜVEN</div>"
                f"<div style='font-family:Syne;font-size:1.3rem;font-weight:800;color:{CYAN};'>%{conf}</div></div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>BORÇ/GELİR</div>"
                f"<div style='font-family:Syne;font-size:1.3rem;font-weight:800;"
                f"color:{RED if debt_ratio > 0.5 else ORANGE};'>{debt_ratio:.0%}</div></div></div>"
                f"<div style='background:{rc}10;border:1px solid {rc}30;border-radius:10px;padding:12px 16px;margin-bottom:14px;'>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{T['text_muted']};margin-bottom:4px;'>ÖNERİ</div>"
                f"<div style='font-family:DM Sans;font-size:.9rem;font-weight:600;color:{rc};'>{reco}</div></div>"
                f"<div style='background:{T['bg_card2']};border-radius:10px;padding:14px 16px;border:1px solid {T['border']};'>"
                f"<div style='font-family:DM Mono;font-size:.55rem;color:{CYAN};margin-bottom:6px;text-transform:uppercase;'>"
                f"🧠 Claude AI Yorumu</div>"
                f"<div style='font-family:DM Sans;font-size:.84rem;color:{T['text_secondary']};line-height:1.7;'>{ai_comment}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            if factors:
                fh = "".join(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:6px 0;"
                    f"border-bottom:1px solid {T['border']};'>"
                    f"<div style='width:6px;height:6px;border-radius:50%;background:{rc};flex-shrink:0;'></div>"
                    f"<div style='font-family:DM Mono;font-size:.68rem;color:{T['text_secondary']};'>{f}</div></div>"
                    for f in factors
                )
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                    f"border:1px solid {T['border']};border-radius:14px;padding:18px 22px;'>"
                    f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};"
                    f"text-transform:uppercase;margin-bottom:10px;'>Risk Faktörleri</div>"
                    f"{fh}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='background:linear-gradient(135deg,{T['bg_card']},{T['bg_card2']});"
                f"border:1px dashed {T['border']};border-radius:18px;padding:60px 30px;"
                f"text-align:center;'>"
                f"<div style='font-size:3rem;margin-bottom:16px;opacity:.4;'>🔮</div>"
                f"<div style='font-family:Syne;font-size:1rem;font-weight:700;color:{T['text_muted']};margin-bottom:8px;'>"
                f"Tahmin Sonucu Burada Görünecek</div>"
                f"<div style='font-family:DM Mono;font-size:.68rem;color:{T['text_muted']};'>"
                f"Formu doldurup Tahmin Yap butonuna tıklayın</div></div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ── ADMIN PANELİ ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif sayfa == "⚙️ Admin":
    if current_role != "admin":
        st.error("Bu sayfaya erişim yetkiniz yok.")
        st.stop()
    if not AUTH_OK:
        st.warning("Auth modülü yüklenemedi.")
        st.stop()

    st.markdown(
        section_header("Admin Panel", "Kullanıcı yönetimi ve sistem kontrolü"),
        unsafe_allow_html=True,
    )

    # Pending action işle
    if st.session_state.pending_action:
        action = st.session_state.pending_action
        try:
            if action["type"] == "approve":
                update_user_role(action["id"], action["role"])
                approve_user(action["id"], st.session_state.username)
                st.session_state.admin_msg = ("success", f"{action['username']} onaylandı! Rol: {action['role']}")
            elif action["type"] == "reject":
                reject_user(action["id"])
                st.session_state.admin_msg = ("warning", f"{action['username']} reddedildi.")
            elif action["type"] == "role_update":
                update_user_role(action["id"], action["role"])
                st.session_state.admin_msg = ("success", f"{action['username']} rol değiştirildi: {action['role']}")
            elif action["type"] == "delete":
                delete_user(action["id"])
                st.session_state.admin_msg = ("warning", f"{action['username']} silindi.")
        except Exception as e:
            st.session_state.admin_msg = ("warning", f"Hata: {e}")
        st.session_state.pending_action = None
        st.rerun()

    if st.session_state.admin_msg:
        msg_type, msg_text = st.session_state.admin_msg
        cl  = GREEN  if msg_type == "success" else ORANGE
        bg  = "rgba(0,227,150,.08)"  if msg_type == "success" else "rgba(255,107,53,.08)"
        brc = "rgba(0,227,150,.3)"   if msg_type == "success" else "rgba(255,107,53,.3)"
        st.markdown(
            f"<div style='background:{bg};border:1px solid {brc};border-left:4px solid {cl};"
            f"border-radius:10px;padding:12px 18px;margin-bottom:16px;"
            f"font-family:DM Mono;font-size:.82rem;color:{cl};'>{msg_text}</div>",
            unsafe_allow_html=True,
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

    mc = st.columns(5)
    for col, (lbl, val, c) in zip(mc, [
        ("Toplam",        f"{len(all_users):,}", GOLD),
        ("Aktif",         f"{len(active):,}",    GREEN),
        ("Bekleyen",      f"{len(pending):,}",   ORANGE),
        ("Reddedilen",    f"{len(rejected):,}",  RED),
        ("Toplam Giriş",  f"{login_stats['toplam']:,}", GOLD),
    ]):
        with col:
            st.markdown(hmetric(lbl, val, c), unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Tab seçimi (DÜZELTİLDİ: tek radio, state ile kontrol) ──────────────
    tab_labels = {"onay": "⏳ Bekleyen Onaylar", "tumu": "👥 Tüm Kullanıcılar", "sistem": "⚙️ Sistem & API"}
    tc1, tc2, tc3, _ = st.columns([1.4, 1.4, 1.2, 3])
    with tc1:
        if st.button("⏳ Bekleyen Onaylar", use_container_width=True, key="atab_onay"):
            st.session_state.admin_tab = "onay"
    with tc2:
        if st.button("👥 Tüm Kullanıcılar", use_container_width=True, key="atab_tumu"):
            st.session_state.admin_tab = "tumu"
    with tc3:
        if st.button("⚙️ Sistem & API", use_container_width=True, key="atab_sistem"):
            st.session_state.admin_tab = "sistem"

    active_tab = st.session_state.admin_tab
    st.markdown(
        f"<div style='font-family:DM Mono;font-size:.65rem;color:{GOLD};"
        f"padding:4px 0 8px;'>{tab_labels.get(active_tab,'')}</div>"
        f"<div style='height:1px;background:{T['border']};margin-bottom:20px;'></div>",
        unsafe_allow_html=True,
    )

    # ── TAB: ONAY ───────────────────────────────────────────────────────────
    if active_tab == "onay":
        if pending:
            for u in pending:
                pc1, pc2, pc3, pc4, pc5 = st.columns([2.2, 1.5, 1.2, 0.9, 0.9])
                with pc1:
                    st.markdown(
                        f"<div style='padding:8px 0;'>"
                        f"<div style='font-family:Syne;font-size:.88rem;font-weight:700;color:{T['text_primary']};'>{u['username']}</div>"
                        f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};'>{u['email']}</div></div>",
                        unsafe_allow_html=True,
                    )
                with pc2:
                    st.markdown(f"<div style='font-family:DM Mono;font-size:.6rem;color:{ORANGE};padding-top:12px;'>Bekliyor</div>", unsafe_allow_html=True)
                with pc3:
                    nr = st.selectbox("", ["viewer","analyst","admin"],
                        index=["viewer","analyst","admin"].index(u["role"]) if u["role"] in ["viewer","analyst","admin"] else 0,
                        key=f"rp_{u['id']}", label_visibility="collapsed")
                with pc4:
                    if st.button("✅ Onayla", key=f"ap_{u['id']}", use_container_width=True):
                        st.session_state.pending_action = {"type":"approve","id":u["id"],"role":nr,"username":u["username"]}
                        st.rerun()
                with pc5:
                    if st.button("❌ Reddet", key=f"rj_{u['id']}", use_container_width=True):
                        st.session_state.pending_action = {"type":"reject","id":u["id"],"username":u["username"]}
                        st.rerun()
                st.markdown(f"<div style='height:1px;background:{T['border']};margin:4px 0;'></div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background:rgba(0,227,150,.05);border:1px solid rgba(0,227,150,.18);"
                f"border-radius:12px;padding:30px;text-align:center;'>"
                f"<div style='font-size:2rem;margin-bottom:8px;'>✅</div>"
                f"<div style='font-family:Syne;font-size:.9rem;font-weight:700;color:{GREEN};'>"
                f"Onay bekleyen kullanıcı yok</div></div>",
                unsafe_allow_html=True,
            )

    # ── TAB: TÜM KULLANICILAR ───────────────────────────────────────────────
    elif active_tab == "tumu":
        for u in all_users:
            sc   = {"active":GREEN,"pending":ORANGE,"rejected":RED}.get(u["status"], T["text_muted"])
            sl   = {"active":"Aktif","pending":"Bekliyor","rejected":"Reddedildi"}.get(u["status"],"—")
            av   = {"admin":"👨‍💼","analyst":"📊","viewer":"👁️"}.get(u["role"],"👤")
            uc1, uc2, uc3, uc4, uc5, uc6, uc7 = st.columns([0.4, 2, 1.5, 1.2, 0.9, 1.2, 0.7])
            with uc1:
                st.markdown(f"<div style='font-size:1.2rem;padding-top:8px;text-align:center;'>{av}</div>", unsafe_allow_html=True)
            with uc2:
                st.markdown(
                    f"<div style='padding:5px 0;'>"
                    f"<div style='font-family:Syne;font-size:.82rem;font-weight:700;color:{T['text_primary']};'>{u['username']}</div>"
                    f"<div style='font-family:DM Mono;font-size:.58rem;color:{T['text_muted']};'>{u['email']}</div></div>",
                    unsafe_allow_html=True,
                )
            with uc3:
                st.markdown(
                    f"<div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};padding-top:9px;'>"
                    f"{u.get('display_name') or '—'}</div>",
                    unsafe_allow_html=True,
                )
            with uc4:
                nr2 = st.selectbox("", ["viewer","analyst","admin"],
                    index=["viewer","analyst","admin"].index(u["role"]) if u["role"] in ["viewer","analyst","admin"] else 0,
                    key=f"cr_{u['id']}", label_visibility="collapsed")
            with uc5:
                if nr2 != u["role"]:
                    if st.button("💾 Kaydet", key=f"sr_{u['id']}", use_container_width=True):
                        st.session_state.pending_action = {"type":"role_update","id":u["id"],"role":nr2,"username":u["username"]}
                        st.rerun()
            with uc6:
                st.markdown(
                    f"<div style='font-family:DM Mono;font-size:.62rem;padding-top:10px;'>"
                    f"<span style='color:{sc};'>{sl}</span></div>",
                    unsafe_allow_html=True,
                )
            with uc7:
                if u["username"] != st.session_state.username:
                    if st.button("🗑 Sil", key=f"dl_{u['id']}", use_container_width=True):
                        st.session_state.pending_action = {"type":"delete","id":u["id"],"username":u["username"]}
                        st.rerun()
            st.markdown(f"<div style='height:1px;background:{T['border']};margin:2px 0;'></div>", unsafe_allow_html=True)

    # ── TAB: SİSTEM & API ───────────────────────────────────────────────────
    elif active_tab == "sistem":

        # Şifre değiştir
        st.markdown(
            f"<div style='font-family:Syne;font-size:1rem;font-weight:700;color:{T['text_primary']};margin-bottom:14px;'>"
            f"🔑 Şifre Değiştir</div>",
            unsafe_allow_html=True,
        )
        pw1, pw2, pw3 = st.columns(3)
        with pw1: old_pw  = st.text_input("Mevcut Şifre",     type="password", key="old_pw")
        with pw2: new_pw  = st.text_input("Yeni Şifre",        type="password", key="new_pw")
        with pw3: new_pw2 = st.text_input("Yeni Şifre Tekrar", type="password", key="new_pw2")
        if st.button("Güncelle", key="pw_update_btn"):
            if new_pw != new_pw2:
                st.error("Şifreler eşleşmiyor!")
            else:
                try:
                    r = change_password(st.session_state.username, old_pw, new_pw)
                    if r["success"]: st.success(r["message"])
                    else:            st.error(r["message"])
                except Exception as e:
                    st.error(f"Hata: {e}")

        st.markdown(hr_line(), unsafe_allow_html=True)

        # Giriş istatistikleri
        ls1, ls2, ls3 = st.columns(3)
        with ls1: st.markdown(hmetric("Toplam Giriş",  f"{login_stats['toplam']:,}"),          unsafe_allow_html=True)
        with ls2: st.markdown(hmetric("Başarılı",      f"{login_stats['basarili']:,}",  GREEN), unsafe_allow_html=True)
        with ls3: st.markdown(hmetric("Başarısız",     f"{login_stats['basarisiz']:,}", RED),   unsafe_allow_html=True)

        st.markdown(hr_line(), unsafe_allow_html=True)

        # ── API Endpoint Testi ───────────────────────────────────────────────
        st.markdown(
            f"<div style='font-family:Syne;font-size:1rem;font-weight:700;color:{T['text_primary']};margin-bottom:6px;'>"
            f"🔌 API Endpoint Testi</div>"
            f"<div style='font-family:DM Mono;font-size:.62rem;color:{T['text_muted']};margin-bottom:16px;'>"
            f"API Durumu: <span style='color:{'#00E396' if API_ALIVE else '#FF4560'};'>"
            f"{'● Çevrimiçi' if API_ALIVE else '● Çevrimdışı — uvicorn src.api:app --port 8000'}</span></div>",
            unsafe_allow_html=True,
        )

        ENDPOINTS = {
            "/health":                {"auth": False, "params": None,         "desc": "Sistem sağlık kontrolü"},
            "/stats":                 {"auth": True,  "params": None,         "desc": "Genel istatistikler"},
            "/stats/fraud":           {"auth": True,  "params": None,         "desc": "Fraud istatistikleri"},
            "/model/metrics":         {"auth": True,  "params": None,         "desc": "Model metrikleri"},
            "/model/confusion-matrix":{"auth": True,  "params": None,         "desc": "Confusion matrix"},
            "/clients/top-risk":      {"auth": True,  "params": {"limit": 5}, "desc": "En riskli müşteriler (top 5)"},
        }

        ep_col, btn_col = st.columns([3, 1])
        with ep_col:
            sel_ep = st.selectbox(
                "Endpoint Seç",
                list(ENDPOINTS.keys()),
                format_func=lambda k: f"{k}  —  {ENDPOINTS[k]['desc']}",
                key="admin_ep_sel",
            )
        with btn_col:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_test = st.button("▶ Test Et", use_container_width=True, key="run_ep_test")

        if run_test:
            ep_info     = ENDPOINTS[sel_ep]
            status_code, resp_data = api_get_raw(sel_ep, params=ep_info["params"])

            if status_code == 200:
                st.markdown(
                    f"<div style='background:rgba(0,227,150,.06);border:1px solid rgba(0,227,150,.25);"
                    f"border-left:4px solid {GREEN};border-radius:10px;padding:10px 16px;margin-bottom:10px;'>"
                    f"<span style='font-family:DM Mono;font-size:.72rem;color:{GREEN};'>"
                    f"✅ {sel_ep} — HTTP {status_code} OK</span></div>",
                    unsafe_allow_html=True,
                )
                if isinstance(resp_data, list):
                    st.json(resp_data[:2])
                else:
                    st.json(resp_data)

            elif status_code == 401:
                st.markdown(
                    f"<div style='background:rgba(255,107,53,.06);border:1px solid rgba(255,107,53,.25);"
                    f"border-left:4px solid {ORANGE};border-radius:10px;padding:10px 16px;'>"
                    f"<div style='font-family:DM Mono;font-size:.72rem;color:{ORANGE};'>⚠️ HTTP 401 — Token gerekli</div>"
                    f"<div style='font-family:DM Mono;font-size:.65rem;color:{T['text_muted']};margin-top:4px;'>"
                    f"Logout yapıp tekrar giriş yaparsanız token otomatik alınır.</div></div>",
                    unsafe_allow_html=True,
                )

            elif status_code == 0:
                err_msg = resp_data.get("error", "—")
                st.markdown(
                    f"<div style='background:rgba(255,69,96,.06);border:1px solid rgba(255,69,96,.25);"
                    f"border-left:4px solid {RED};border-radius:10px;padding:10px 16px;'>"
                    f"<div style='font-family:DM Mono;font-size:.72rem;color:{RED};'>🔴 Bağlantı Hatası — API çevrimdışı</div>"
                    f"<div style='font-family:DM Mono;font-size:.65rem;color:{T['text_muted']};margin-top:4px;'>"
                    f"Başlatmak için: <code>uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload</code></div>"
                    f"<div style='font-family:DM Mono;font-size:.65rem;color:{T['text_muted']};margin-top:2px;'>"
                    f"Hata: {err_msg}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background:rgba(255,69,96,.06);border:1px solid rgba(255,69,96,.25);"
                    f"border-left:4px solid {RED};border-radius:10px;padding:10px 16px;'>"
                    f"<div style='font-family:DM Mono;font-size:.72rem;color:{RED};'>"
                    f"❌ HTTP {status_code} — {sel_ep}</div></div>",
                    unsafe_allow_html=True,
                )
                if resp_data:
                    st.json(resp_data)

        st.markdown(hr_line(), unsafe_allow_html=True)

        # ── Hızlı Metrik Okuma ───────────────────────────────────────────────
        st.markdown(
            f"<div style='font-family:Syne;font-size:1rem;font-weight:700;color:{T['text_primary']};margin-bottom:14px;'>"
            f"📊 Hızlı Metrik Okuma</div>",
            unsafe_allow_html=True,
        )
        q1, q2, q3 = st.columns(3)
        with q1:
            if st.button("📊 /stats", use_container_width=True, key="quick_stats"):
                stats_data = api_get("/stats")
                if stats_data:
                    st.json(stats_data)
                else:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        dfs = pd.read_sql("""
                            SELECT COUNT(*) as toplam,
                                   SUM(CASE WHEN fraud_tahmini='Yuksek Risk' THEN 1 ELSE 0 END) as yuksek_risk,
                                   SUM(CASE WHEN fraud_tahmini='Supheli' THEN 1 ELSE 0 END) as supheli,
                                   AVG(fraud_skoru) as ort_fraud_skoru
                            FROM client_ml
                        """, conn)
                        conn.close()
                        st.json(dfs.iloc[0].to_dict())
                    except Exception as e:
                        st.error(f"DB hatası: {e}")
        with q2:
            if st.button("⚠️ /stats/fraud", use_container_width=True, key="quick_fraud"):
                fd = api_get("/stats/fraud")
                if fd:
                    st.json(fd[:3] if isinstance(fd, list) else fd)
                else:
                    try:
                        conn = sqlite3.connect(DB_PATH)
                        dff = pd.read_sql("""
                            SELECT fraud_tahmini,
                                   COUNT(*) as musteri_sayisi,
                                   AVG(fraud_skoru) as ort_skor
                            FROM client_ml
                            GROUP BY fraud_tahmini
                        """, conn)
                        conn.close()
                        st.dataframe(dff, use_container_width=True)
                    except Exception as e:
                        st.error(f"DB hatası: {e}")
        with q3:
            if st.button("🤖 Model Metrics", use_container_width=True, key="quick_metrics"):
                mm = api_get("/model/metrics")
                show_keys = ["auc_roc","f1_skoru","precision","recall","accuracy","en_iyi_model","egitim_tarihi"]
                if mm:
                    st.json({k: mm[k] for k in show_keys if k in mm})
                else:
                    ml_loc = load_model_metrics()
                    if ml_loc:
                        st.json({k: ml_loc[k] for k in show_keys[:5] if k in ml_loc})
                    else:
                        st.warning("model_metrics.json bulunamadı — `python src/ml_model.py` çalıştırın.")

        st.markdown(hr_line(), unsafe_allow_html=True)

        # ── ML Özet + İndirme ────────────────────────────────────────────────
        ml_ozet_s = load_ml_ozet()
        if ml_ozet_s:
            st.markdown(
                f"<div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);"
                f"border-radius:12px;padding:18px 22px;margin-bottom:16px;'>"
                f"<div style='font-family:DM Mono;font-size:.6rem;color:{T['text_muted']};"
                f"text-transform:uppercase;margin-bottom:10px;'>Son ML Çalışması</div>"
                f"<div style='font-family:DM Mono;font-size:.72rem;color:{T['text_secondary']};line-height:2;'>"
                f"Hesaplama: <span style='color:{CYAN};'>{str(ml_ozet_s.get('hesaplama_tarihi',''))[:16]}</span> · "
                f"Toplam: <span style='color:{GOLD};'>{ml_ozet_s.get('toplam',0):,}</span> · "
                f"Yüksek Risk: <span style='color:{RED};'>{ml_ozet_s.get('yuksek_risk',0):,}</span> · "
                f"Pipeline: <span style='color:{GREEN};'>v{ml_ozet_s.get('pipeline_version','—')}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button(
                "⬇️ Müşteri Verisi",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="musteri.csv", mime="text/csv",
                use_container_width=True, key="dl_musteri",
            )
        with dl2:
            metrics_loc = load_model_metrics()
            if metrics_loc:
                st.download_button(
                    "⬇️ Model Metrikleri",
                    data=json.dumps(metrics_loc, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="model_metrics.json", mime="application/json",
                    use_container_width=True, key="dl_metrics",
                )
        with dl3:
            if st.button("🔄 Cache Temizle", use_container_width=True, key="cache_clear"):
                st.cache_data.clear()
                st.success("Cache temizlendi!")