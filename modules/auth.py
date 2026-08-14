"""
Authentication Module for Asmeranda
Handles user login, registration, OTP verification, and logout
"""

import streamlit as st
import os
import smtplib
import ssl
from email.message import EmailMessage
from auth_db import auth_db
from captcha_utils import captcha_gen, verify_captcha


def safe_rerun():
    """Compatibility helper for rerun across Streamlit versions"""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def send_otp_email(recipient_email: str, otp_code: str) -> bool:
    """Send OTP code to recipient email using SMTP settings from environment. Returns True if sent."""
    cfg_db = auth_db.get_smtp_config()
    smtp_host = cfg_db.get('host') or os.getenv('SMTP_HOST')
    smtp_port = int(cfg_db.get('port') or os.getenv('SMTP_PORT', '587'))
    smtp_user = cfg_db.get('user') or os.getenv('SMTP_USER')
    smtp_pass = cfg_db.get('password') or os.getenv('SMTP_PASS')
    use_tls = bool(cfg_db.get('tls')) if 'tls' in cfg_db else (os.getenv('SMTP_TLS', 'true').lower() in ['1', 'true', 'yes'])
    sender_email = cfg_db.get('sender') or os.getenv('SMTP_SENDER', smtp_user or 'no-reply@localhost')
    
    if not smtp_host or not smtp_user or not smtp_pass:
        return False
    
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Your OTP Code'
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content(f'Kode OTP Anda: {otp_code}\nKode berlaku selama 10 menit.')
        
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        return True
    except Exception:
        return False


def render_loginizer():
    """Render landing page with login/register card. Returns True if authenticated."""
    # Add premium CSS - Light Theme (White, Green, Gold)
    premium_css = """
    <style>
    /* Main Theme Colors - Premium Light/Luxury (White, Green, Gold) */
    :root {
        --primary-green: #0d5c45;
        --primary-green-light: #146c43;
        --accent-gold: #d4af37;
        --accent-gold-light: #e6c057;
        --bg-white: #ffffff;
        --bg-soft: #f8f9fa;
        --bg-card: #ffffff;
        --text-primary: #212529;
        --text-secondary: #495057;
        --text-muted: #6c757d;
        --success-color: #198754;
        --warning-color: #ffc107;
        --error-color: #dc3545;
    }

    /* Global Styles */
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: var(--text-primary);
    }

    /* Header and Titles */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary-green), var(--accent-gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Landing Page Specific */
    .app-header-banner {
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    .app-header-banner h1 {
        font-size: clamp(1.8rem, 4vw, 2.5rem) !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
        line-height: 1.2 !important;
    }
    .app-header-banner p {
        font-size: clamp(0.9rem, 2vw, 1.05rem);
        color: var(--text-secondary);
        margin-bottom: 0;
        line-height: 1.5;
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.03);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
        height: 100%;
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.06);
        background: rgba(255, 255, 255, 0.9);
        border-color: rgba(255, 255, 255, 1);
    }
    .feature-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(13, 92, 69, 0.08), rgba(212, 175, 55, 0.15));
        margin-bottom: 0.8rem;
    }
    .feature-icon svg {
        width: 24px;
        height: 24px;
        stroke: var(--primary-green);
        stroke-width: 1.5px;
    }
    /* Override gradient text for feature-box and promo headings */
    .feature-box-title {
        margin-top: 0 !important;
        font-size: 1.2rem !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #0d5c45 !important;
        background-clip: unset !important;
        color: #0d5c45 !important;
        font-weight: 700;
    }
    .promo-title {
        margin-top: 0 !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        -webkit-text-fill-color: #0d5c45 !important;
        background-clip: unset !important;
        color: #0d5c45 !important;
        font-weight: 700;
    }
    .captcha-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.4rem;
        margin-bottom: 0.75rem;
    }
    .captcha-wrapper img {
        border: 2px solid rgba(13, 92, 69, 0.25);
        border-radius: 10px;
        background: #f8f9fa;
        padding: 6px;
        max-width: 240px;
        height: auto;
        display: block;
    }
    .login-card-container {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-top: 5px solid var(--primary-green);
        margin-top: 1rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    /* Button Styling */
    button {
        background: linear-gradient(135deg, var(--primary-green), var(--primary-green-light)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 15px rgba(13, 92, 69, 0.25) !important;
        position: relative;
        overflow: hidden;
    }

    button::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0) 100%);
        transform: skewX(-25deg);
        transition: left 0.6s ease;
    }

    button:hover::after {
        left: 150%;
    }

    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(13, 92, 69, 0.35) !important;
    }

    /* Tab Styling */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }

    [data-baseweb="tab"] {
        background: var(--bg-soft);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(212, 175, 55, 0.25);
        color: var(--text-secondary);
        font-weight: 500;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-green), var(--primary-green-light));
        color: white;
        border: 1px solid var(--primary-green);
        font-weight: 600;
    }

    /* Input Fields */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select,
    [data-testid="stTextarea"] textarea {
        background: var(--bg-white);
        border: 1px solid rgba(13, 92, 69, 0.2);
        color: var(--text-primary);
        border-radius: 10px;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] select:focus,
    [data-testid="stTextarea"] textarea:focus {
        border-color: var(--accent-gold);
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.15);
    }
    </style>
    """
    st.markdown(premium_css, unsafe_allow_html=True)
    
    if 'auth_step' not in st.session_state:
        st.session_state.auth_step = 'login'
    if 'captcha_text' not in st.session_state:
        _, captcha_text = captcha_gen.get_captcha_base64(mode='math')
        st.session_state.captcha_text = captcha_text
    
    # ---------------------------------------------------------
    # Layout: Compact Header → Login Card (centered) → Feature Grid
    # ---------------------------------------------------------

    # 1. Compact app header banner
    header_html = (
        '<div class="app-header-banner">'
        '<div style="width: 50px; height: 4px; background: linear-gradient(90deg, #0d5c45, #d4af37); border-radius: 2px; margin: 0 auto 0.75rem;"></div>'
        '<h1>ASMERANDA AI</h1>'
        '<p>Platform Machine Learning Cerdas untuk Analisis Data dan Prediksi Berbasis Explainable AI.</p>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # 2. Login card — centered with max-width
    _, col_login, _ = st.columns([0.15, 0.7, 0.15])
    
    with col_login:
        st.markdown('<div class="login-card-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; margin-top: 0;'>Portal Pengguna</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        
        tabs = st.tabs(["Masuk", "Daftar", "OTP"])
        
        # Login Tab
        with tabs[0]:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            img_b64_login, _ = captcha_gen.get_captcha_base64(st.session_state.captcha_text, mode='math')
            captcha_html_login = (
                "<div class='captcha-wrapper'>"
                "<p style='font-size: 0.9rem; margin: 0 0 0.3rem 0; color: #495057; font-weight: 500;'>Selesaikan hitungan berikut:</p>"
                "<img src='" + img_b64_login + "' alt='captcha'"
                " style='border: 2px solid rgba(13,92,69,0.25); border-radius: 10px; background: #f8f9fa; padding: 6px; width: 220px; height: auto; display: block;'>"
                "</div>"
            )
            st.markdown(captcha_html_login, unsafe_allow_html=True)
            login_captcha = st.text_input("Jawaban hitungan", key="login_captcha")
            
            if st.button("Masuk & Kirim OTP", type="primary", key="btn_login", use_container_width=True):
                if not verify_captcha(login_captcha, st.session_state.captcha_text, case_sensitive=True):
                    st.error("Hitungan salah. Silakan coba lagi.")
                    _, st.session_state.captcha_text = captcha_gen.get_captcha_base64(mode='math')
                else:
                    user_data = auth_db.authenticate_user(login_username, login_password)
                    
                    if user_data and 'error' in user_data:
                        if user_data['error'] == 'locked':
                            wait_time = user_data['locked_until']
                            st.error(f"Akun terkunci. Coba lagi setelah {wait_time.strftime('%H:%M:%S')}")
                    elif user_data:
                        st.session_state.temp_user_data = user_data
                        otp_info = auth_db.generate_otp(login_username)
                        if otp_info:
                            sent = send_otp_email(otp_info['email'], otp_info['code'])
                            st.session_state.auth_pending_user = login_username
                            st.session_state.auth_step = 'verify'
                            if sent:
                                st.success("OTP telah dikirim ke email terdaftar. Buka tab OTP.")
                            else:
                                st.warning("Mode demo: OTP ditampilkan karena email belum terkonfigurasi.")
                                st.info(f"Kode OTP: {otp_info['code']}")
                        else:
                            st.error("Email pengguna tidak ditemukan.")
                    else:
                        st.error("Username atau password salah.")
        
        # Register Tab
        with tabs[1]:
            reg_username = st.text_input("Username Baru", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            
            if 'generated_password' not in st.session_state:
                st.session_state.generated_password = ""
            
            col_pass, col_gen = st.columns([0.7, 0.3])
            with col_gen:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔑 Buat", key="btn_gen_pass", help="Buat sandi kuat secara otomatis"):
                    st.session_state.generated_password = auth_db.generate_strong_password()
                    st.info(f"`{st.session_state.generated_password}`")
            
            with col_pass:
                reg_password = st.text_input(
                    "Password", 
                    type="password", 
                    key="reg_password",
                    value=st.session_state.generated_password if st.session_state.generated_password else ""
                )
            
            if reg_password:
                score, label, feedback = auth_db.check_password_strength(reg_password)
                strength_colors = {
                    "Sangat Lemah": "🔴", "Lemah": "🟠", "Sedang": "🟡", "Kuat": "🟢", "Sangat Kuat": "✅"
                }
                st.markdown(f"Kekuatan: **{strength_colors.get(label, '')} {label}**")
                st.progress(min(score / 4.0, 1.0))
                
                if feedback:
                    with st.expander("Saran Perbaikan Sandi"):
                        for msg in feedback:
                            st.write(f"- {msg}")
            
            reg_password2 = st.text_input("Konfirmasi Password", type="password", key="reg_password2")
            
            img_b64, _ = captcha_gen.get_captcha_base64(st.session_state.captcha_text, mode='math')
            captcha_html_reg = (
                "<div class='captcha-wrapper'>"
                "<p style='font-size: 0.9rem; margin: 0 0 0.3rem 0; color: #495057; font-weight: 500;'>Selesaikan hitungan berikut:</p>"
                "<img src='" + img_b64 + "' alt='captcha'"
                " style='border: 2px solid rgba(13,92,69,0.25); border-radius: 10px; background: #f8f9fa; padding: 6px; width: 220px; height: auto; display: block;'>"
                "</div>"
            )
            st.markdown(captcha_html_reg, unsafe_allow_html=True)
            captcha_input = st.text_input("Jawaban hitungan", key="captcha_input")
            
            if st.button("Daftar Sekarang", type="primary", key="btn_register", use_container_width=True):
                score, label, _ = auth_db.check_password_strength(reg_password)
                
                if reg_password != reg_password2:
                    st.error("Password dan konfirmasi tidak cocok.")
                elif score < 3:
                    st.error(f"Sandi terlalu lemah ({label}). Harap gunakan sandi yang lebih kuat.")
                elif not verify_captcha(captcha_input, st.session_state.captcha_text, case_sensitive=True):
                    st.error("Hitungan salah. Silakan coba lagi.")
                    _, st.session_state.captcha_text = captcha_gen.get_captcha_base64(mode='math')
                elif not reg_email or '@' not in reg_email:
                    st.error("Email tidak valid.")
                else:
                    if not auth_db.is_username_available(reg_username):
                        st.error("Username sudah digunakan.")
                    elif auth_db.get_user_by_email(reg_email):
                        st.error("Email sudah terdaftar.")
                    else:
                        created = auth_db.create_user(reg_username, reg_password, reg_email)
                        if created:
                            st.success("Akun berhasil dibuat. Silakan masuk di tab Masuk.")
                            st.session_state.auth_step = 'login'
                        else:
                            st.error("Pembuatan akun gagal. Periksa input Anda.")
        
        # Verify Tab
        with tabs[2]:
            pending_user = st.session_state.get('auth_pending_user')
            if pending_user:
                st.info(f"Verifikasi untuk: {pending_user}")
            otp_input = st.text_input("Masukkan Kode OTP", key="otp_input")
            
            if st.button("Verifikasi OTP", type="primary", key="btn_verify", use_container_width=True):
                user = pending_user
                if not user:
                    st.error("Tidak ada sesi OTP yang aktif. Silakan masuk terlebih dahulu.")
                else:
                    ok = auth_db.verify_otp(user, otp_input)
                    if ok:
                        token = auth_db.create_session(user)
                        st.session_state.session_token = token
                        st.session_state.authenticated = True
                        st.session_state.current_username = user
                        
                        user_details = auth_db.get_user_by_username(user)
                        if user_details and user_details.get('trial_ends_at'):
                            st.session_state.trial_ends_at = user_details['trial_ends_at']
                        
                        auth_db.record_activity(user, 'login')
                        st.success("Verifikasi berhasil. Mengalihkan...")
                        safe_rerun()
                    else:
                        st.error("Kode OTP salah atau kedaluwarsa.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Feature highlights grid below login card
    st.markdown("<br>", unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown(
            '<div class="feature-box">'
            '<div class="feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg></div>'
            '<h3 class="feature-box-title">Exploratory Data Analysis</h3>'
            '<p style="margin-bottom: 0; color: #495057; font-size: 0.88rem;">Analisis distribusi &amp; korelasi data sebelum pemodelan.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    with col_f2:
        st.markdown(
            '<div class="feature-box">'
            '<div class="feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg></div>'
            '<h3 class="feature-box-title">Machine Learning &amp; XAI</h3>'
            '<p style="margin-bottom: 0; color: #495057; font-size: 0.88rem;">Model Supervised/Unsupervised dengan interpretasi SHAP &amp; LIME.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    with col_f3:
        st.markdown(
            '<div class="feature-box">'
            '<div class="feature-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg></div>'
            '<h3 class="feature-box-title">Time Series Analysis</h3>'
            '<p style="margin-bottom: 0; color: #495057; font-size: 0.88rem;">Deteksi anomali &amp; prediksi data deret waktu.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # 4. Promo banner full-width
    st.markdown("<br>", unsafe_allow_html=True)
    promo_html = (
        '<div style="padding: 1.25rem 1.5rem; background: linear-gradient(135deg, rgba(212,175,55,0.1), rgba(13,92,69,0.05));'
        ' border-radius: 12px; border-left: 4px solid #d4af37; display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap;">'
        '<div style="flex: 1; min-width: 200px;">'
        '<h4 class="promo-title" style="margin-bottom: 0.4rem;">\U0001f31f Penawaran Membership Premium</h4>'
        '<p style="font-size: 0.9rem; margin-bottom: 0;">Akses penuh + pendampingan WA group + konsultasi desain penelitian hanya <strong>Rp 1.185.000</strong>.</p>'
        '</div>'
        '<a href="https://wa.me/6281238742321" target="_blank"'
        ' style="display: inline-block; background: #25D366; color: white; padding: 10px 20px; border-radius: 8px; text-decoration: none; font-weight: bold; white-space: nowrap; box-shadow: 0 4px 10px rgba(37,211,102,0.3);">'
        '\U0001f4ac Hubungi Admin'
        '</a>'
        '</div>'
    )
    st.markdown(promo_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    return st.session_state.get('authenticated', False)


def logout_user():
    """Logout function to clear session and record activity"""
    current_user = st.session_state.get('current_username')
    if current_user:
        try:
            auth_db.record_activity(current_user, 'logout')
        except Exception:
            pass
    
    keys_to_clear = ['authenticated', 'current_username', 'session_token', 'trial_ends_at']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    if st.session_state.get('language') == 'id':
        st.success("Anda telah berhasil logout. Sampai jumpa!")
    else:
        st.success("You have been successfully logged out. Goodbye!")
    
    safe_rerun()


def check_trial_period():
    """Check and handle trial period for non-admin users"""
    from datetime import datetime
    
    try:
        current_user = st.session_state.get('current_username')
        if current_user:
            auth_db.record_activity(current_user, 'access_app')
            
            if not auth_db.is_super_admin(current_user):
                if 'trial_ends_at' not in st.session_state:
                    user_info = auth_db.get_user_by_username(current_user)
                    if user_info and user_info.get('trial_ends_at'):
                        st.session_state.trial_ends_at = user_info['trial_ends_at']
                
                trial_end_str = st.session_state.get('trial_ends_at')
                if trial_end_str:
                    trial_end = datetime.fromisoformat(trial_end_str)
                    now = datetime.now()
                    
                    if now > trial_end:
                        st.error("⚠️ Masa percobaan gratis 30 hari Anda telah berakhir.")
                        st.info("Silakan hubungi administrator untuk memperpanjang akses.")
                        if st.button("Keluar"):
                            st.session_state.clear()
                            safe_rerun()
                        st.stop()
                    else:
                        days_left = (trial_end - now).days
                        st.sidebar.info(f"⏳ Masa Trial: {days_left} hari tersisa")
    except Exception:
        pass


def log_feature(feature_name: str):
    """Helper to record feature usage"""
    try:
        user = st.session_state.get('current_username')
        if user:
            auth_db.record_feature_usage(user, feature_name)
    except Exception:
        pass
