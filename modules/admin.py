"""
Admin Dashboard Module for Asmeranda
Super Admin Dashboard functionality
"""

import streamlit as st
import pandas as pd
from auth_db import auth_db


def render_admin_dashboard():
    """Render Super Admin Dashboard"""
    current_user = st.session_state.get('current_username')
    if not current_user or not auth_db.is_super_admin(current_user):
        return
    
    admin_tabs = st.tabs(["🛡️ Super Admin Dashboard"])
    
    with admin_tabs[0]:
        st.subheader("🛡️ Super Admin Dashboard")
        
        # Users metrics
        users = pd.DataFrame(auth_db.get_users_dataframe())
        total_users = len(users)
        active_users = int(users['is_active'].sum()) if total_users > 0 else 0
        inactive_users = total_users - active_users
        super_admins = int(users['is_super_admin'].sum()) if total_users > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pengguna", total_users)
        c2.metric("Aktif", active_users)
        c3.metric("Non-aktif", inactive_users)
        c4.metric("Super Admin", super_admins)
        
        st.markdown("---")
        st.markdown("### 👥 Data Pengguna Terdaftar")
        
        if total_users > 0:
            st.dataframe(
                users[['id', 'username', 'email', 'is_active', 'is_super_admin', 'created_at', 'last_login']], 
                use_container_width=True
            )
        else:
            st.info("Belum ada pengguna terdaftar.")
        
        # Activity summary
        st.markdown("### 📈 Ringkasan Aktivitas")
        activity = pd.DataFrame(auth_db.get_activity_summary())
        if len(activity) > 0:
            st.bar_chart(activity.set_index('action')['total'])
            st.table(activity)
        else:
            st.info("Belum ada aktivitas yang tercatat.")
        
        # Feature usage
        st.markdown("### ⭐ Fitur Paling Sering Digunakan")
        feature_stats = pd.DataFrame(auth_db.get_feature_usage_stats())
        if len(feature_stats) > 0:
            st.bar_chart(feature_stats.set_index('feature_name')['total'])
            st.table(feature_stats)
        else:
            st.info("Belum ada penggunaan fitur yang tercatat.")
        
        # SMTP Configuration
        st.markdown("---")
        st.markdown("### ⚙️ Konfigurasi SMTP")
        cfg = auth_db.get_smtp_config()
        host = st.text_input("SMTP Host", value=cfg.get('host', ''))
        port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=int(cfg.get('port', 587)))
        user = st.text_input("SMTP User", value=cfg.get('user', ''))
        password = st.text_input("SMTP Password", type="password", value=cfg.get('password', ''))
        sender = st.text_input("Sender Email", value=cfg.get('sender', user or ''))
        tls = st.checkbox("Gunakan TLS", value=bool(cfg.get('tls', True)))
        
        if st.button("Simpan Konfigurasi SMTP", key="admin_save_smtp"):
            cfg_save = {
                'host': host.strip(),
                'port': int(port),
                'user': user.strip(),
                'password': password,
                'sender': sender.strip() or user.strip(),
                'tls': tls
            }
            auth_db.set_smtp_config(cfg_save)
            auth_db.record_activity(current_user, 'update_smtp')
            st.success("Konfigurasi SMTP disimpan.")
        
        # Password Change
        st.markdown("---")
        st.markdown("### 🔒 Ganti Kata Sandi")
        old_pw = st.text_input("Kata sandi saat ini", type="password", key="admin_old_pw")
        
        if 'admin_gen_password' not in st.session_state:
            st.session_state.admin_gen_password = ""
        
        col_pass, col_gen = st.columns([0.7, 0.3])
        with col_gen:
            if st.button("🔑 Buat Sandi Kuat", key="btn_gen_admin_pw"):
                st.session_state.admin_gen_password = auth_db.generate_strong_password()
                st.info(f"Sandi dibuat: `{st.session_state.admin_gen_password}`")
        
        with col_pass:
            new_pw = st.text_input(
                "Kata sandi baru",
                type="password",
                key="admin_new_pw",
                value=st.session_state.admin_gen_password if st.session_state.admin_gen_password else ""
            )
        
        if new_pw:
            score, label, feedback = auth_db.check_password_strength(new_pw)
            strength_colors = {
                "Sangat Lemah": "🔴", "Lemah": "🟠", "Sedang": "🟡", "Kuat": "🟢", "Sangat Kuat": "✅"
            }
            st.markdown(f"Kekuatan Sandi: **{strength_colors.get(label, '')} {label}**")
            st.progress(min(score / 4.0, 1.0))
            
            if feedback:
                with st.expander("Saran Perbaikan Sandi"):
                    for msg in feedback:
                        st.write(f"- {msg}")
        
        new_pw2 = st.text_input("Konfirmasi kata sandi baru", type="password", key="admin_new_pw2")
        
        if st.button("Ganti Kata Sandi", key="admin_change_pw"):
            score, _, _ = auth_db.check_password_strength(new_pw)
            if new_pw != new_pw2:
                st.error("Konfirmasi kata sandi tidak cocok.")
            elif score < 3:
                st.error(f"Sandi terlalu lemah ({label}). Harap gunakan sandi yang lebih kuat.")
            else:
                if auth_db.change_password(current_user, old_pw, new_pw):
                    auth_db.record_activity(current_user, 'change_password')
                    st.success("Kata sandi berhasil diubah.")
                else:
                    st.error("Kata sandi saat ini salah.")
        
        # User Management
        st.markdown("---")
        st.markdown("### 👤 Manajemen Pengguna")
        st.markdown("#### Tambah Pengguna Baru")
        add_user_username = st.text_input("Username", key="add_user_username")
        add_user_email = st.text_input("Email", key="add_user_email")
        
        if 'add_user_gen_password' not in st.session_state:
            st.session_state.add_user_gen_password = ""
        
        col_add_pass, col_add_gen = st.columns([0.7, 0.3])
        with col_add_gen:
            if st.button("🔑 Buat Sandi", key="btn_gen_add_user_pw"):
                st.session_state.add_user_gen_password = auth_db.generate_strong_password()
                st.info(f"Sandi dibuat: `{st.session_state.add_user_gen_password}`")
        
        with col_add_pass:
            add_user_password = st.text_input(
                "Password",
                type="password",
                key="add_user_password",
                value=st.session_state.add_user_gen_password if st.session_state.add_user_gen_password else ""
            )
        
        add_user_is_admin = st.checkbox("Jadikan Super Admin", key="add_user_is_admin")
        
        if st.button("Tambah Pengguna", key="btn_add_user"):
            score, _, _ = auth_db.check_password_strength(add_user_password)
            if not add_user_username or not add_user_email or not add_user_password:
                st.error("Semua field harus diisi.")
            elif score < 3:
                st.error("Sandi terlalu lemah. Gunakan sandi yang lebih kuat.")
            else:
                created = auth_db.create_user(add_user_username, add_user_password, add_user_email)
                if created:
                    if add_user_is_admin:
                        auth_db.set_user_super_admin(add_user_username, True)
                    auth_db.record_activity(current_user, 'add_user', metadata=add_user_username)
                    st.success("Pengguna berhasil ditambahkan.")
                else:
                    st.error("Gagal menambahkan pengguna (username atau email mungkin sudah digunakan).")
        
        # Delete User
        st.markdown("#### Hapus Pengguna")
        user_list = users['username'].tolist() if total_users > 0 else []
        del_user = st.selectbox("Pilih pengguna untuk dihapus", options=user_list, key="del_user_select")
        
        if st.button("Hapus Pengguna", key="btn_delete_user"):
            if del_user == current_user:
                st.error("Tidak dapat menghapus diri sendiri.")
            else:
                # Prevent deleting last super admin
                if auth_db.is_super_admin(del_user):
                    if int(users['is_super_admin'].sum()) <= 1:
                        st.error("Tidak dapat menghapus satu-satunya super admin.")
                    else:
                        auth_db.delete_user(del_user)
                        auth_db.record_activity(current_user, 'delete_user', metadata=del_user)
                        st.success(f"Pengguna {del_user} telah dihapus.")
                        st.rerun()
                else:
                    auth_db.delete_user(del_user)
                    auth_db.record_activity(current_user, 'delete_user', metadata=del_user)
                    st.success(f"Pengguna {del_user} telah dihapus.")
                    st.rerun()
