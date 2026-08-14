import sqlite3
import bcrypt
import secrets
import string
import re
from datetime import datetime, timedelta
import os
from db_pool import get_pool

class AuthDatabase:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self._pool = get_pool(db_path, pool_size=5)
        self.init_database()
    
    def init_database(self):
        """Initialize the database with users table"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    is_super_admin BOOLEAN DEFAULT 0,
                    trial_ends_at TIMESTAMP,
                    password_reset_required INTEGER DEFAULT 0
                )
            ''')
            # Ensure email is unique (create index) - handle existing duplicates gracefully
            try:
                cursor.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
                ''')
            except sqlite3.IntegrityError:
                # Normalize duplicates then retry
                self.normalize_duplicate_emails()
                with self._pool.get_connection() as conn2:
                    cursor2 = conn2.cursor()
                    cursor2.execute('''
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
                    ''')
                    conn2.commit()
            
            # Create sessions table for tracking active sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            ''')
            
            # Activity logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    action TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Feature usage aggregate
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    username TEXT,
                    use_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_usage ON feature_usage(feature_name, username)
            ''')
            
            # App settings (SMTP configuration)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS app_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    smtp_host TEXT,
                    smtp_port INTEGER,
                    smtp_user TEXT,
                    smtp_pass TEXT,
                    smtp_sender TEXT,
                    smtp_tls INTEGER DEFAULT 1
                )
            ''')
            
            # Create OTP table for email verification
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    email TEXT NOT NULL,
                    otp_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            ''')
            
            conn.commit()
        
        # Migrate schema to ensure required columns exist on existing databases
        try:
            self.migrate_schema()
        except Exception:
            pass
        
        # Create default super admin if missing
        self.ensure_super_admin_exists()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt"""
        # Ensure password is bytes
        password_bytes = password.encode('utf-8')
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored bcrypt hash"""
        try:
            password_bytes = password.encode('utf-8')
            stored_hash_bytes = stored_hash.encode('utf-8')
            return bcrypt.checkpw(password_bytes, stored_hash_bytes)
        except Exception:
            return False
    
    def _sanitize_input(self, value: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(value, str):
            return str(value)
        # Remove null bytes and limit length
        sanitized = value.replace('\x00', '').strip()
        # Limit to reasonable length
        return sanitized[:255]
    
    def check_password_strength(self, password):
        """
        Check password strength and return score (0-4) and feedback message.
        """
        if not password:
            return 0, "Password kosong"
            
        score = 0
        feedback = []
        
        # Length check
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("Minimal 8 karakter")
            
        # Uppercase check
        if any(c.isupper() for c in password):
            score += 1
        else:
            feedback.append("Gunakan huruf kapital (A-Z)")
            
        # Lowercase check
        if any(c.islower() for c in password):
            score += 1
        else:
            feedback.append("Gunakan huruf kecil (a-z)")
            
        # Digit and Special char check
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in string.punctuation for c in password)
        
        if has_digit and has_special:
            score += 1
        elif has_digit:
            feedback.append("Tambahkan simbol/karakter khusus")
            score += 0.5
        elif has_special:
            feedback.append("Tambahkan angka (0-9)")
            score += 0.5
        else:
            feedback.append("Gunakan angka dan simbol")

        # Map score to labels
        if score < 2:
            label = "Sangat Lemah" if score == 0 else "Lemah"
        elif score < 3:
            label = "Sedang"
        elif score < 4:
            label = "Kuat"
        else:
            label = "Sangat Kuat"
            
        return score, label, feedback

    def generate_strong_password(self, length=14):
        """Generate a random strong password"""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        # Ensure at least one of each required type
        password = [
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.digits),
            secrets.choice(string.punctuation)
        ]
        # Fill the rest
        password += [secrets.choice(alphabet) for _ in range(length - 4)]
        # Shuffle to randomize positions
        secrets.SystemRandom().shuffle(password)
        return "".join(password)

    def create_user(self, username, password, email=None):
        """Create a new user with sanitized inputs"""
        try:
            # Sanitize inputs
            username = self._sanitize_input(username)
            if email:
                email = self._sanitize_input(email)
            
            # Validate username format (alphanumeric and underscore only)
            if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
                return False
            
            # Validate email format if provided
            if email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                return False
            
            with self._pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # Prevent duplicate email proactively
                if email:
                    cursor.execute('SELECT COUNT(*) FROM users WHERE email = ?', (email,))
                    if cursor.fetchone()[0] > 0:
                        return False
                
                password_hash = self.hash_password(password)
                trial_ends_at = (datetime.now() + timedelta(days=30)).isoformat()
                cursor.execute('''
                    INSERT INTO users (username, password_hash, email, trial_ends_at)
                    VALUES (?, ?, ?, ?)
                ''', (username, password_hash, email, trial_ends_at))
                
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            # Log error but don't expose details
            print(f"Error creating user: {e}")
            return False
    
    def get_user_by_username(self, username):
        """Fetch user record by username"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, email, is_active, is_super_admin, trial_ends_at, password_reset_required FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0], 
                    'username': row[1], 
                    'email': row[2], 
                    'is_active': row[3], 
                    'is_super_admin': row[4],
                    'trial_ends_at': row[5],
                    'password_reset_required': row[6]
                }
            return None
    
    def normalize_duplicate_emails(self):
        """Find duplicate emails and set to NULL for non-primary records to satisfy unique index"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            # Find duplicate emails
            cursor.execute('''
                SELECT email FROM users 
                WHERE email IS NOT NULL
                GROUP BY email
                HAVING COUNT(*) > 1
            ''')
            duplicates = [row[0] for row in cursor.fetchall()]
            for email in duplicates:
                # Keep the earliest created (lowest id) and nullify others
                cursor.execute('SELECT id FROM users WHERE email = ? ORDER BY id ASC', (email,))
                ids = [r[0] for r in cursor.fetchall()]
                # Nullify all except first
                for user_id in ids[1:]:
                    cursor.execute('UPDATE users SET email = NULL WHERE id = ?', (user_id,))
            conn.commit()
    
    def migrate_schema(self):
        """Ensure required columns exist in legacy databases."""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            # Check users table columns
            cursor.execute("PRAGMA table_info(users)")
            cols = [row[1] for row in cursor.fetchall()]
            # Add is_super_admin if missing
            if 'is_super_admin' not in cols:
                cursor.execute("ALTER TABLE users ADD COLUMN is_super_admin INTEGER DEFAULT 0")
                cursor.execute("UPDATE users SET is_super_admin = 0 WHERE is_super_admin IS NULL")
            
            # Add trial_ends_at if missing
            if 'trial_ends_at' not in cols:
                cursor.execute("ALTER TABLE users ADD COLUMN trial_ends_at TIMESTAMP")
                # Set default trial for existing users (30 days from now, as a courtesy)
                trial_end = (datetime.now() + timedelta(days=30)).isoformat()
                cursor.execute("UPDATE users SET trial_ends_at = ? WHERE trial_ends_at IS NULL", (trial_end,))
            
            # Add password_reset_required if missing (for bcrypt migration)
            if 'password_reset_required' not in cols:
                cursor.execute("ALTER TABLE users ADD COLUMN password_reset_required INTEGER DEFAULT 0")
                # Mark SHA-256 passwords (64 hex chars) for reset
                cursor.execute('''
                    UPDATE users 
                    SET password_reset_required = 1 
                    WHERE LENGTH(password_hash) = 64 
                    AND password_hash NOT LIKE '$2%'
                ''')
            
            # Ensure indexes (partial unique for non-NULL emails)
            try:
                cursor.execute('''
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL
                ''')
            except sqlite3.OperationalError:
                # Fallback: attempt normal unique index after normalization
                self.normalize_duplicate_emails()
                try:
                    cursor.execute('''
                        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)
                    ''')
                except Exception:
                    pass
            conn.commit()
    
    def get_user_by_email(self, email):
        """Fetch user record by email"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, email, is_active, is_super_admin FROM users WHERE email = ?', (email,))
            row = cursor.fetchone()
            if row:
                return {'id': row[0], 'username': row[1], 'email': row[2], 'is_active': row[3], 'is_super_admin': row[4]}
            return None
    
    def is_super_admin(self, username) -> bool:
        """Check if user is super admin"""
        u = self.get_user_by_username(username)
        return bool(u and u.get('is_super_admin'))
    
    def ensure_super_admin_exists(self):
        """Ensure there is at least one super admin account, create from environment variables only"""
        try:
            with self._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM users WHERE is_super_admin = 1')
                count = cursor.fetchone()[0]
                if count == 0:
                    # Get from environment variables only (no defaults for security)
                    username = os.getenv('SUPER_ADMIN_USER')
                    password = os.getenv('SUPER_ADMIN_PASS')
                    email = os.getenv('SUPER_ADMIN_EMAIL')
                    
                    # Require all credentials to be set via environment
                    if not all([username, password, email]):
                        import warnings
                        warnings.warn(
                            "Super admin credentials not configured. "
                            "Set SUPER_ADMIN_USER, SUPER_ADMIN_PASS, and SUPER_ADMIN_EMAIL environment variables."
                        )
                        return
                    
                    # Validate username format
                    if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
                        return
                        
                    pw_hash = self.hash_password(password)
                    cursor.execute('''
                        INSERT OR IGNORE INTO users (username, password_hash, email, is_active, is_super_admin)
                        VALUES (?, ?, ?, 1, 1)
                    ''', (username, pw_hash, email))
                    conn.commit()
        except Exception as e:
            # Log but don't expose details
            print(f"Error ensuring super admin: {e}")
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user info"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, password_hash, failed_attempts, locked_until, is_active, trial_ends_at, password_reset_required
                FROM users WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            
            if not user:
                return None
            
            user_id, username, stored_hash, failed_attempts, locked_until, is_active, trial_ends_at, pwd_reset_required = user
            
            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                return {'error': 'locked', 'locked_until': locked_until}
            
            # Check if account is active
            if not is_active:
                return None
            
            # Check if password reset is required (migration from SHA-256 to bcrypt)
            if pwd_reset_required:
                return {'error': 'password_reset_required', 'message': 'Password reset required for security update. Please use forgot password flow.'}
            
            # Verify password using bcrypt
            if self.verify_password(password, stored_hash):
                # Reset failed attempts on successful login
                self.reset_failed_attempts(username)
                return {
                    'id': user_id, 
                    'username': username,
                    'trial_ends_at': trial_ends_at
                }
            else:
                # Increment failed attempts
                self.increment_failed_attempts(username)
                return None
    
    def increment_failed_attempts(self, username):
        """Increment failed login attempts"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET failed_attempts = failed_attempts + 1,
                    locked_until = CASE 
                        WHEN failed_attempts >= 5 THEN datetime('now', '+15 minutes')
                        ELSE locked_until
                    END
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
    
    def reset_failed_attempts(self, username):
        """Reset failed attempts on successful login"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET failed_attempts = 0,
                    locked_until = NULL,
                    last_login = CURRENT_TIMESTAMP
                WHERE username = ?
            ''', (username,))
            
            conn.commit()
    
    def generate_otp(self, username, length: int = 6, ttl_minutes: int = 10):
        """Generate and store OTP code for the given user, returns (code, expires_at)"""
        user = self.get_user_by_username(username)
        if not user or not user.get('email'):
            return None
        
        # Create OTP code (digits only)
        digits = string.digits
        otp_code = ''.join(secrets.choice(digits) for _ in range(length))
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            # Remove previous OTPs for this user
            cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
            # Insert new OTP
            cursor.execute('''
                INSERT INTO otp_codes (username, email, otp_code, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (username, user['email'], otp_code, expires_at))
            conn.commit()
        
        return {'code': otp_code, 'email': user['email'], 'expires_at': expires_at}
    
    def verify_otp(self, username, code):
        """Verify OTP code for user; returns True on success, False otherwise"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT otp_code, expires_at FROM otp_codes
                WHERE username = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (username,))
            row = cursor.fetchone()
            if not row:
                return False
            otp_code, expires_at = row
            # Check expiration
            try:
                exp_dt = datetime.fromisoformat(expires_at)
            except Exception:
                # SQLite may return already datetime
                exp_dt = expires_at if isinstance(expires_at, datetime) else datetime.now() - timedelta(days=1)
            is_valid = (str(code).strip() == str(otp_code).strip()) and (exp_dt > datetime.now())
            if is_valid:
                # consume OTP
                cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
                conn.commit()
            return is_valid
    
    def record_activity(self, username, action: str, metadata: str = None):
        """Record an activity event"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO activity_logs (username, action, metadata)
                VALUES (?, ?, ?)
            ''', (username, action, metadata))
            conn.commit()
    
    def record_feature_usage(self, username, feature_name: str):
        """Increment feature usage count"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            # Upsert-like behavior
            cursor.execute('''
                SELECT id, use_count FROM feature_usage WHERE feature_name = ? AND username = ?
            ''', (feature_name, username))
            row = cursor.fetchone()
            if row:
                new_count = (row[1] or 0) + 1
                cursor.execute('''
                    UPDATE feature_usage SET use_count = ?, last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_count, row[0]))
            else:
                cursor.execute('''
                    INSERT INTO feature_usage (feature_name, username, use_count, last_used)
                    VALUES (?, ?, 1, CURRENT_TIMESTAMP)
                ''', (feature_name, username))
            conn.commit()
    
    def get_users_dataframe(self):
        """Return users as list of dicts"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, username, email, is_active, is_super_admin, created_at, last_login FROM users')
            rows = cursor.fetchall()
            cols = ['id','username','email','is_active','is_super_admin','created_at','last_login']
            return [dict(zip(cols, r)) for r in rows]
    
    def get_feature_usage_stats(self):
        """Return feature usage stats"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT feature_name, SUM(use_count) as total, MAX(last_used) FROM feature_usage GROUP BY feature_name ORDER BY total DESC')
            rows = cursor.fetchall()
            return [{'feature_name': r[0], 'total': r[1], 'last_used': r[2]} for r in rows]
    
    def get_activity_summary(self):
        """Return recent activity summary"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT action, COUNT(*) as total FROM activity_logs GROUP BY action ORDER BY total DESC')
            rows = cursor.fetchall()
            return [{'action': r[0], 'total': r[1]} for r in rows]
    
    # Admin operations
    def change_password(self, username, old_password, new_password):
        """Change user's password after verifying old password"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if not row:
                return False
            current_hash = row[0]
            if not self.verify_password(old_password, current_hash):
                return False
            new_hash = self.hash_password(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE username = ?', (new_hash, username))
            conn.commit()
            return True
    
    def delete_user(self, username):
        """Delete user and related sessions/OTP"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE username = ?', (username,))
            cursor.execute('DELETE FROM otp_codes WHERE username = ?', (username,))
            cursor.execute('DELETE FROM users WHERE username = ?', (username,))
            conn.commit()
            return True
    
    def set_user_active(self, username, is_active: bool):
        """Activate or deactivate user"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_active = ? WHERE username = ?', (1 if is_active else 0, username))
            conn.commit()
            return True
    
    def set_user_super_admin(self, username, is_super_admin: bool):
        """Grant or revoke super admin role"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE users SET is_super_admin = ? WHERE username = ?', (1 if is_super_admin else 0, username))
            conn.commit()
            return True
    
    # SMTP settings
    def get_smtp_config(self):
        """Get SMTP configuration from app settings"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT smtp_host, smtp_port, smtp_user, smtp_pass, smtp_sender, smtp_tls FROM app_settings WHERE id = 1')
            row = cursor.fetchone()
            if not row:
                return {}
            return {
                'host': row[0],
                'port': row[1],
                'user': row[2],
                'password': row[3],
                'sender': row[4],
                'tls': bool(row[5]) if row[5] is not None else True
            }
    
    def set_smtp_config(self, cfg: dict):
        """Upsert SMTP configuration into app settings"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM app_settings WHERE id = 1')
            exists = cursor.fetchone()[0] > 0
            if exists:
                cursor.execute('''
                    UPDATE app_settings 
                    SET smtp_host = ?, smtp_port = ?, smtp_user = ?, smtp_pass = ?, smtp_sender = ?, smtp_tls = ?
                    WHERE id = 1
                ''', (cfg.get('host'), cfg.get('port'), cfg.get('user'), cfg.get('password'), cfg.get('sender'), 1 if cfg.get('tls', True) else 0))
            else:
                cursor.execute('''
                    INSERT INTO app_settings (id, smtp_host, smtp_port, smtp_user, smtp_pass, smtp_sender, smtp_tls)
                    VALUES (1, ?, ?, ?, ?, ?, ?)
                ''', (cfg.get('host'), cfg.get('port'), cfg.get('user'), cfg.get('password'), cfg.get('sender'), 1 if cfg.get('tls', True) else 0))
            conn.commit()
            return True
    
    def create_session(self, username):
        """Create a new session token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=24)
        
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (username, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (username, token, expires_at))
            
            conn.commit()
        
        return token
    
    def validate_session(self, token):
        """Validate session token"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT username, expires_at
                FROM sessions
                WHERE session_token = ? AND expires_at > datetime('now')
            ''', (token,))
            
            session = cursor.fetchone()
            
            if session:
                return {'username': session[0]}
            return None
    
    def delete_session(self, token):
        """Delete session (logout)"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sessions WHERE session_token = ?', (token,))
            
            conn.commit()
    
    def is_username_available(self, username):
        """Check if username is available"""
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
            count = cursor.fetchone()[0]
            
            return count == 0

# Initialize global auth database
auth_db = AuthDatabase()
