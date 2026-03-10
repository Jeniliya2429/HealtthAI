"""
=============================================================================
  AI Multi-Disease Risk Assessment System
  FILE: auth.py  |  v1.0
  PURPOSE: User Registration, Login, Session Management
  Uses: SQLite (built-in) + hashlib + secrets (no extra installs needed)
=============================================================================
"""

import sqlite3
import hashlib
import secrets
import os
import re
from datetime import datetime

DB_PATH = "users.db"

# ─────────────────────────────────────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create the users table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                username     TEXT    UNIQUE NOT NULL,
                email        TEXT    UNIQUE NOT NULL,
                full_name    TEXT    NOT NULL,
                password_hash TEXT   NOT NULL,
                salt         TEXT    NOT NULL,
                gender       TEXT    DEFAULT 'Not specified',
                age          INTEGER DEFAULT 0,
                created_at   TEXT    NOT NULL,
                last_login   TEXT
            )
        """)
        # Assessment history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS assessment_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                assessed_at  TEXT    NOT NULL,
                diseases     TEXT    NOT NULL,
                health_score REAL,
                risk_summary TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB init error: {e}")
        return False


def get_connection():
    """Get a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, )
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
#  PASSWORD UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _generate_salt() -> str:
    """Generate a cryptographically secure random salt."""
    return secrets.token_hex(32)   # 64-char hex string


def _hash_password(password: str, salt: str) -> str:
    """SHA-256 hash with salt — secure for this use case."""
    combined = (password + salt).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


def _verify_password(password: str, salt: str, stored_hash: str) -> bool:
    """Verify a password against its stored hash."""
    return _hash_password(password, salt) == stored_hash


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_email(email: str) -> bool:
    pattern = r'^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_password(password: str) -> tuple[bool, str]:
    """
    Returns (is_valid, message).
    Password rules: 8+ chars, at least 1 digit, 1 uppercase, 1 lowercase.
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number."
    return True, "OK"


def validate_username(username: str) -> tuple[bool, str]:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 30:
        return False, "Username must be under 30 characters."
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def register_user(username: str, email: str, full_name: str,
                  password: str, gender: str = "Not specified",
                  age: int = 0) -> dict:
    """
    Register a new user.
    Returns {"success": True/False, "message": str}
    """
    # Validate inputs
    ok, msg = validate_username(username.strip())
    if not ok:
        return {"success": False, "message": msg}

    if not full_name.strip() or len(full_name.strip()) < 2:
        return {"success": False, "message": "Please enter your full name (min 2 characters)."}

    if not validate_email(email.strip()):
        return {"success": False, "message": "Please enter a valid email address."}

    ok, msg = validate_password(password)
    if not ok:
        return {"success": False, "message": msg}

    # Hash password
    salt         = _generate_salt()
    password_hash= _hash_password(password, salt)
    now          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO users
              (username, email, full_name, password_hash, salt, gender, age, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (username.strip().lower(), email.strip().lower(),
              full_name.strip(), password_hash, salt, gender, age, now))
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Account created! Welcome, {full_name.split()[0]} 🎉"}
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return {"success": False, "message": "Username already taken. Choose another."}
        if "email" in str(e):
            return {"success": False, "message": "Email already registered. Try logging in."}
        return {"success": False, "message": "Registration failed. Please try again."}
    except Exception as e:
        return {"success": False, "message": f"Unexpected error: {str(e)[:100]}"}


def login_user(username_or_email: str, password: str) -> dict:
    """
    Log in with username OR email.
    Returns {"success": True/False, "message": str, "user": dict or None}
    """
    if not username_or_email.strip() or not password:
        return {"success": False, "message": "Please enter your username/email and password.", "user": None}

    try:
        conn = get_connection()
        cur  = conn.cursor()
        ident = username_or_email.strip().lower()

        # Try username first, then email
        cur.execute("SELECT * FROM users WHERE username = ? OR email = ?", (ident, ident))
        row = cur.fetchone()

        if row is None:
            conn.close()
            return {"success": False, "message": "No account found with that username or email.", "user": None}

        if not _verify_password(password, row["salt"], row["password_hash"]):
            conn.close()
            return {"success": False, "message": "Incorrect password. Please try again.", "user": None}

        # Update last login
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
        conn.commit()
        conn.close()

        user_data = {
            "id"        : row["id"],
            "username"  : row["username"],
            "email"     : row["email"],
            "full_name" : row["full_name"],
            "gender"    : row["gender"],
            "age"       : row["age"],
            "created_at": row["created_at"],
            "last_login": now,
        }
        return {"success": True,
                "message": f"Welcome back, {row['full_name'].split()[0]}! 👋",
                "user"   : user_data}

    except Exception as e:
        return {"success": False, "message": f"Login error: {str(e)[:100]}", "user": None}


def save_assessment(user_id: int, diseases: list, health_score: float, risk_summary: str):
    """Save a completed assessment to history."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("""
            INSERT INTO assessment_history
              (user_id, assessed_at, diseases, health_score, risk_summary)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, now, ", ".join(diseases),
              round(health_score, 1), risk_summary))
        conn.commit()
        conn.close()
    except Exception:
        pass  # silently skip — non-critical


def get_user_history(user_id: int) -> list:
    """Get last 10 assessments for a user."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT assessed_at, diseases, health_score, risk_summary
            FROM assessment_history
            WHERE user_id = ?
            ORDER BY assessed_at DESC
            LIMIT 10
        """, (user_id,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def update_profile(user_id: int, full_name: str, gender: str, age: int) -> dict:
    """Update user profile info."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            UPDATE users SET full_name = ?, gender = ?, age = ?
            WHERE id = ?
        """, (full_name.strip(), gender, age, user_id))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Profile updated successfully!"}
    except Exception as e:
        return {"success": False, "message": str(e)[:100]}


def change_password(user_id: int, current_password: str, new_password: str) -> dict:
    """Change user password after verifying current one."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT password_hash, salt FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"success": False, "message": "User not found."}
        if not _verify_password(current_password, row["salt"], row["password_hash"]):
            conn.close()
            return {"success": False, "message": "Current password is incorrect."}
        ok, msg = validate_password(new_password)
        if not ok:
            conn.close()
            return {"success": False, "message": msg}
        new_salt = _generate_salt()
        new_hash = _hash_password(new_password, new_salt)
        cur.execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                    (new_hash, new_salt, user_id))
        conn.commit()
        conn.close()
        return {"success": True, "message": "Password changed successfully!"}
    except Exception as e:
        return {"success": False, "message": str(e)[:100]}


def get_total_users() -> int:
    """Return total registered user count."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0
