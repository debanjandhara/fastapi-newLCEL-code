import sqlite3
import os

DB_NAME = 'chatbot.db'

def create_connection():
    """Create a database connection and create the database file if it doesn't exist."""
    db_exists = os.path.exists(DB_NAME)
    conn = sqlite3.connect(DB_NAME)
    if not db_exists:
        create_tables(conn)
    return conn

def create_tables(conn):
    """Create the Messages and SessionHistory tables if they don't exist."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                history_id TEXT,
                sender TEXT CHECK(sender IN ('AI', 'User')) NOT NULL,
                message_text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS SessionHistory (
                instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT ,
                history_id TEXT,
                history_name TEXT DEFAULT 'New Chat',
                session_creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                history_creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

def insert_message(history_id, sender, message_text):
    conn = create_connection()
    with conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Messages (history_id, sender, message_text) VALUES (?, ?, ?)",
            (history_id, sender, message_text)
        )
    conn.close()

def check_and_insert_session(session_id, history_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM SessionHistory WHERE session_id = ? AND history_id = ?", (session_id, history_id))
    row = cur.fetchone()
    if not row:
        insert_history(session_id, history_id)
    conn.close()

def get_recent_messages(history_id=None, limit=6):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT sender, message_text FROM (SELECT sender, message_text, message_id FROM Messages WHERE history_id = ? ORDER BY message_id DESC LIMIT ?) sub ORDER BY message_id ASC;",
        (history_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def insert_history(session_id, history_id):
    conn = create_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO SessionHistory (session_id, history_id) VALUES (?, ?)", (session_id, history_id))
    conn.close()

def update_history_name(session_id, history_id, history_name):
    conn = create_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("UPDATE SessionHistory SET history_name = ? WHERE session_id = ? AND history_id = ?", (history_name, session_id, history_id))
    conn.close()

def get_history_list(session_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT history_id, history_name FROM SessionHistory WHERE session_id = ?", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_chat_history(history_id):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT sender, message_text FROM Messages WHERE history_id = ? ORDER BY message_id",
        (history_id,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows

import csv

def export_messages_to_csv(file_path='messages.csv'):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM Messages ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    
    # Write data to CSV
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['message_id', 'history_id', 'sender', 'message_text', 'timestamp'])
        # Write the data
        writer.writerows(rows)
    
    print(f"Data successfully exported to {file_path}")

def export_messages_to_csv2(file_path='messages2.csv'):
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM SessionHistory")
    rows = cur.fetchall()
    conn.close()
    
    # Write data to CSV
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['instance_id', 'session_id', 'history_id', 'history_name', 'session_creation_timestamp', 'history_creation_timestamp'])
        # Write the data
        writer.writerows(rows)
    
    print(f"Data successfully exported to {file_path}")

export_messages_to_csv()
export_messages_to_csv2()
