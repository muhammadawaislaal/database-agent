import os
import sqlite3
import pandas as pd
import streamlit as st
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI SQL Assistant", layout="wide")
st.title("🧠 AI SQL Assistant")

# Database selection
st.sidebar.header("Database")
db_type = st.sidebar.selectbox("DB type", ["SQLite (local demo)"])
db_path = st.sidebar.text_input("SQLite path (local)", "data.sqlite")

if st.sidebar.button("Init demo DB (if missing)"):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL,
            qty INTEGER,
            order_date TEXT
        );
        """)
        cursor.executemany("""
        INSERT INTO orders (name, category, price, qty, order_date) VALUES (?, ?, ?, ?, ?)
        """, [
            ("Widget A", "Widgets", 9.99, 3, "2024-08-01"),
            ("Gadget X", "Gadgets", 29.50, 1, "2024-08-15"),
            ("Widget B", "Widgets", 19.99, 2, "2024-08-20"),
            ("Widget A", "Widgets", 9.99, 1, "2024-09-01"),
        ])
        conn.commit()
        conn.close()
        st.success("Demo DB initialized!")

if st.sidebar.button("Connect / Refresh schema"):
    st.session_state["schema"] = None

# -------------------------
# Helper: Get schema
# -------------------------
def get_schema(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for t in tables:
        table_name = t[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        cols = cursor.fetchall()
        schema[table_name] = [c[1] for c in cols]
    return schema

# -------------------------
# AI SQL Generator
# -------------------------
def generate_sql(user_request, schema):
    # Handle special pivot cases
    if "details for" in user_request.lower() and "2024" in user_request:
        if any(month in user_request.lower() for month in 
               ["january","february","march","april","may","june",
                "july","august","september","october","november","december"]):
            # Monthly breakdown into days
            return build_daily_pivot_sql(user_request)
        else:
            # Yearly breakdown into months
            return build_monthly_pivot_sql(user_request)

    # Otherwise, use AI to generate SQL
    prompt = f"""
    You are an expert SQL generator. Convert the user request into a valid SQLite SQL query.
    Schema: {schema}
    User request: {user_request}
    Only return the SQL code, nothing else.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# -------------------------
# Pivot SQL builders
# -------------------------
def build_monthly_pivot_sql(user_request):
    sql = """
    SELECT 
        name,
        SUM(CASE WHEN strftime('%m', order_date) = '01' THEN qty ELSE 0 END) AS Jan,
        SUM(CASE WHEN strftime('%m', order_date) = '02' THEN qty ELSE 0 END) AS Feb,
        SUM(CASE WHEN strftime('%m', order_date) = '03' THEN qty ELSE 0 END) AS Mar,
        SUM(CASE WHEN strftime('%m', order_date) = '04' THEN qty ELSE 0 END) AS Apr,
        SUM(CASE WHEN strftime('%m', order_date) = '05' THEN qty ELSE 0 END) AS May,
        SUM(CASE WHEN strftime('%m', order_date) = '06' THEN qty ELSE 0 END) AS Jun,
        SUM(CASE WHEN strftime('%m', order_date) = '07' THEN qty ELSE 0 END) AS Jul,
        SUM(CASE WHEN strftime('%m', order_date) = '08' THEN qty ELSE 0 END) AS Aug,
        SUM(CASE WHEN strftime('%m', order_date) = '09' THEN qty ELSE 0 END) AS Sep,
        SUM(CASE WHEN strftime('%m', order_date) = '10' THEN qty ELSE 0 END) AS Oct,
        SUM(CASE WHEN strftime('%m', order_date) = '11' THEN qty ELSE 0 END) AS Nov,
        SUM(CASE WHEN strftime('%m', order_date) = '12' THEN qty ELSE 0 END) AS Dec
    FROM orders
    WHERE strftime('%Y', order_date) = '2024'
    GROUP BY name;
    """
    return sql.strip()

def build_daily_pivot_sql(user_request):
    sql = "SELECT name,\n"
    for day in range(1, 32):
        sql += f"    SUM(CASE WHEN strftime('%d', order_date) = '{day:02}' THEN qty ELSE 0 END) AS Day{day},\n"
    sql = sql.rstrip(",\n") + "\n"
    sql += """FROM orders
    WHERE strftime('%Y-%m', order_date) = '2024-08'
    GROUP BY name;"""
    return sql.strip()

# -------------------------
# Main App
# -------------------------
user_request = st.text_area("🔍 Your request in plain English:")

if st.button("✨ Generate SQL"):
    if "schema" not in st.session_state:
        conn = sqlite3.connect(db_path)
        st.session_state["schema"] = get_schema(conn)
        conn.close()
    schema = st.session_state["schema"]
    sql_query = generate_sql(user_request, schema)
    st.session_state["sql_query"] = sql_query

if "sql_query" in st.session_state:
    st.subheader("📝 SQL Query (editable before running):")
    sql_query = st.text_area("SQL Query", value=st.session_state["sql_query"], height=200)

    if st.button("▶️ Run SQL"):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            st.success(f"Returned {len(df)} rows.")
            st.dataframe(df)

            # Quick visualization
            st.subheader("📊 Quick visualization")
            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if numeric_cols:
                choice = st.selectbox("Choose numeric column to chart", numeric_cols)
                st.bar_chart(df.set_index(df.columns[0])[choice])
        except Exception as e:
            st.error(f"❌ Error: {e}")
