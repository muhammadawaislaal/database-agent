import streamlit as st
import sqlite3
import pandas as pd
import re

st.set_page_config(page_title="AI SQL Agent", layout="wide")

# -------------------------------
# Database Connection Handler
# -------------------------------
@st.cache_resource
def init_connection(db_type, db_path):
    if db_type == "SQLite (local demo)":
        return sqlite3.connect(db_path, check_same_thread=False)
    return None

# -------------------------------
# Detect Pivot Query Type
# -------------------------------
def detect_query_type(user_request: str):
    text = user_request.lower()
    if "week" in text:
        return "weekly"
    elif "month" in text:
        return "monthly"
    elif "year" in text:
        return "yearly"
    return "normal"

# -------------------------------
# Generate Pivot Queries
# -------------------------------
def build_pivot_query(request_type, table="orders"):
    if request_type == "yearly":
        return f"""
        SELECT 
            name,
            SUM(CASE WHEN strftime('%m', order_date)='01' THEN qty ELSE 0 END) AS Jan,
            SUM(CASE WHEN strftime('%m', order_date)='02' THEN qty ELSE 0 END) AS Feb,
            SUM(CASE WHEN strftime('%m', order_date)='03' THEN qty ELSE 0 END) AS Mar,
            SUM(CASE WHEN strftime('%m', order_date)='04' THEN qty ELSE 0 END) AS Apr,
            SUM(CASE WHEN strftime('%m', order_date)='05' THEN qty ELSE 0 END) AS May,
            SUM(CASE WHEN strftime('%m', order_date)='06' THEN qty ELSE 0 END) AS Jun,
            SUM(CASE WHEN strftime('%m', order_date)='07' THEN qty ELSE 0 END) AS Jul,
            SUM(CASE WHEN strftime('%m', order_date)='08' THEN qty ELSE 0 END) AS Aug,
            SUM(CASE WHEN strftime('%m', order_date)='09' THEN qty ELSE 0 END) AS Sep,
            SUM(CASE WHEN strftime('%m', order_date)='10' THEN qty ELSE 0 END) AS Oct,
            SUM(CASE WHEN strftime('%m', order_date)='11' THEN qty ELSE 0 END) AS Nov,
            SUM(CASE WHEN strftime('%m', order_date)='12' THEN qty ELSE 0 END) AS Dec
        FROM {table}
        WHERE strftime('%Y', order_date)='2024'
        GROUP BY name;
        """

    elif request_type == "monthly":
        query = f"SELECT name,\n"
        for d in range(1, 32):
            query += f"    SUM(CASE WHEN strftime('%d', order_date)='{d:02d}' THEN qty ELSE 0 END) AS Day{d},\n"
        query = query.rstrip(",\n") + f"\nFROM {table}\nWHERE strftime('%Y-%m', order_date)='2024-08'\nGROUP BY name;"
        return query

    elif request_type == "weekly":
        query = f"""
        SELECT 
            name,
            SUM(CASE WHEN strftime('%W', order_date)='01' THEN qty ELSE 0 END) AS Week1,
            SUM(CASE WHEN strftime('%W', order_date)='02' THEN qty ELSE 0 END) AS Week2,
            SUM(CASE WHEN strftime('%W', order_date)='03' THEN qty ELSE 0 END) AS Week3,
            SUM(CASE WHEN strftime('%W', order_date)='04' THEN qty ELSE 0 END) AS Week4,
            SUM(CASE WHEN strftime('%W', order_date)='05' THEN qty ELSE 0 END) AS Week5,
            SUM(CASE WHEN strftime('%W', order_date)='06' THEN qty ELSE 0 END) AS Week6,
            SUM(CASE WHEN strftime('%W', order_date)='07' THEN qty ELSE 0 END) AS Week7,
            SUM(CASE WHEN strftime('%W', order_date)='08' THEN qty ELSE 0 END) AS Week8,
            SUM(CASE WHEN strftime('%W', order_date)='09' THEN qty ELSE 0 END) AS Week9,
            SUM(CASE WHEN strftime('%W', order_date)='10' THEN qty ELSE 0 END) AS Week10
        FROM {table}
        WHERE strftime('%Y', order_date)='2024'
        GROUP BY name;
        """
        return query

    return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧠 AI SQL Assistant")

db_type = st.sidebar.selectbox("DB type", ["SQLite (local demo)"])
db_path = st.sidebar.text_input("SQLite path (local)", "data.sqlite")

if st.sidebar.button("Connect / Refresh schema"):
    conn = init_connection(db_type, db_path)
    st.sidebar.success("Database connected")

user_request = st.text_area("Your request in plain English:")

if st.button("Generate SQL"):
    query_type = detect_query_type(user_request)
    sql_query = build_pivot_query(query_type) if query_type != "normal" else user_request
    st.code(sql_query, language="sql")

    if "select" in sql_query.lower():
        try:
            conn = init_connection(db_type, db_path)
            df = pd.read_sql_query(sql_query, conn)
            st.success(f"Returned {len(df)} rows.")
            st.dataframe(df)

            # Quick visualization
            if not df.empty:
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                if num_cols:
                    chart_col = st.selectbox("Choose numeric column to chart", num_cols)
                    st.bar_chart(df.set_index(df.columns[0])[chart_col])
        except Exception as e:
            st.error(f"❌ Error: {e}")
