import streamlit as st
import sqlite3
import pandas as pd
from openai import OpenAI
import re

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Title
st.set_page_config(page_title="AI SQL Agent", layout="wide")
st.title("🧠 AI SQL Assistant")

# Database connection
DB_PATH = "data.sqlite"

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_schema():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = {}
    for t in tables:
        table_name = t[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema[table_name] = cursor.fetchall()
    conn.close()
    return schema

schema = get_schema()

# Detect if request is yearly or monthly
def detect_special_query(user_request):
    year_match = re.search(r"(\d{4})", user_request)
    month_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)", user_request, re.IGNORECASE)

    if year_match and not month_match:
        return "yearly", year_match.group(1)
    if year_match and month_match:
        return "monthly", (month_match.group(0), year_match.group(1))
    return None, None

# Generate pivot SQL
def generate_pivot_sql(user_request):
    qtype, value = detect_special_query(user_request)

    if qtype == "yearly":
        year = value
        sql = f"""
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
        WHERE strftime('%Y', order_date) = '{year}'
        GROUP BY name;
        """
        return sql.strip()

    if qtype == "monthly":
        month, year = value
        months_map = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12"
        }
        month_num = months_map[month.lower()]

        day_cases = []
        for d in range(1, 32):
            day_str = str(d).zfill(2)
            day_cases.append(f"SUM(CASE WHEN strftime('%d', order_date) = '{day_str}' THEN qty ELSE 0 END) AS Day{d}")

        sql = f"""
        SELECT 
            name,
            {', '.join(day_cases)}
        FROM orders
        WHERE strftime('%Y-%m', order_date) = '{year}-{month_num}'
        GROUP BY name;
        """
        return sql.strip()

    return None

# Generate SQL using OpenAI if not special pivot
def generate_sql(user_request):
    special_sql = generate_pivot_sql(user_request)
    if special_sql:
        return special_sql

    schema_text = "\n".join(
        [f"{t}: {[(c[1], c[2]) for c in cols]}" for t, cols in schema.items()]
    )

    prompt = f"""
    You are a helpful SQL generator for SQLite.
    Schema:
    {schema_text}

    User request: {user_request}

    Generate a correct SELECT query in SQLite syntax.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You convert natural language to SQL."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    sql = response.choices[0].message.content.strip()
    return sql

# Run query safely
def run_query(sql):
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# Streamlit UI
user_request = st.text_input("🔍 Ask your database in plain English:")

if st.button("✨ Generate SQL"):
    sql = generate_sql(user_request)
    st.code(sql, language="sql")

    if st.button("▶️ Run SQL"):
        df, error = run_query(sql)
        if error:
            st.error(f"❌ {error}")
        else:
            st.success(f"✅ Returned {len(df)} rows.")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
