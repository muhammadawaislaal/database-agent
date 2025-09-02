"""
AI SQL Assistant (Streamlit)
- Works locally & on Streamlit Cloud
- Natural language → SQL → Executes on SQLite DB
- Uses OpenAI API (key from st.secrets or env)
"""

import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import openai

# ------------------ API KEY HANDLING ------------------
if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
else:
    st.error("❌ No OpenAI API key found. Please add it in Streamlit Secrets.")
    st.stop()

# ------------------ INIT DB ------------------
DB_PATH = "data.sqlite"

SAMPLE_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    qty INTEGER,
    order_date TEXT,
    FOREIGN KEY(customer_id) REFERENCES customers(id),
    FOREIGN KEY(product_id) REFERENCES products(id)
);

INSERT INTO customers (name,email,signup_date) VALUES
  ('Alice','alice@example.com','2024-06-10'),
  ('Bob','bob@example.com','2024-07-05'),
  ('Carol','carol@example.com','2024-08-12');

INSERT INTO products (name,category,price) VALUES
  ('Widget A','Widgets',9.99),
  ('Widget B','Widgets',19.99),
  ('Gadget X','Gadgets',29.5);

INSERT INTO orders (customer_id,product_id,qty,order_date) VALUES
  (1,1,3,'2024-08-01'),
  (1,3,1,'2024-08-15'),
  (2,2,2,'2024-08-20'),
  (3,1,1,'2024-09-01');
"""

def init_sample_db():
    if not os.path.exists(DB_PATH):
        engine = create_engine(f"sqlite:///{DB_PATH}")
        with engine.begin() as conn:
            for stmt in SAMPLE_SQL.strip().split(";"):
                if stmt.strip():
                    try:
                        conn.execute(text(stmt))
                    except Exception:
                        pass
        engine.dispose()

init_sample_db()
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

# ------------------ SCHEMA INTROSPECTION ------------------
def get_schema(engine):
    insp = inspect(engine)
    schema = {}
    for table in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(table)]
        schema[table] = cols
    return schema

# ------------------ NL → SQL via OpenAI ------------------
def nl_to_sql(nl_request: str, schema: dict) -> str:
    schema_text = "\n".join([f"- {t}: {', '.join(cols)}" for t, cols in schema.items()])
    system_msg = (
        "You are a helpful assistant that ONLY outputs a single SQL query (SQLite). "
        "No explanations. If not answerable, output CANNOT_ANSWER."
    )
    user_msg = f"User request: {nl_request}\n\nSchema:\n{schema_text}"
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # use gpt-4o-mini (cheap & good)
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0,
    )
    return resp["choices"][0]["message"]["content"].strip()

# ------------------ SQL SAFETY ------------------
FORBIDDEN = ["DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"]

def validate_sql(sql: str) -> (bool, str):
    if not sql:
        return False, "Empty query."
    sql_upper = sql.upper()
    for bad in FORBIDDEN:
        if bad in sql_upper:
            return False, f"Forbidden keyword: {bad}"
    if sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        return True, "Query OK"
    return False, "Only SELECT/READ queries are allowed."

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="AI SQL Assistant", layout="wide")
st.title("🧠 AI SQL Assistant")

st.markdown(
    "Type a natural language request, let AI convert it to SQL, "
    "then run the query against the database."
)

schema = get_schema(engine)
with st.expander("📊 Database Schema", expanded=False):
    for t, cols in schema.items():
        st.write(f"**{t}**: {', '.join(cols)}")

# User input
nl_input = st.text_area("🔍 Your request in plain English:", 
                        "Show me total orders per product in August 2024")

if st.button("✨ Generate SQL"):
    with st.spinner("Thinking..."):
        try:
            sql_query = nl_to_sql(nl_input, schema)
            st.session_state["sql_query"] = sql_query
            st.success("SQL generated successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

sql_query = st.session_state.get("sql_query", "")
sql_query = st.text_area("📝 SQL Query (editable before running):", sql_query, height=150)

# Run query
if st.button("▶️ Run SQL"):
    valid, msg = validate_sql(sql_query)
    if not valid:
        st.error(f"❌ {msg}")
    else:
        try:
            df = pd.read_sql_query(text(sql_query), engine)
            st.success(f"Returned {len(df)} rows.")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, file_name="results.csv")
        except Exception as e:
            st.error(f"Execution error: {e}")
