"""
AI SQL Assistant (Streamlit + OpenAI)
- Natural language → SQL → Run safely on SQLite
- Supports Read-only and Full SQL modes
"""

import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from openai import OpenAI

# ------------------ API KEY HANDLING ------------------
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    st.error("❌ No OpenAI API key found. Please add it in Streamlit Secrets or environment.")
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
    """Create a sample SQLite database if missing"""
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
def nl_to_sql(nl_request: str, schema: dict, read_only: bool) -> str:
    schema_text = "\n".join([f"- {t}: {', '.join(cols)}" for t, cols in schema.items()])

    if read_only:
        system_msg = (
            "You are a helpful assistant that ONLY outputs a single SQLite SELECT query. "
            "The query must be read-only (SELECT, WITH, or EXPLAIN). "
            "Do not generate INSERT, UPDATE, DELETE, or schema changes. "
            "No explanations. If not answerable, output CANNOT_ANSWER."
        )
    else:
        system_msg = (
            "You are a helpful assistant that outputs a single valid SQLite query. "
            "It can be SELECT, INSERT, UPDATE, or DELETE, but never DROP/ALTER/PRAGMA. "
            "No explanations. If not answerable, output CANNOT_ANSWER."
        )

    user_msg = f"User request: {nl_request}\n\nSchema:\n{schema_text}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# ------------------ SQL CLEANING + SAFETY ------------------
FORBIDDEN = ["DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"]

def clean_sql_output(sql: str) -> str:
    """Remove code fences and extra text from AI output."""
    if not sql:
        return ""
    # Remove Markdown code fences
    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "")
    # Remove leading explanations like "Here is the query:"
    if "SELECT" in sql.upper():
        sql = re.sub(r"^.*?(SELECT)", r"\1", sql, flags=re.IGNORECASE | re.DOTALL)
    elif any(kw in sql.upper() for kw in ["INSERT", "UPDATE", "DELETE"]):
        sql = re.sub(r"^.*?(INSERT|UPDATE|DELETE)", r"\1", sql, flags=re.IGNORECASE | re.DOTALL)
    return sql.strip()

def validate_sql(sql: str, read_only: bool) -> (bool, str):
    """Check if SQL is safe & allowed for current mode"""
    if not sql:
        return False, "Empty query."

    # 🔥 Clean first
    sql = clean_sql_output(sql)
    sql_upper = sql.upper()

    if sql_upper.startswith("ERROR"):
        return False, sql
    for bad in FORBIDDEN:
        if bad in sql_upper:
            return False, f"Forbidden keyword: {bad}"

    # Regex → detect first keyword ignoring spaces
    first_word = re.match(r"^\s*([A-Z]+)", sql_upper)
    if not first_word:
        return False, "Invalid SQL syntax."

    keyword = first_word.group(1)

    read_only_allowed = {"SELECT", "WITH", "EXPLAIN"}
    full_allowed = read_only_allowed | {"INSERT", "UPDATE", "DELETE"}

    if read_only and keyword in read_only_allowed:
        return True, "Query OK"
    elif not read_only and keyword in full_allowed:
        return True, "Query OK"
    else:
        return False, f"❌ Query type not supported in this mode ({keyword})."

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="AI SQL Assistant", layout="wide")
st.title("🧠 AI SQL Assistant")

st.markdown("Type a natural language request, let AI convert it to SQL, then run it safely.")

schema = get_schema(engine)
with st.expander("📊 Database Schema", expanded=False):
    for t, cols in schema.items():
        st.write(f"**{t}**: {', '.join(cols)}")

# Mode toggle
read_only = st.toggle("🔒 Read-only Mode (safe)", value=True)

if not read_only:
    st.warning("⚠️ Full SQL mode enabled. INSERT/UPDATE/DELETE are allowed. Be careful!")

# User input
nl_input = st.text_area("🔍 Your request in plain English:", 
                        "Show me total orders per product in August 2024")

if st.button("✨ Generate SQL"):
    with st.spinner("Thinking..."):
        sql_query = nl_to_sql(nl_input, schema, read_only)
        st.session_state["sql_query"] = clean_sql_output(sql_query)
        if sql_query.startswith("ERROR"):
            st.error(sql_query)
        elif sql_query == "CANNOT_ANSWER":
            st.warning("⚠️ AI could not generate a query for this request.")
        else:
            st.success("✅ SQL generated successfully!")

sql_query = st.session_state.get("sql_query", "")
sql_query = st.text_area("📝 SQL Query (editable before running):", sql_query, height=150)

# Run query
if st.button("▶️ Run SQL"):
    valid, msg = validate_sql(sql_query, read_only)
    if not valid:
        st.error(f"❌ {msg}")
    else:
        try:
            with engine.begin() as conn:
                if sql_query.strip().upper().startswith("SELECT"):
                    df = pd.read_sql_query(text(sql_query), conn)
                    st.success(f"✅ Returned {len(df)} rows.")
                    st.dataframe(df)
                    if not df.empty:
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download CSV", csv, file_name="results.csv")
                else:
                    conn.execute(text(sql_query))
                    st.success("✅ Query executed successfully (no results to show).")
        except Exception as e:
            st.error(f"❌ Execution error: {e}")
