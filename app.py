"""
app.py - Minimal AI SQL Assistant (Streamlit)
- Creates a small sample SQLite DB if missing
- Lets you enter natural-language requests and (optionally) uses OpenAI to convert NL -> SQL
- Shows the generated SQL, allows editing, validates it, executes safely, displays and exports results
- Minimal, documented, and easy to extend

Run:
    export OPENAI_API_KEY="sk-..."   # optional, for better NL->SQL
    streamlit run app.py
"""

import os
import re
import json
from typing import Dict
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import sqlalchemy
import openai

# --------------- Config & small helpers ---------------

DB_PATH_DEFAULT = "data.sqlite"
SAVED_QUERIES_FILE = "saved_queries.json"

st.set_page_config(page_title="AI SQL Assistant", layout="wide")

def set_openai_key_from_env():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        openai.api_key = key
    return bool(key)

# --------------- Sample DB init (runs only if DB missing) ---------------

SAMPLE_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    signup_date TEXT
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price REAL
);

CREATE TABLE orders (
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

def init_sample_db(path: str = DB_PATH_DEFAULT):
    if os.path.exists(path):
        return False
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        for stmt in SAMPLE_SQL.strip().split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    engine.dispose()
    return True

# --------------- Schema introspection ---------------

def get_schema(engine) -> Dict[str, list]:
    insp = inspect(engine)
    schema = {}
    for table_name in insp.get_table_names():
        cols = [c["name"] for c in insp.get_columns(table_name)]
        schema[table_name] = cols
    return schema

def pretty_schema_markdown(schema: Dict[str, list]) -> str:
    lines = []
    for t, cols in schema.items():
        lines.append(f"**{t}**: {', '.join(cols)}")
    return "\n\n".join(lines)

# --------------- NL -> SQL via OpenAI ---------------

def prompt_for_sql(nl: str, schema: Dict[str, list]) -> str:
    # Build a helpful prompt that includes schema and constraints
    schema_text = "\n".join([f"- {t}: {', '.join(cols)}" for t, cols in schema.items()])
    system_msg = (
        "You are an assistant that *only* outputs a single SQL query (SQLite dialect) "
        "that answers the user's request. Do NOT explain, do NOT add any extra text. "
        "If the request cannot be answered with SQL (for example needs external API), say 'CANNOT_ANSWER'."
    )
    user_msg = (
        f"User request: {nl}\n\n"
        f"Database schema:\n{schema_text}\n\n"
        "Produce a single valid SQL query (SQLite). Use standard functions and ISO date patterns if needed. "
        "Return only the SQL statement."
    )
    return system_msg, user_msg

def nl_to_sql_openai(nl: str, schema: Dict[str, list]) -> str:
    system_msg, user_msg = prompt_for_sql(nl, schema)
    # call OpenAI Chat Completion
    resp = openai.ChatCompletion.create(
        model="gpt-4",  # typical; change if not available in your account
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    sql = resp["choices"][0]["message"]["content"].strip()
    return sql

# --------------- SQL validation / safety ---------------

FORBIDDEN_KEYWORDS = ["DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"]
ALLOWED_READ_STARTS = ["SELECT", "WITH", "EXPLAIN"]

def validate_sql(sql: str, allow_writes: bool = False) -> (bool, str):
    if not sql or not sql.strip():
        return False, "Empty query."
    # remove comments
    sql_no_comments = re.sub(r"(--.*?$)|(/\*[\s\S]*?\*/)", "", sql, flags=re.MULTILINE).strip()
    if sql_no_comments.count(";") > 1:
        return False, "Multiple statements detected (only single statement allowed)."
    sql_upper = sql_no_comments.strip().upper()
    # check forbidden keywords anywhere
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_upper:
            return False, f"Forbidden SQL keyword detected: {kw}"
    # check start of statement
    m = re.match(r"^\s*(\w+)", sql_upper)
    if not m:
        return False, "Couldn't parse the start of the SQL statement."
    first = m.group(1)
    if first in ALLOWED_READ_STARTS:
        return True, sql_no_comments
    if first in ("INSERT", "UPDATE", "DELETE") and allow_writes:
        return True, sql_no_comments
    if first in ("INSERT", "UPDATE", "DELETE") and not allow_writes:
        return False, f"{first} queries are write operations. Toggle 'Allow write queries' to enable."
    return False, f"SQL starting with '{first}' not allowed by default."

# --------------- Saved queries helpers ---------------

def load_saved_queries():
    if os.path.exists(SAVED_QUERIES_FILE):
        try:
            with open(SAVED_QUERIES_FILE, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_saved_queries(qs: dict):
    with open(SAVED_QUERIES_FILE, "w", encoding="utf8") as f:
        json.dump(qs, f, indent=2)

# --------------- Streamlit UI ---------------

st.title("🧠 AI SQL Assistant — Minimal, user-friendly")

st.markdown(
    """
This tiny app:
- Creates a small sample SQLite database (`data.sqlite`) if missing.
- Converts natural language to SQL using OpenAI (optional).
- Shows the generated SQL so you can review/modify before running.
- Validates for safety (no DROP/ALTER/etc. and single statement).
- Executes queries and shows results (downloadable CSV).

**Quick notes**
- Best results if you set `OPENAI_API_KEY` in your environment before running.
- By default only `SELECT` / `WITH` queries are allowed. You can enable write queries with a toggle.
"""
)

# DB controls
st.sidebar.header("Database")
db_path = st.sidebar.text_input("SQLite DB path", value=DB_PATH_DEFAULT)
if st.sidebar.button("Init sample DB (if missing)"):
    created = init_sample_db(db_path)
    if created:
        st.sidebar.success(f"Sample DB created at {db_path}")
    else:
        st.sidebar.info(f"DB already exists: {db_path}")

# Connect to DB
engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

# Show schema
try:
    schema = get_schema(engine)
except Exception as e:
    st.error(f"Could not inspect database at {db_path}: {e}")
    st.stop()

st.subheader("Database schema")
st.markdown(pretty_schema_markdown(schema))

# NL input area
st.subheader("Write a request in natural language")
nl_input = st.text_area("Describe what you want (example: \"Top 5 customers by total spend in August 2024\")",
                        value="Show total orders per product category in August 2024", height=120)

# OpenAI key status
has_key = set_openai_key_from_env()
col1, col2, col3 = st.columns([1,1,2])
with col1:
    if has_key:
        st.success("OpenAI key found (using LLM).")
    else:
        st.info("No OPENAI_API_KEY found. LLM disabled. Install an API key for best results.")

with col2:
    allow_writes = st.checkbox("Allow write queries (INSERT/UPDATE/DELETE)", value=False)

with col3:
    if st.button("Generate SQL (AI)"):
        if not has_key:
            st.warning("Set OPENAI_API_KEY to use the AI generator. See the README instructions above.")
            generated_sql = ""
        else:
            with st.spinner("Asking the model for SQL..."):
                try:
                    generated_sql = nl_to_sql_openai(nl_input, schema)
                    # strip outer backticks if any
                    generated_sql = generated_sql.strip().strip("```").strip()
                    st.session_state["generated_sql"] = generated_sql
                    st.success("SQL generated. Review & run below.")
                except Exception as e:
                    st.error(f"Error from OpenAI: {e}")
                    generated_sql = ""
    else:
        generated_sql = st.session_state.get("generated_sql", "")

# Show generated SQL in an editable box
st.subheader("SQL (edit before executing)")
sql_to_run = st.text_area("SQL", value=generated_sql or "", height=160, key="sql_area")

# Quick validation preview
is_valid, msg = validate_sql(sql_to_run, allow_writes=allow_writes)
if is_valid:
    st.info("Query looks OK (basic checks).")
else:
    st.warning(f"Validation: {msg}")

# Execute button
if st.button("Run SQL"):
    is_valid, validation_msg = validate_sql(sql_to_run, allow_writes=allow_writes)
    if not is_valid:
        st.error(f"Blocked: {validation_msg}")
    else:
        try:
            # Detect if query looks like SELECT
            first_word = re.match(r"^\s*(\w+)", sql_to_run.strip().upper()).group(1)
            if first_word in ("SELECT", "WITH", "EXPLAIN"):
                df = pd.read_sql_query(text(sql_to_run), engine)
                st.success(f"Returned {len(df)} rows.")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, file_name="query_results.csv")
            else:
                # write operation
                with engine.begin() as conn:
                    result = conn.execute(text(sql_to_run))
                    # rowcount may be -1 for some backends; show a summary
                    st.success(f"Write query executed. rowcount={getattr(result, 'rowcount', 'N/A')}")
        except Exception as e:
            st.error(f"Execution error: {e}")

# --------------- Save / Run saved queries ---------------

st.subheader("Save & reuse queries")
saved = load_saved_queries()
col_a, col_b = st.columns([3,1])
with col_a:
    new_name = st.text_input("Save current SQL under name (optional)")
with col_b:
    if st.button("Save query"):
        if not new_name:
            st.warning("Pick a name to save the current SQL.")
        elif not sql_to_run.strip():
            st.warning("No SQL to save.")
        else:
            saved[new_name] = sql_to_run
            save_saved_queries(saved)
            st.success(f"Saved as '{new_name}'.")

# List saved queries
if saved:
    st.markdown("**Saved queries**")
    chosen = st.selectbox("Choose a saved query to load/run", options=[""] + list(saved.keys()))
    if chosen:
        st.code(saved[chosen])
        if st.button("Load into editor"):
            st.session_state["sql_area"] = saved[chosen]
        if st.button("Run saved query"):
            st.session_state["sql_area"] = saved[chosen]
            st.experimental_rerun()

# --------------- Footer / next steps ---------------
st.markdown("---")
st.markdown(
    """
**Next steps / automation ideas**
- Run this app plus a tiny scheduler script (separate process) that reads `saved_queries.json` and executes queries on a schedule, saving CSVs or emailing reports.
- Connect to Postgres/MySQL by changing SQLAlchemy URL (for prod; set up proper credentials).
- Add user auth and role-based access so only safe users can run write queries.

If you want, I can:
- provide a small `scheduler.py` snippet to run saved queries on a schedule,
- adapt this to Postgres, or
- convert this into a tiny API (FastAPI) instead of Streamlit.
"""
)