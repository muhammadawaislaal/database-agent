"""
AI SQL Agent — Robust, user-friendly Streamlit app

Features:
- SQLite demo (auto-init sample DB)
- Optional Postgres/MySQL connection via SQLAlchemy URL (enter in sidebar)
- Robust extraction & cleaning of model output (handles code fences / commentary)
- Read-only (safe) toggle and Full-SQL mode (INSERT/UPDATE/DELETE allowed)
- Schema auto-discovery (inspector)
- Query history (session)
- Auto visualization for numeric results, CSV download
- Uses OpenAI Python SDK (>=1.0.0) client.chat.completions.create
"""

import os
import re
import time
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from openai import OpenAI
from datetime import datetime

# ----------------- Config & OpenAI client -----------------
st.set_page_config(page_title="AI SQL Agent", layout="wide")

# OpenAI client (Streamlit secrets preferred)
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

# ----------------- Utility functions -----------------
FORBIDDEN = {"DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"}

def init_sqlite_demo(path: str = "data.sqlite"):
    """Create a small demo SQLite DB if missing."""
    if os.path.exists(path):
        return False
    engine = create_engine(f"sqlite:///{path}")
    demo_sql = """
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
    order_date TEXT
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
    with engine.begin() as conn:
        for stmt in demo_sql.strip().split(";"):
            if stmt.strip():
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass
    engine.dispose()
    return True

def make_engine(db_type: str, sqlite_path: str = None, url: str = None):
    """Return a SQLAlchemy engine for given db_type."""
    if db_type == "SQLite (local demo)":
        p = sqlite_path or "data.sqlite"
        return create_engine(f"sqlite:///{p}", connect_args={"check_same_thread": False})
    else:
        # Expect user provided a SQLAlchemy URL for Postgres/MySQL
        if not url:
            raise ValueError("No connection URL provided.")
        return create_engine(url)

def get_schema_text(engine):
    """Return a readable schema summary using SQLAlchemy inspector."""
    try:
        insp = inspect(engine)
        tables = insp.get_table_names()
        lines = []
        for t in tables:
            try:
                cols = [c["name"] for c in insp.get_columns(t)]
            except Exception:
                cols = []
            lines.append(f"{t}: {', '.join(cols)}")
        return "\n".join(lines) if lines else "No tables found."
    except Exception as e:
        return f"Could not introspect schema: {e}"

def extract_sql(model_output: str) -> str:
    """
    Robustly extract SQL from model output:
    - If code fence (```sql ... ```), return inner content.
    - Else find first SQL keyword (SELECT/INSERT/UPDATE/DELETE/WITH/EXPLAIN)
      and return substring from that point.
    - Fallback: return stripped text.
    """
    if not model_output:
        return ""
    # 1) Look for code fence with optional "sql"
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", model_output, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) Try to find first SQL keyword
    m2 = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|EXPLAIN)\b", model_output, flags=re.IGNORECASE)
    if m2:
        start = m2.start(1)
        candidate = model_output[start:].strip()
        # Remove any trailing code fence markers
        candidate = re.sub(r"```+\s*$", "", candidate).strip()
        return candidate
    # 3) fallback to stripped text
    return model_output.strip()

def validate_sql(clean_sql: str, read_only: bool):
    """Return (ok:bool, message:str, cleaned_sql:str). cleaned_sql is normalized and trimmed."""
    if not clean_sql:
        return False, "Empty query.", ""
    sql = clean_sql.strip()
    # Uppercase copy for checks but keep original for execution
    sql_upper = sql.upper()
    if sql_upper.startswith("ERROR"):
        return False, sql, sql
    # Forbidden keywords check (word boundary)
    for bad in FORBIDDEN:
        if re.search(rf"\b{bad}\b", sql_upper):
            return False, f"Forbidden keyword detected: {bad}", sql
    # Find first meaningful keyword
    m = re.search(r"^\s*([A-Z]+)", sql_upper)
    if not m:
        return False, "Invalid SQL syntax (couldn't detect first keyword).", sql
    kw = m.group(1)
    read_allowed = {"SELECT", "WITH", "EXPLAIN"}
    full_allowed = read_allowed | {"INSERT", "UPDATE", "DELETE"}
    if read_only:
        if kw in read_allowed:
            return True, "OK", sql
        else:
            return False, f"Only read queries allowed in Read-only mode (found: {kw}).", sql
    else:
        if kw in full_allowed:
            return True, "OK", sql
        else:
            return False, f"Query type not supported (found: {kw}).", sql

def ask_model_to_sql(nl_request: str, schema_text: str, read_only: bool, prefer_full_columns: bool=True) -> (bool, str, str):
    """
    Call OpenAI to generate SQL. Returns (ok, raw_output, cleaned_sql).
    ok=False on error.
    """
    if not nl_request.strip():
        return False, "Empty natural language request.", ""
    # Provide explicit, careful instructions to the model:
    system_msg = (
        "You are an expert SQL assistant. You must respond with a single SQL statement "
        "in valid SQLite syntax (unless user DB is Postgres/MySQL, but default is SQLite). "
        "Do NOT include any explanatory text outside the SQL. If you cannot answer, reply: CANNOT_ANSWER."
    )
    # Guidance: include all columns if user asks for details
    extra = ""
    if prefer_full_columns:
        extra = (
            "If the user asks for 'details' or 'client details', include all relevant columns from the table(s). "
            "If the user requests data for a year (e.g., 2024), return rows for the entire year (all months). "
        )
    if read_only:
        extra += "Generate only read-only queries (SELECT / WITH / EXPLAIN). Do not produce INSERT/UPDATE/DELETE."
    else:
        extra += "You may generate INSERT/UPDATE/DELETE if requested, but never use DROP/ALTER/PRAGMA."
    user_msg = f"""Database schema:
{schema_text}

User request:
{nl_request}

Rules:
{extra}
Return only the SQL statement. Prefer explicit date ranges (e.g., use BETWEEN or >= / < with dates)
and include all columns when user asks for 'details'."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content
        cleaned = extract_sql(raw)
        return True, raw, cleaned
    except Exception as e:
        return False, f"OpenAI error: {e}", ""

# ----------------- UI: Sidebar (DB settings) -----------------
st.sidebar.header("Database")
db_type = st.sidebar.selectbox("DB type", ["SQLite (local demo)", "Postgres / MySQL (custom URL)"])
sqlite_path = st.sidebar.text_input("SQLite path (local)", value="data.sqlite") if db_type.startswith("SQLite") else None

if db_type != "SQLite (local demo)":
    db_url = st.sidebar.text_input("SQLAlchemy connection URL (postgresql://... or mysql+pymysql://...)", value="")
else:
    db_url = None

if db_type == "SQLite (local demo)":
    if st.sidebar.button("Init demo DB (if missing)"):
        created = init_sqlite_demo(sqlite_path)
        if created:
            st.sidebar.success(f"Demo DB created at {sqlite_path}")
        else:
            st.sidebar.info(f"DB already exists at {sqlite_path}")

# Connect button
connect = st.sidebar.button("Connect / Refresh schema")

# Create engine (on-demand)
engine = None
try:
    if db_type == "SQLite (local demo)":
        engine = make_engine(db_type, sqlite_path=sqlite_path)
    else:
        if db_url:
            engine = make_engine(db_type, url=db_url)
        else:
            engine = None
except Exception as e:
    st.sidebar.error(f"Connection error: {e}")
    engine = None

# If the user asked to refresh or connected URL changed, re-introspect
if connect and engine is None:
    st.sidebar.error("No connection available. Provide a valid connection URL or init the demo DB.")

# ----------------- Main UI -----------------
st.title("🤖 AI SQL Agent")
st.caption("Ask plain-English questions. The agent generates SQL, you review, then run. Read-only mode is ON by default.")

# Get schema_text for prompt
if engine:
    schema_text = get_schema_text(engine)
else:
    schema_text = "No DB connected."

with st.expander("📚 Database schema (auto-discovered)"):
    st.text(schema_text)

# Mode toggle
read_only = st.checkbox("🔒 Read-only Mode (safe)", value=True)
if not read_only:
    st.warning("Full mode enabled: INSERT/UPDATE/DELETE allowed. Use carefully.")

# Natural language input
nl_input = st.text_area("📝 Ask in plain English (example: 'Show client details for 2024')", height=120)

# Generate SQL
col1, col2 = st.columns([1,1])
with col1:
    if st.button("✨ Generate SQL"):
        if not engine:
            st.error("No DB connected. Initialize demo DB or provide a connection URL in the sidebar.")
        else:
            with st.spinner("Generating SQL..."):
                ok, raw, cleaned = ask_model_to_sql(nl_input, schema_text, read_only, prefer_full_columns=True)
                if not ok:
                    st.error(raw)
                else:
                    # store raw and cleaned in session for editing/execution
                    st.session_state["ai_raw"] = raw
                    st.session_state["ai_sql"] = cleaned
                    st.success("SQL generated. Review/edit before running.")
with col2:
    # show raw AI output toggle
    show_raw = st.checkbox("Show raw AI output (for debugging)", value=False)

if show_raw and "ai_raw" in st.session_state:
    st.expander("Raw AI output", expanded=True).write(st.session_state.get("ai_raw", ""))

# SQL editor (cleaned)
sql_editor_initial = st.session_state.get("ai_sql", "")
sql_to_run = st.text_area("📋 SQL Query (editable)", value=sql_editor_initial, height=180, key="sql_editor")

# Run SQL
if st.button("▶️ Run SQL"):
    if not engine:
        st.error("No DB connection available.")
    else:
        cleaned_sql = extract_sql(sql_to_run)
        ok, msg, norm_sql = validate_sql(cleaned_sql, read_only)
        if not ok:
            st.error(msg)
            # helpful hint: show cleaned SQL for debugging
            st.info(f"Cleaned SQL preview:\n\n{cleaned_sql}")
        else:
            try:
                # SELECT → use pandas read_sql_query (returns DataFrame)
                first_kw_match = re.search(r"^\s*([A-Z]+)", norm_sql.strip().upper())
                first_kw = first_kw_match.group(1) if first_kw_match else "SELECT"
                if first_kw in ("SELECT", "WITH", "EXPLAIN"):
                    df = pd.read_sql_query(text(norm_sql), engine)
                    st.success(f"Returned {len(df)} rows.")
                    st.dataframe(df, use_container_width=True)

                    # auto visualization: if at least one numeric column exists, offer chart
                    numeric_cols = df.select_dtypes(include="number").columns.tolist()
                    if numeric_cols and df.shape[0] > 0:
                        st.markdown("### 📊 Quick visualization")
                        # choose index column (prefer first non-numeric) for x-axis
                        idx_col = None
                        for c in df.columns:
                            if c not in numeric_cols:
                                idx_col = c
                                break
                        if idx_col is None:
                            idx_col = df.columns[0]
                        # default chart: bar chart of first numeric column
                        chart_col = st.selectbox("Choose numeric column to chart", numeric_cols, index=0)
                        st.bar_chart(data=df.set_index(idx_col)[chart_col])
                    # CSV download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download CSV", csv, "results.csv", "text/csv")

                    # append to history
                    history = st.session_state.get("history", [])
                    history.append({
                        "ts": datetime.utcnow().isoformat(),
                        "query": norm_sql,
                        "rows": len(df)
                    })
                    st.session_state["history"] = history
                else:
                    # write operation
                    with engine.begin() as conn:
                        conn.execute(text(norm_sql))
                    st.success("Write query executed successfully.")
                    # record history entry with rows = -1 for non-select
                    history = st.session_state.get("history", [])
                    history.append({
                        "ts": datetime.utcnow().isoformat(),
                        "query": norm_sql,
                        "rows": -1
                    })
                    st.session_state["history"] = history
            except Exception as e:
                st.error(f"Execution error: {e}")
                st.info(f"SQL sent for execution:\n\n{norm_sql}")

# Query history panel
if st.session_state.get("history"):
    with st.expander("🕒 Query History (session) - click to load"):
        hist = st.session_state["history"]
        # show latest first
        for i, entry in enumerate(reversed(hist[-20:]), 1):
            ts = entry["ts"]
            q = entry["query"]
            rows = entry["rows"]
            col1, col2 = st.columns([8,2])
            with col1:
                st.code(q, language="sql")
                st.write(f"→ rows: {rows if rows>=0 else 'write op'} • {ts}")
            with col2:
                if st.button(f"Load #{i}", key=f"load_{i}"):
                    st.session_state["ai_sql"] = q
                    st.experimental_rerun()

# Footer tips
st.markdown("---")
st.markdown(
    "- Tip: If the agent returns partial columns, edit the query to include `*` or the desired columns. "
    "- If you ask for 'details' the agent is instructed to include all relevant columns. "
    "- For external DBs (Postgres/MySQL), provide a SQLAlchemy URL in the sidebar and press Connect."
)
