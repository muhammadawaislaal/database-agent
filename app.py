"""
AI SQL Agent — Robust, user-friendly Streamlit app (with pivot features + SQL auto-fix)

Features:
- SQLite demo (auto-init sample DB)
- Optional Postgres/MySQL connection via SQLAlchemy URL (enter in sidebar)
- Robust extraction & cleaning of model output (handles code fences / commentary)
- Read-only (safe) toggle and Full-SQL mode (INSERT/UPDATE/DELETE allowed)
- Schema auto-discovery (inspector)
- Query history (session)
- Auto visualization for numeric results, CSV download
- Yearly / Monthly / Weekly pivoting when user requests (12 months / days / weeks)
- OpenAI prompt instructs: do NOT use AS in GROUP BY/ORDER BY
- Auto-fix: removes invalid "AS alias" from GROUP BY/ORDER BY before execution
"""

import os
import re
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from openai import OpenAI
from datetime import datetime
from typing import Tuple, Dict, Optional, List

# ----------------- Config & OpenAI client -----------------
st.set_page_config(page_title="AI SQL Agent", layout="wide")

if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    st.error("No OpenAI API key found. Add OPENAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

# ----------------- Utility / safety -----------------
FORBIDDEN = {"DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "PRAGMA"}

def fix_group_order_aliases(sql: str) -> str:
    """
    Remove invalid 'AS alias' inside GROUP BY and ORDER BY clauses produced by the model.
    e.g.
      GROUP BY orders.product_id AS product_id  -> GROUP BY orders.product_id
      ORDER BY product_id AS product_id -> ORDER BY product_id
    Also handles multiple columns: GROUP BY a AS a, b AS b -> GROUP BY a, b
    """
    if not sql:
        return sql
    # Normalize whitespace for reliable regex
    # Handle GROUP BY clause
    def _clean_clause(match):
        clause = match.group(0)
        # remove occurrences of "AS alias" (alias being identifier possibly quoted)
        cleaned = re.sub(r"\s+AS\s+[`\"']?([A-Za-z0-9_]+)[`\"']?", r"", clause, flags=re.IGNORECASE)
        # also remove duplicate alias after comma segments: ", col AS col" -> ", col"
        cleaned = re.sub(r",\s*([A-Za-z0-9_.]+)\s*$", r", \1", cleaned)
        return cleaned

    # Replace in GROUP BY
    sql = re.sub(r"GROUP\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)
    # Replace in ORDER BY
    sql = re.sub(r"ORDER\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)

    # Additionally remove trailing duplicate alias in ORDER BY like "ORDER BY x AS x;"
    sql = re.sub(r"\b([A-Za-z0-9_.]+)\s+AS\s+\1\b", r"\1", sql, flags=re.IGNORECASE)

    return sql

# ----------------- DB helpers -----------------
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
    if db_type == "SQLite (local demo)":
        p = sqlite_path or "data.sqlite"
        return create_engine(f"sqlite:///{p}", connect_args={"check_same_thread": False})
    else:
        if not url:
            raise ValueError("No connection URL provided.")
        return create_engine(url)

def get_schema_text(engine) -> str:
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

def get_schema_dict(engine) -> Dict[str, list]:
    try:
        insp = inspect(engine)
        schema = {}
        for t in insp.get_table_names():
            try:
                schema[t] = [c["name"] for c in insp.get_columns(t)]
            except Exception:
                schema[t] = []
        return schema
    except Exception:
        return {}

# ----------------- SQL extraction / validation -----------------
def extract_sql(model_output: str) -> str:
    if not model_output:
        return ""
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", model_output, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|EXPLAIN)\b", model_output, flags=re.IGNORECASE)
    if m2:
        start = m2.start(1)
        candidate = model_output[start:].strip()
        candidate = re.sub(r"```+\s*$", "", candidate).strip()
        return candidate
    return model_output.strip()

def validate_sql(clean_sql: str, read_only: bool) -> Tuple[bool,str,str]:
    if not clean_sql:
        return False, "Empty query.", ""
    sql = clean_sql.strip()
    sql_upper = sql.upper()
    if sql_upper.startswith("ERROR"):
        return False, sql, sql
    for bad in FORBIDDEN:
        if re.search(rf"\b{bad}\b", sql_upper):
            return False, f"Forbidden keyword detected: {bad}", sql
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

# ----------------- Pivot helpers -----------------
def infer_date_and_entity(schema: Dict[str, list]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if "orders" in schema:
        cols = schema["orders"]
        date_col = next((c for c in cols if "date" in c.lower()), None)
        if "products" in schema and "name" in schema["products"]:
            return "orders", date_col or "order_date", "product_id"
        if "customers" in schema and "name" in schema["customers"]:
            return "orders", date_col or "order_date", "customer_id"
        if "clients" in schema and "name" in schema["clients"]:
            return "orders", date_col or "order_date", "client_id"
        if "product_id" in cols:
            return "orders", date_col or "order_date", "product_id"
        if "customer_id" in cols:
            return "orders", date_col or "order_date", "customer_id"
        if "client_id" in cols:
            return "orders", date_col or "order_date", "client_id"
    
    # Also check for clients table directly
    if "clients" in schema:
        cols = schema["clients"]
        date_col = next((c for c in cols if "date" in c.lower()), None)
        entity = next((c for c in cols if any(k in c.lower() for k in ("name","client","customer"))), None)
        return "clients", date_col, entity
        
    for table, cols in schema.items():
        date_col = next((c for c in cols if "date" in c.lower()), None)
        if date_col:
            entity = next((c for c in cols if any(k in c.lower() for k in ("name","client","customer","product"))), None)
            return table, date_col, entity
    return None, None, None

def build_year_pivot_sql(table: str, date_col: str, entity_col: str, year: str, schema: Dict[str, list]) -> str:
    join_clause = ""
    if entity_col and entity_col.endswith("_id"):
        base = entity_col[:-3]
        # Handle client_id -> customers mapping
        if base == "client":
            base = "customer"
            entity_col = "customer_id"
            
        if base in schema and "name" in schema[base]:
            join_clause = f"LEFT JOIN {base} ON {table}.{entity_col} = {base}.id"
            select_entity = f"{base}.name AS {base}_name"
            group_by_entity = f"{base}.name"
            order_by_entity = f"{base}.name"
        else:
            select_entity = f"{table}.{entity_col} AS {entity_col}"
            group_by_entity = f"{table}.{entity_col}"
            order_by_entity = group_by_entity
    else:
        select_entity = f"{table}.{entity_col}" if entity_col else "rowid"
        group_by_entity = select_entity
        order_by_entity = select_entity

    month_cases = []
    for m in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
        alias = datetime.strptime(m, "%m").strftime("%b")
        # aggregate qty if exists, else count rows
        month_cases.append(f"SUM(CASE WHEN strftime('%m', {table}.{date_col}) = '{m}' THEN COALESCE({table}.qty,1) ELSE 0 END) AS \"{alias}\"")
    cases_sql = ",\n    ".join(month_cases)
    sql = f"""SELECT
    {select_entity},
    {cases_sql}
FROM {table}
{join_clause}
WHERE strftime('%Y', {table}.{date_col}) = '{year}'
GROUP BY {group_by_entity}
ORDER BY {order_by_entity};"""
    return sql

def build_month_pivot_sql(table: str, date_col: str, entity_col: str, year: str, month: str, days: int, schema: Dict[str, list]) -> str:
    join_clause = ""
    if entity_col and entity_col.endswith("_id"):
        base = entity_col[:-3]
        # Handle client_id -> customers mapping
        if base == "client":
            base = "customer"
            entity_col = "customer_id"
            
        if base in schema and "name" in schema[base]:
            join_clause = f"LEFT JOIN {base} ON {table}.{entity_col} = {base}.id"
            select_entity = f"{base}.name AS {base}_name"
            group_by_entity = f"{base}.name"
            order_by_entity = f"{base}.name"
        else:
            select_entity = f"{table}.{entity_col} AS {entity_col}"
            group_by_entity = f"{table}.{entity_col}"
            order_by_entity = group_by_entity
    else:
        select_entity = f"{table}.{entity_col}" if entity_col else "rowid"
        group_by_entity = select_entity
        order_by_entity = select_entity

    day_cases = []
    for d in range(1, days+1):
        dd = f"{d:02d}"
        day_cases.append(f"SUM(CASE WHEN strftime('%d', {table}.{date_col}) = '{dd}' THEN COALESCE({table}.qty,1) ELSE 0 END) AS \"Day{d}\"")
    cases_sql = ",\n    ".join(day_cases)
    sql = f"""SELECT
    {select_entity},
    {cases_sql}
FROM {table}
{join_clause}
WHERE strftime('%Y-%m', {table}.{date_col}) = '{year}-{month}'
GROUP BY {group_by_entity}
ORDER BY {order_by_entity};"""
    return sql

def build_week_pivot_sql(table: str, date_col: str, entity_col: str, year: str, schema: Dict[str, list]) -> str:
    join_clause = ""
    if entity_col and entity_col.endswith("_id"):
        base = entity_col[:-3]
        # Handle client_id -> customers mapping
        if base == "client":
            base = "customer"
            entity_col = "customer_id"
            
        if base in schema and "name" in schema[base]:
            join_clause = f"LEFT JOIN {base} ON {table}.{entity_col} = {base}.id"
            select_entity = f"{base}.name AS {base}_name"
            group_by_entity = f"{base}.name"
            order_by_entity = f"{base}.name"
        else:
            select_entity = f"{table}.{entity_col} AS {entity_col}"
            group_by_entity = f"{table}.{entity_col}"
            order_by_entity = group_by_entity
    else:
        select_entity = f"{table}.{entity_col}" if entity_col else "rowid"
        group_by_entity = select_entity
        order_by_entity = select_entity

    week_cases = []
    for w in range(0,54):
        alias = f"W{w:02d}"
        week_cases.append(f"SUM(CASE WHEN strftime('%Y', {table}.{date_col}) = '{year}' AND strftime('%W', {table}.{date_col}) = '{w:02d}' THEN COALESCE({table}.qty,1) ELSE 0 END) AS \"{alias}\"")
    cases_sql = ",\n    ".join(week_cases)
    sql = f"""SELECT
    {select_entity},
    {cases_sql}
FROM {table}
{join_clause}
WHERE strftime('%Y', {table}.{date_col}) = '{year}'
GROUP BY {group_by_entity}
ORDER BY {order_by_entity};"""
    return sql

# ----------------- Model prompt / SQL generation -----------------
def ask_model_to_sql_llm(nl_request: str, schema_text: str, read_only: bool, prefer_full_columns: bool=True) -> Tuple[bool,str,str]:
    """
    Use the LLM to generate SQL. Returns (ok, raw_output, cleaned_sql).
    The prompt includes a strict rule: DO NOT use 'AS' inside GROUP BY / ORDER BY.
    """
    if not nl_request.strip():
        return False, "Empty natural language request.", ""
    
    # Add schema mapping instructions
    schema_notes = ""
    if "customers" in schema_text and "clients" not in schema_text.lower():
        schema_notes = "IMPORTANT: When user says 'clients', they mean the 'customers' table. When they say 'client_id', they mean 'customer_id'. When they say 'registration_date', they mean 'signup_date'."
    
    system_msg = (
        "You are an expert SQL assistant. Respond with a single SQL statement only (no explanation). "
        "Default dialect: SQLite. IMPORTANT RULE: Never use 'AS' inside GROUP BY or ORDER BY clauses. "
        "When grouping or ordering, reference the column or alias directly (e.g., GROUP BY product_id). "
        f"{schema_notes}"
    )
    extra = ""
    if prefer_full_columns:
        extra = "If user asks for 'details', return all relevant columns rather than a small sample. "
    if read_only:
        extra += "Generate only read-only queries (SELECT/WITH/EXPLAIN). Do not produce INSERT/UPDATE/DELETE."
    else:
        extra += "INSERT/UPDATE/DELETE are allowed if requested, but never DROP/ALTER/PRAGMA."
    user_msg = f"""Database schema:
{schema_text}

User request:
{nl_request}

Rules:
{extra}
Return only the SQL statement. If you include GROUP BY or ORDER BY, do not use 'AS' inside them.
Prefer explicit date ranges and include all columns when user asks for 'details'."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg}
            ],
            temperature=0,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content
        cleaned = extract_sql(raw)
        return True, raw, cleaned
    except Exception as e:
        return False, f"OpenAI error: {e}", ""

# ----------------- UI Sidebar -----------------
st.sidebar.header("Database")
db_type = st.sidebar.selectbox("DB type", ["SQLite (local demo)", "Postgres / MySQL (custom URL)"])
sqlite_path = st.sidebar.text_input("SQLite path (local)", value="data.sqlite") if db_type.startswith("SQLite") else None
db_url = st.sidebar.text_input("SQLAlchemy connection URL (postgresql://... or mysql+pymysql://...)", value="") if db_type != "SQLite (local demo)" else ""

if db_type == "SQLite (local demo)":
    if st.sidebar.button("Init demo DB (if missing)"):
        created = init_sqlite_demo(sqlite_path)
        if created:
            st.sidebar.success(f"Demo DB created at {sqlite_path}")
        else:
            st.sidebar.info(f"DB already exists at {sqlite_path}")

connect = st.sidebar.button("Connect / Refresh schema")

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

if connect and engine is None:
    st.sidebar.error("No connection available. Provide a valid connection URL or init the demo DB.")

# ----------------- Main UI -----------------
st.title("🤖 AI SQL Agent")
st.caption("Ask plain-English questions. The agent generates SQL, you review, then run. Read-only mode is ON by default.")

if engine:
    schema_text = get_schema_text(engine)
    schema_dict = get_schema_dict(engine)
else:
    schema_text = "No DB connected."
    schema_dict = {}

with st.expander("📚 Database schema (auto-discovered)"):
    st.text(schema_text)

read_only = st.checkbox("🔒 Read-only Mode (safe)", value=True)
if not read_only:
    st.warning("Full mode enabled: INSERT/UPDATE/DELETE allowed. Use carefully.")

nl_input = st.text_area("📝 Ask in plain English (e.g. 'Show client details for 2024' or 'Show daily details for Aug 2024')", height=120)

def detect_pivot_intent(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not text:
        return None, None, None
    t = text.lower()
    y_match = re.search(r"\b(20\d{2})\b", t)
    mon_match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s*(20\d{2})?", t)
    y_m = re.search(r"\b(20\d{2})[-/](0[1-9]|1[0-2])\b", t)
    
    # Weekly detection
    if ("weekly" in t or "per week" in t or "by week" in t or ("week" in t and "day" not in t)):
        year = y_match.group(1) if y_match else datetime.utcnow().strftime("%Y")
        return "week", year, None
    
    # Year-month format detection (YYYY-MM)
    if y_m:
        year = y_m.group(1)
        month = y_m.group(2)
        return "month", year, month
    
    # Month name detection
    if mon_match:
        mon_str = mon_match.group(1)
        year = mon_match.group(2) if mon_match.group(2) else (y_match.group(1) if y_match else datetime.utcnow().strftime("%Y"))
        month_map = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06","jul":"07","aug":"08","sep":"09","sept":"09","oct":"10","nov":"11","dec":"12"}
        monnum = month_map.get(mon_str[:3], None)
        if monnum:
            return "month", year, monnum
    
    # Year detection
    if y_match and any(k in t for k in ("year","yearly","for","in")):
        return "year", y_match.group(1), None
    
    if y_match and ("for " + y_match.group(1) in t or "in " + y_match.group(1) in t):
        return "year", y_match.group(1), None
    
    return None, None, None

col1, col2 = st.columns([1,1])
with col1:
    if st.button("✨ Generate SQL"):
        if not engine:
            st.error("No DB connected. Initialize demo DB or provide a connection URL in the sidebar.")
        else:
            with st.spinner("Generating SQL..."):
                intent, yorn, mon = detect_pivot_intent(nl_input)
                fact_table, date_col, entity_col = infer_date_and_entity(schema_dict)
                
                # FIX: Check if entity_col is None and provide a default
                if entity_col is None:
                    entity_col = "id"  # Default to 'id' if no entity column found
                
                if not fact_table:
                    ok, raw, cleaned = ask_model_to_sql_llm(nl_input, schema_text, read_only, prefer_full_columns=True)
                else:
                    if intent == "year":
                        sql_pivot = build_year_pivot_sql(fact_table, date_col, entity_col, yorn, schema_dict)
                        ok, raw, cleaned = True, sql_pivot, sql_pivot
                    elif intent == "month":
                        y = yorn
                        m = mon
                        mdays = {"01":31,"02":29,"03":31,"04":30,"05":31,"06":30,"07":31,"08":31,"09":30,"10":31,"11":30,"12":31}
                        days = mdays.get(m,31)
                        sql_pivot = build_month_pivot_sql(fact_table, date_col, entity_col, y, m, days, schema_dict)
                        ok, raw, cleaned = True, sql_pivot, sql_pivot
                    elif intent == "week":
                        y = yorn
                        sql_pivot = build_week_pivot_sql(fact_table, date_col, entity_col, y, schema_dict)
                        ok, raw, cleaned = True, sql_pivot, sql_pivot
                    else:
                        ok, raw, cleaned = ask_model_to_sql_llm(nl_input, schema_text, read_only, prefer_full_columns=True)
                if not ok:
                    st.error(raw)
                else:
                    st.session_state["ai_raw"] = raw
                    st.session_state["ai_sql"] = cleaned
                    st.success("SQL generated. Review/edit before running.")
with col2:
    show_raw = st.checkbox("Show raw AI output (for debugging)", value=False)

if show_raw and "ai_raw" in st.session_state:
    st.expander("Raw AI output", expanded=True).write(st.session_state.get("ai_raw",""))

sql_editor_initial = st.session_state.get("ai_sql","")
sql_to_run = st.text_area("📋 SQL Query (editable)", value=sql_editor_initial, height=220, key="sql_editor")

if st.button("▶️ Run SQL"):
    if not engine:
        st.error("No DB connection available.")
    else:
        cleaned_sql = extract_sql(sql_to_run)
        ok, msg, norm_sql = validate_sql(cleaned_sql, read_only)
        if not ok:
            st.error(msg)
            st.info(f"Cleaned SQL preview:\n\n{cleaned_sql}")
        else:
            # Auto-fix GROUP/ORDER AS alias mistakes before execution
            norm_sql_fixed = fix_group_order_aliases(norm_sql)
            try:
                first_kw_match = re.search(r"^\s*([A-Z]+)", norm_sql_fixed.strip().upper())
                first_kw = first_kw_match.group(1) if first_kw_match else "SELECT"
                if first_kw in ("SELECT","WITH","EXPLAIN"):
                    try:
                        df = pd.read_sql_query(text(norm_sql_fixed), engine)
                        st.success(f"Returned {len(df)} rows.")
                        st.dataframe(df, use_container_width=True)
                        numeric_cols = df.select_dtypes(include="number").columns.tolist()
                        if numeric_cols and df.shape[0] > 0:
                            st.markdown("### 📊 Quick visualization")
                            idx_col = None
                            for c in df.columns:
                                if c not in numeric_cols:
                                    idx_col = c
                                    break
                            if idx_col is None: 
                                idx_col = df.columns[0]
                            chart_col = st.selectbox("Choose numeric column to chart", numeric_cols, index=0)
                            st.bar_chart(data=df.set_index(idx_col)[chart_col])
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("💾 Download CSV", csv, "results.csv", "text/csv")
                        history = st.session_state.get("history", [])
                        history.append({"ts": datetime.utcnow().isoformat(), "query": norm_sql_fixed, "rows": len(df)})
                        st.session_state["history"] = history
                    except Exception as e:
                        st.error(f"SQL execution error: {e}")
                        st.info(f"SQL sent for execution (fixed):\n\n{norm_sql_fixed}")
                else:
                    with engine.begin() as conn:
                        conn.execute(text(norm_sql_fixed))
                    st.success("Write query executed successfully.")
                    history = st.session_state.get("history", [])
                    history.append({"ts": datetime.utcnow().isoformat(), "query": norm_sql_fixed, "rows": -1})
                    st.session_state["history"] = history
            except Exception as e:
                st.error(f"Execution error: {e}")
                st.info(f"SQL sent for execution (fixed):\n\n{norm_sql_fixed}")

if st.session_state.get("history"):
    with st.expander("🕒 Query History (session) - click to load"):
        hist = st.session_state["history"]
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
                    st.rerun()

st.markdown("---")
st.markdown(
    "- Tip: Ask 'Show client details for 2024' → returns a 12-month pivot (Jan..Dec). "
    "- Ask 'Show client daily details for Aug 2024' → returns Day1..Day31 columns. "
    "- Ask 'Show client weekly details for 2024' → returns week columns W00..W53. "
    "- For external DBs (Postgres/MySQL), provide a SQLAlchemy URL in the sidebar and press Connect."
)
