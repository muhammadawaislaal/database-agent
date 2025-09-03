"""
AI SQL Agent — Robust, user-friendly Streamlit app (with pivot features + SQL auto-fix)
Fixed database connection and initialization issues
"""

import os
import re
import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine, text, inspect
from openai import OpenAI
from datetime import datetime
from typing import Tuple, Dict, Optional, List
import tempfile

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
    """
    if not sql:
        return sql
    
    def _clean_clause(match):
        clause = match.group(0)
        cleaned = re.sub(r"\s+AS\s+[`\"']?([A-Za-z0-9_]+)[`\"']?", r"", clause, flags=re.IGNORECASE)
        cleaned = re.sub(r",\s*([A-Za-z0-9_.]+)\s*$", r", \1", cleaned)
        return cleaned

    sql = re.sub(r"GROUP\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)
    sql = re.sub(r"ORDER\s+BY\s+[^\n;]+", lambda m: _clean_clause(m), sql, flags=re.IGNORECASE)
    sql = re.sub(r"\b([A-Za-z0-9_.]+)\s+AS\s+\1\b", r"\1", sql, flags=re.IGNORECASE)

    return sql

# ----------------- DB helpers -----------------
def init_sqlite_demo(path: str = "demo_db.sqlite"):
    """Create a small demo SQLite DB if missing."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Drop tables if they exist to start fresh
    cursor.execute("DROP TABLE IF EXISTS orders")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS customers")
    
    # Create tables
    demo_sql = """
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
    FOREIGN KEY (customer_id) REFERENCES customers (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
);

INSERT INTO customers (name, email, signup_date) VALUES
  ('Alice Smith', 'alice@example.com', '2024-01-15'),
  ('Bob Johnson', 'bob@example.com', '2024-02-20'),
  ('Carol Williams', 'carol@example.com', '2024-03-10'),
  ('David Brown', 'david@example.com', '2024-04-05'),
  ('Eva Davis', 'eva@example.com', '2024-05-12');

INSERT INTO products (name, category, price) VALUES
  ('Widget A', 'Widgets', 9.99),
  ('Widget B', 'Widgets', 19.99),
  ('Gadget X', 'Gadgets', 29.50),
  ('Gadget Y', 'Gadgets', 39.99),
  ('Tool Z', 'Tools', 15.00);

INSERT INTO orders (customer_id, product_id, qty, order_date) VALUES
  (1, 1, 3, '2024-06-01'),
  (1, 3, 1, '2024-06-15'),
  (2, 2, 2, '2024-06-20'),
  (2, 4, 1, '2024-07-05'),
  (3, 1, 1, '2024-07-10'),
  (3, 5, 2, '2024-07-15'),
  (4, 3, 1, '2024-08-01'),
  (4, 2, 3, '2024-08-10'),
  (5, 4, 2, '2024-08-20'),
  (5, 1, 1, '2024-09-01'),
  (1, 5, 1, '2024-09-05'),
  (2, 3, 2, '2024-09-10');
"""
    
    try:
        cursor.executescript(demo_sql)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error creating demo database: {e}")
        return False
    finally:
        conn.close()

def make_engine(db_type: str, sqlite_path: str = None, url: str = None):
    if db_type == "SQLite (local demo)":
        if not sqlite_path:
            sqlite_path = "demo_db.sqlite"
        return create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
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
                cols = [f"{c['name']} ({c['type']})" for c in insp.get_columns(t)]
                lines.append(f"📊 {t}: {', '.join(cols)}")
            except Exception:
                cols = []
                lines.append(f"📊 {t}: Could not get columns")
        return "\n\n".join(lines) if lines else "No tables found."
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
    # Remove code fences
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", model_output, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    # Find SQL keywords
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
    # First check for orders table (most common)
    if "orders" in schema:
        cols = schema["orders"]
        date_col = next((c for c in cols if "date" in c.lower()), "order_date")
        
        # Check for customer references
        if "customer_id" in cols:
            return "orders", date_col, "customer_id"
        elif "product_id" in cols:
            return "orders", date_col, "product_id"
    
    # Check for other tables with date columns
    for table, cols in schema.items():
        date_col = next((c for c in cols if "date" in c.lower()), None)
        if date_col:
            # Look for entity columns
            entity_col = next((c for c in cols if any(keyword in c.lower() for keyword in 
                                                     ["name", "id", "product", "customer", "client", "user"])), None)
            return table, date_col, entity_col
    
    return None, None, None

def build_year_pivot_sql(table: str, date_col: str, entity_col: str, year: str, schema: Dict[str, list]) -> str:
    # Handle entity column mapping
    if entity_col and entity_col.endswith("_id"):
        base_table = entity_col.replace("_id", "")
        if base_table in schema:
            join_clause = f"LEFT JOIN {base_table} ON {table}.{entity_col} = {base_table}.id"
            select_entity = f"{base_table}.name AS {base_table}_name"
            group_by_entity = f"{base_table}.name"
        else:
            join_clause = ""
            select_entity = f"{table}.{entity_col}"
            group_by_entity = f"{table}.{entity_col}"
    else:
        join_clause = ""
        select_entity = f"{table}.{entity_col}" if entity_col else "1"
        group_by_entity = select_entity

    # Build monthly cases
    month_cases = []
    for m in range(1, 13):
        month_str = f"{m:02d}"
        month_name = datetime(2024, m, 1).strftime("%b")
        month_cases.append(f"SUM(CASE WHEN strftime('%m', {table}.{date_col}) = '{month_str}' THEN COALESCE({table}.qty, 1) ELSE 0 END) AS \"{month_name}\"")

    cases_sql = ",\n    ".join(month_cases)
    
    sql = f"""SELECT
    {select_entity},
    {cases_sql}
FROM {table}
{join_clause}
WHERE strftime('%Y', {table}.{date_col}) = '{year}'
GROUP BY {group_by_entity}
ORDER BY {group_by_entity};"""
    
    return sql

# ----------------- Model prompt / SQL generation -----------------
def ask_model_to_sql_llm(nl_request: str, schema_text: str, read_only: bool, prefer_full_columns: bool = True) -> Tuple[bool, str, str]:
    """
    Use the LLM to generate SQL. Returns (ok, raw_output, cleaned_sql).
    """
    if not nl_request.strip():
        return False, "Empty natural language request.", ""

    system_msg = """You are an expert SQL assistant. Respond with a single SQL statement only (no explanation). 
Default dialect: SQLite. IMPORTANT RULES:
1. Never use 'AS' inside GROUP BY or ORDER BY clauses
2. Use explicit table.column references when needed
3. When user says 'clients', use 'customers' table
4. When user says 'client_id', use 'customer_id' column
5. Use proper date formatting for SQLite: strftime() functions"""

    user_msg = f"""Database schema:
{schema_text}

User request: {nl_request}

Generate a clean SQL query that answers the user's request. 
Return only the SQL statement without any explanations or code fences."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content
        cleaned = extract_sql(raw)
        return True, raw, cleaned
    except Exception as e:
        return False, f"OpenAI error: {e}", ""

# ----------------- UI Setup -----------------
st.sidebar.header("🔧 Database Configuration")
db_type = st.sidebar.selectbox("Database Type", ["SQLite (local demo)", "Postgres / MySQL (custom URL)"])

# Initialize session state
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
if "engine" not in st.session_state:
    st.session_state.engine = None
if "schema_text" not in st.session_state:
    st.session_state.schema_text = "No DB connected."
if "schema_dict" not in st.session_state:
    st.session_state.schema_dict = {}

# Database connection setup
if db_type == "SQLite (local demo)":
    sqlite_path = st.sidebar.text_input("SQLite Database Path", value="demo_db.sqlite")
    
    if st.sidebar.button("🚀 Initialize Demo Database", help="Create sample database with demo data"):
        with st.spinner("Creating demo database..."):
            success = init_sqlite_demo(sqlite_path)
            if success:
                st.sidebar.success("✅ Demo database created successfully!")
                st.session_state.db_initialized = True
            else:
                st.sidebar.error("❌ Failed to create demo database")

    if st.sidebar.button("🔗 Connect to Database"):
        try:
            engine = make_engine(db_type, sqlite_path=sqlite_path)
            st.session_state.engine = engine
            st.session_state.schema_text = get_schema_text(engine)
            st.session_state.schema_dict = get_schema_dict(engine)
            st.sidebar.success("✅ Connected to database!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

else:  # External database
    db_url = st.sidebar.text_input("Database URL", 
                                  placeholder="postgresql://user:pass@host:port/dbname or mysql+pymysql://...")
    if st.sidebar.button("🔗 Connect to External Database") and db_url:
        try:
            engine = make_engine(db_type, url=db_url)
            st.session_state.engine = engine
            st.session_state.schema_text = get_schema_text(engine)
            st.session_state.schema_dict = get_schema_dict(engine)
            st.sidebar.success("✅ Connected to external database!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

# ----------------- Main UI -----------------
st.title("🤖 AI SQL Agent")
st.caption("Ask questions in plain English and get SQL results instantly!")

# Display schema
with st.expander("📊 Database Schema", expanded=True):
    st.code(st.session_state.schema_text)

# Read-only mode
read_only = st.checkbox("🔒 Read-only Mode", value=True, 
                       help="Prevents INSERT/UPDATE/DELETE operations for safety")
if not read_only:
    st.warning("⚠️ Full access mode enabled. Use with caution!")

# Query input
nl_input = st.text_area(
    "💬 Ask your question", 
    placeholder="e.g., 'Show me all customers', 'Monthly sales for 2024', 'Products by category'",
    height=100
)

# Action buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("✨ Generate SQL", use_container_width=True):
        if not st.session_state.engine:
            st.error("❌ Please connect to a database first!")
        elif not nl_input.strip():
            st.error("❌ Please enter a question!")
        else:
            with st.spinner("Generating SQL query..."):
                ok, raw, cleaned = ask_model_to_sql_llm(
                    nl_input, 
                    st.session_state.schema_text, 
                    read_only
                )
                if ok:
                    st.session_state.ai_raw = raw
                    st.session_state.ai_sql = cleaned
                    st.success("✅ SQL generated successfully!")
                else:
                    st.error(f"❌ {raw}")

with col2:
    if st.button("▶️ Run SQL", use_container_width=True):
        if not st.session_state.engine:
            st.error("❌ Please connect to a database first!")
        else:
            sql_to_run = st.session_state.get("ai_sql", "")
            if not sql_to_run.strip():
                st.error("❌ No SQL query to execute!")
            else:
                with st.spinner("Executing query..."):
                    cleaned_sql = extract_sql(sql_to_run)
                    ok, msg, norm_sql = validate_sql(cleaned_sql, read_only)
                    
                    if not ok:
                        st.error(f"❌ Validation failed: {msg}")
                    else:
                        norm_sql_fixed = fix_group_order_aliases(norm_sql)
                        try:
                            df = pd.read_sql_query(text(norm_sql_fixed), st.session_state.engine)
                            st.success(f"✅ Query executed successfully! Returned {len(df)} rows.")
                            
                            # Display results
                            st.dataframe(df, use_container_width=True)
                            
                            # Add to history
                            if "history" not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append({
                                "timestamp": datetime.now().isoformat(),
                                "query": norm_sql_fixed,
                                "rows": len(df)
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Execution failed: {e}")
                            st.code(norm_sql_fixed, language="sql")

# Display raw AI output if available
if st.session_state.get("ai_raw"):
    with st.expander("📝 Raw AI Output", expanded=False):
        st.text(st.session_state.ai_raw)

# SQL editor
sql_editor = st.text_area(
    "📋 SQL Query Editor", 
    value=st.session_state.get("ai_sql", "SELECT * FROM customers LIMIT 5;"),
    height=200,
    help="Edit the generated SQL or write your own query"
)

# Query history
if st.session_state.get("history"):
    with st.expander("🕒 Query History", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            cols = st.columns([4, 1])
            with cols[0]:
                st.code(entry["query"], language="sql")
                st.caption(f"Rows: {entry['rows']} • {entry['timestamp'][:19]}")
            with cols[1]:
                if st.button("↩️ Load", key=f"load_{i}"):
                    st.session_state.ai_sql = entry["query"]
                    st.rerun()

# Example queries
with st.expander("💡 Example Questions", expanded=True):
    st.markdown("""
    **Try these examples:**
    - `Show all customers`
    - `List products with prices`
    - `Monthly sales for 2024`
    - `Orders from the last month`
    - `Products by category`
    - `Customer order history`
    - `Top selling products`
    """)

# Footer
st.markdown("---")
st.caption("💡 Tip: Be specific with your questions for better results! Include timeframes, entities, and what you want to see.
🚀 To Use:
Click "Auto-setup Demo" in sidebar first

Then click "Connect to Database"

Ask questions like:

"Show all customers"

"List products with prices"

"Monthly sales for 2024"

"Orders from the last month"

The database will now be properly created and connected!")


# Auto-initialize demo DB if not connected
if not st.session_state.engine and db_type == "SQLite (local demo)":
    if st.sidebar.button("🔄 Auto-setup Demo", help="Automatically setup demo database"):
        with st.spinner("Setting up demo database..."):
            success = init_sqlite_demo("demo_db.sqlite")
            if success:
                try:
                    engine = make_engine(db_type, sqlite_path="demo_db.sqlite")
                    st.session_state.engine = engine
                    st.session_state.schema_text = get_schema_text(engine)
                    st.session_state.schema_dict = get_schema_dict(engine)
                    st.sidebar.success("✅ Demo database setup complete!")
                except Exception as e:
                    st.sidebar.error(f"❌ Setup failed: {e}")

