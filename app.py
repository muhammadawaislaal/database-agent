import streamlit as st
import sqlite3
import pandas as pd
import re
from openai import OpenAI

# ===============================
# Streamlit Page Setup
# ===============================
st.set_page_config(page_title="AI SQL Agent", layout="wide")
st.title("🤖 AI SQL Agent")
st.caption("Ask questions in plain English. The agent generates SQL, runs it, and shows results.")

# ===============================
# Initialize OpenAI client
# ===============================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ===============================
# Database Connection (default: SQLite demo)
# ===============================
st.sidebar.header("⚙️ Settings")
db_choice = st.sidebar.selectbox("Choose Database", ["SQLite (local demo)"])
db_file = "demo.db"  # extendable later

# connect to SQLite
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# ===============================
# Auto-discover Schema
# ===============================
def get_schema():
    schema = []
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for t in tables:
        tname = t[0]
        cols = cursor.execute(f"PRAGMA table_info({tname});").fetchall()
        col_names = [c[1] for c in cols]
        schema.append(f"Table {tname}: {', '.join(col_names)}")
    return "\n".join(schema)

schema_text = get_schema()
with st.expander("📑 Database Schema (auto-discovered)"):
    st.text(schema_text)

# ===============================
# SQL Safety Check
# ===============================
FORBIDDEN = ["DROP", "ALTER", "TRUNCATE", "ATTACH", "DETACH"]

def validate_sql(sql: str, read_only: bool):
    if not sql:
        return False, "Empty query."
    sql_upper = sql.strip().upper()
    if sql_upper.startswith("ERROR"):
        return False, sql
    for bad in FORBIDDEN:
        if bad in sql_upper:
            return False, f"❌ Forbidden keyword: {bad}"
    first_word_match = re.match(r"^\s*([A-Z]+)", sql_upper)
    if not first_word_match:
        return False, "❌ Invalid SQL syntax."
    keyword = first_word_match.group(1)
    read_only_allowed = {"SELECT", "WITH", "EXPLAIN"}
    full_allowed = read_only_allowed | {"INSERT", "UPDATE", "DELETE"}
    if read_only and keyword in read_only_allowed:
        return True, "Query OK"
    elif not read_only and keyword in full_allowed:
        return True, "Query OK"
    else:
        return False, f"❌ Query type not supported in this mode ({keyword})."

# ===============================
# User Input
# ===============================
read_only = st.sidebar.toggle("🔒 Read-only Mode (safe)", value=True)
user_request = st.text_area("📝 Ask in plain English:", placeholder="Show client details for 2024")

# ===============================
# AI → SQL Generator
# ===============================
if st.button("✨ Generate SQL"):
    if not user_request.strip():
        st.warning("Please enter a request first.")
    else:
        with st.spinner("Generating SQL..."):
            prompt = f"""
You are a helpful AI SQL assistant. 
Database schema:
{schema_text}

User request: "{user_request}"

Rules:
- Always return full SQL (not partial).
- Do NOT summarize columns; include ALL relevant fields unless user specifies otherwise.
- If user asks for 'details', include all columns from the table(s).
- Default is SQLite SQL syntax.
- Do NOT include explanation, only SQL code.
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are an expert SQL generator."},
                              {"role": "user", "content": prompt}],
                    temperature=0,
                )
                sql_query = response.choices[0].message.content.strip()
                st.session_state["last_sql"] = sql_query
                st.code(sql_query, language="sql")
            except Exception as e:
                st.error(f"❌ Error generating SQL: {e}")

# ===============================
# Run SQL
# ===============================
if "last_sql" in st.session_state:
    sql_query = st.text_area("📋 SQL Query (editable before running):", st.session_state["last_sql"], height=150)
    if st.button("▶️ Run SQL"):
        ok, msg = validate_sql(sql_query, read_only)
        if not ok:
            st.error(msg)
        else:
            try:
                df = pd.read_sql_query(sql_query, conn)
                if df.empty:
                    st.warning("⚠️ Query returned no rows.")
                else:
                    st.success(f"✅ Returned {len(df)} rows.")
                    st.dataframe(df, use_container_width=True)

                    # Auto visualization for numeric results
                    if df.shape[1] >= 2 and df.dtypes[1] in ("int64", "float64"):
                        st.subheader("📊 Visualization")
                        st.bar_chart(df.set_index(df.columns[0]))

                    # CSV download
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download CSV", csv, "results.csv", "text/csv")
                    
                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({"query": sql_query, "rows": len(df)})
            except Exception as e:
                st.error(f"❌ SQL Execution Error: {e}")

# ===============================
# Query History
# ===============================
if "history" in st.session_state and st.session_state["history"]:
    st.sidebar.subheader("🕒 Query History")
    for i, h in enumerate(reversed(st.session_state["history"][-5:]), 1):
        st.sidebar.text(f"{i}. {h['query'][:40]}... → {h['rows']} rows")
