"""Streamlit GUI for browsing all research papers in PostgreSQL database"""

import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from psycopg import sql
from abstrag.core.database import (
    PostgresParams,
    get_article_id_data,
    open_db_connection,
)
from abstrag.config import get_config

load_dotenv(".env")

# Load config
config_dict = get_config()
if config_dict:
    from abstrag.config import IngestionConfig
    config_ingestion = IngestionConfig(**config_dict["ingestion"])
    EMBEDDING_MODEL_NAME = config_ingestion.embedding_model_name
else:
    st.error("Failed to load configuration")
    st.stop()

# Construct table names
TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace("-", "_")

# Page config
st.set_page_config(
    page_icon="📚",
    layout="wide",
    page_title="Paper Browser - abstRAG",
    initial_sidebar_state="expanded",
)

# Connect to database
@st.cache_resource
def get_db_connection():
    postgres_connection_params = PostgresParams(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"],
        pwd=os.environ["POSTGRES_PWD"],
        database=os.environ["POSTGRES_DB"],
    )
    return open_db_connection(connection_params=postgres_connection_params, autocommit=True)

conn = get_db_connection()

if conn is None:
    st.error("❌ Failed to connect to database. Please check your connection settings.")
    st.stop()

# Header
st.title("📚 Research Paper Browser")
st.markdown("Browse all research papers stored in the PostgreSQL database")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Table selection
    table_type = st.radio(
        "View papers from:",
        ["Article Table", "Abstract Table"],
        help="Choose which table to browse"
    )
    
    selected_table = TABLE_EMBEDDING_ARTICLE if table_type == "Article Table" else TABLE_EMBEDDING_ABSTRACT
    
    st.divider()
    
    # Search filter
    search_query = st.text_input(
        "🔎 Search paper IDs:",
        placeholder="e.g., 2401.12345",
        help="Filter papers by ID (partial match supported)"
    )

# Load all paper IDs
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_paper_ids(table_name: str):
    return get_article_id_data(conn=conn, table_name=table_name)

# Load paper details
@st.cache_data(ttl=300)
def load_paper_details(table_name: str, paper_ids: list):
    """Load details for papers including chunk counts"""
    if not paper_ids:
        return pd.DataFrame()
    
    table_identifier = sql.Identifier(table_name)
    query = sql.SQL("""
        SELECT 
            article_id,
            COUNT(*) as chunk_count,
            MIN(id) as first_chunk_id
        FROM {}
        WHERE article_id = ANY(%s::text[])
        GROUP BY article_id
        ORDER BY article_id
    """).format(table_identifier)
    
    with conn.cursor() as cur:
        cur.execute(query, (paper_ids,))
        results = cur.fetchall()
        df = pd.DataFrame(results, columns=['paper_id', 'chunk_count', 'first_chunk_id'])
        return df

# Get all papers
with st.spinner("Loading papers..."):
    all_paper_ids = load_paper_ids(selected_table)
    
    # Filter by search query
    if search_query:
        filtered_paper_ids = [pid for pid in all_paper_ids if search_query.lower() in pid.lower()]
    else:
        filtered_paper_ids = all_paper_ids
    
    # Load details
    if filtered_paper_ids:
        papers_df = load_paper_details(selected_table, filtered_paper_ids)
    else:
        papers_df = pd.DataFrame(columns=['paper_id', 'chunk_count', 'first_chunk_id'])

# Display statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Papers", len(all_paper_ids))
with col2:
    st.metric("Filtered Papers", len(filtered_paper_ids))
with col3:
    st.metric("Total Chunks", papers_df['chunk_count'].sum() if not papers_df.empty else 0)

st.divider()

# Display papers table
if not papers_df.empty:
    st.subheader("📄 Papers List")
    
    # Format paper IDs as clickable links
    papers_df['arxiv_link'] = papers_df['paper_id'].apply(
        lambda x: f"https://arxiv.org/abs/{x}"
    )
    papers_df['paper_id_display'] = papers_df.apply(
        lambda row: f"[{row['paper_id']}]({row['arxiv_link']})",
        axis=1
    )
    
    # Display table
    display_df = papers_df[['paper_id_display', 'chunk_count']].copy()
    display_df.columns = ['Paper ID', 'Number of Chunks']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Paper ID": st.column_config.LinkColumn("Paper ID", display_text="Open on arXiv"),
            "Number of Chunks": st.column_config.NumberColumn("Number of Chunks", format="%d")
        }
    )
    
    # Paper detail viewer
    st.divider()
    st.subheader("📖 View Paper Content")
    
    selected_paper = st.selectbox(
        "Select a paper to view content:",
        options=filtered_paper_ids,
        format_func=lambda x: f"{x} ({papers_df[papers_df['paper_id']==x]['chunk_count'].values[0]} chunks)" if not papers_df.empty else x
    )
    
    if selected_paper:
        # Load content for selected paper
        table_identifier = sql.Identifier(selected_table)
        query = sql.SQL("""
            SELECT id, article_id, content
            FROM {}
            WHERE article_id = %s
            ORDER BY id
            LIMIT 10
        """).format(table_identifier)
        
        with conn.cursor() as cur:
            cur.execute(query, (selected_paper,))
            chunks = cur.fetchall()
        
        if chunks:
            st.info(f"Showing first 10 chunks of {len(chunks)} total chunks for paper **{selected_paper}**")
            
            for i, (chunk_id, paper_id, content) in enumerate(chunks, 1):
                with st.expander(f"Chunk {i} (ID: {chunk_id})"):
                    st.text_area(
                        "Content:",
                        value=content,
                        height=200,
                        key=f"chunk_{chunk_id}",
                        label_visibility="collapsed"
                    )
                    st.caption(f"Paper: [{paper_id}](https://arxiv.org/abs/{paper_id})")
        else:
            st.warning(f"No content found for paper {selected_paper}")
else:
    if search_query:
        st.warning(f"No papers found matching '{search_query}'")
    else:
        st.info("No papers found in the database. Run `python update_database.py` to ingest papers.")

# Footer
st.divider()
st.caption(f"Viewing papers from table: `{selected_table}`")