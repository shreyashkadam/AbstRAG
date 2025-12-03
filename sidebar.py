"""Sidebar component for abstRAG UI"""

import streamlit as st
from typing import Optional


def render_sidebar(metrics_collector=None):
    """Render the sidebar with settings and metrics"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Database", "🟢 Online")
        with col2:
            st.metric("API", "🟢 Ready")
        
        # Metrics
        if metrics_collector:
            st.subheader("📊 Performance Metrics")
            stats = metrics_collector.get_stats()
            
            if stats["total_queries"] > 0:
                st.metric("Total Queries", stats["total_queries"])
                st.metric("Avg Latency", f"{stats['average_latency']:.2f}s")
                st.metric("Total Errors", stats["total_errors"])
            else:
                st.info("No queries yet")
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        **abstRAG** uses RAG (Retrieval-Augmented Generation) 
        to answer questions about quantitative finance papers 
        from arXiv.
        
        - Semantic search across paper abstracts and content
        - Context-aware responses using LLMs
        - Real-time document retrieval
        """)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

