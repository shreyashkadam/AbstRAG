"""Header component for abstRAG UI"""

import streamlit as st


def render_header():
    """Render the application header"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("<div style='font-size: 60px; text-align: center; padding-top: 10px;'>📚</div>", unsafe_allow_html=True)
    
    with col2:
        st.title("abstRAG")
        st.caption("Retrieval-Augmented Generation with arXiv Knowledge Base")
        st.markdown("**Ask questions about quantitative finance research papers**")
    
    st.divider()

