"""Metrics display component"""

import streamlit as st
from typing import Optional


def render_metrics(
    num_documents: int,
    num_papers: int,
    response_time: float,
    avg_latency: Optional[float] = None,
):
    """Render metrics in a visually appealing way"""
    st.markdown("### 📊 Retrieval Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Documents",
            value=num_documents,
            help="Number of document chunks retrieved"
        )
    
    with col2:
        st.metric(
            label="Papers",
            value=num_papers,
            help="Number of unique papers referenced"
        )
    
    with col3:
        st.metric(
            label="Response Time",
            value=f"{response_time:.2f}s",
            help="Time taken to generate response"
        )
    
    with col4:
        if avg_latency:
            st.metric(
                label="Avg Latency",
                value=f"{avg_latency:.2f}s",
                help="Average latency over last hour"
            )
        else:
            st.metric(
                label="Avg Latency",
                value="N/A"
            )


