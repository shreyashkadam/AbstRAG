"""References display component"""

import streamlit as st
from typing import List


def render_references(references: List[str]):
    """Render paper references in a modern way"""
    if not references:
        return
    
    st.markdown("### 📚 Referenced Papers")
    
    # Create a container for references
    for i, ref in enumerate(references, 1):
        with st.container():
            col1, col2 = st.columns([1, 20])
            with col1:
                st.markdown(f"**{i}.**")
            with col2:
                # Make it clickable
                st.markdown(f"[{ref}]({ref})", unsafe_allow_html=False)
        
        if i < len(references):
            st.divider()


