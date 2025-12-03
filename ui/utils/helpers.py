"""UI helper functions"""

import streamlit as st
from typing import List, Dict, Any
import uuid


def format_paper_reference(paper_url: str) -> str:
    """Format paper reference for display"""
    # Extract paper ID from URL
    if "/abs/" in paper_url:
        paper_id = paper_url.split("/abs/")[-1]
        return f"arXiv:{paper_id}"
    return paper_url


def create_chat_message(role: str, content: str, avatar: str = None) -> Dict[str, Any]:
    """Create a chat message dictionary"""
    if avatar is None:
        avatar = "🤖" if role == "assistant" else "👨‍💻"
    
    return {
        "role": role,
        "content": content,
        "avatar": avatar,
        "timestamp": None,  # Can add timestamp if needed
    }


def get_unique_session_id() -> str:
    """Get or create unique session ID"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


