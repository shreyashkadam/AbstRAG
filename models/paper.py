"""Paper data models"""

import datetime
from typing import List, TypedDict


class PaperID(TypedDict):
    """Metadata for an arXiv paper"""
    id: str
    summary: str
    authors: List[str]
    entry_url: str
    published: datetime.datetime
    primary_category: str
    categories: List[str]


