"""Prompt template system"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Prompt template manager"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """Initialize prompt template system
        
        Args:
            template_dir: Directory containing template files
        """
        if template_dir is None:
            # Default to utils/templates directory
            template_dir = str(Path(__file__).parent / "templates")
        
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info(f"Initialized prompt template system with directory: {template_dir}")
    
    def load_template(self, template_name: str) -> Template:
        """Load a template by name
        
        Args:
            template_name: Name of template file
            
        Returns:
            Jinja2 Template object
        """
        try:
            template = self.env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def render(
        self,
        template_name: str,
        **kwargs,
    ) -> str:
        """Render a template with variables
        
        Args:
            template_name: Name of template file
            **kwargs: Variables to pass to template
            
        Returns:
            Rendered prompt string
        """
        template = self.load_template(template_name)
        return template.render(**kwargs)
    
    def render_string(self, template_string: str, **kwargs) -> str:
        """Render a template from string
        
        Args:
            template_string: Template as string
            **kwargs: Variables to pass to template
            
        Returns:
            Rendered prompt string
        """
        template = self.env.from_string(template_string)
        return template.render(**kwargs)


# Global template instance
_template_manager: Optional[PromptTemplate] = None


def get_template_manager() -> PromptTemplate:
    """Get global template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = PromptTemplate()
    return _template_manager


