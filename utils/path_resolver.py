"""Utility functions for resolving file paths across different directory structures"""

import os
from typing import List, Optional
from pathlib import Path


def find_file(
    filename: str,
    search_dirs: Optional[List[str]] = None,
    script_dir: Optional[str] = None,
) -> str:
    """Find a file in multiple possible locations
    
    Args:
        filename: Name of the file to find (can be relative or absolute)
        search_dirs: List of directories to search in (relative to script_dir or absolute)
        script_dir: Base directory for relative paths (defaults to project root)
    
    Returns:
        Path to the file if found
        
    Raises:
        FileNotFoundError: If file is not found in any location
    """
    # If absolute path and exists, return it
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    # If relative path exists in current directory, return it
    if os.path.exists(filename):
        return os.path.abspath(filename)
    
    # Determine base directory
    if script_dir is None:
        # Try to find project root (directory containing config.yaml or .git)
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "config.yaml").exists() or (parent / ".git").exists():
                script_dir = str(parent)
                break
        else:
            script_dir = str(current)
    
    # Default search directories
    if search_dirs is None:
        search_dirs = [
            "",  # Current directory
            "reports/evaluation/data",  # Default evaluation data location
            "data",  # Data directory
            ".",  # Root directory
        ]
    
    # Try each search directory
    tried_paths = []
    for search_dir in search_dirs:
        if search_dir:
            full_path = os.path.join(script_dir, search_dir, filename)
        else:
            full_path = os.path.join(script_dir, filename)
        
        tried_paths.append(full_path)
        if os.path.exists(full_path):
            return os.path.abspath(full_path)
    
    # Also try just the filename in current working directory
    cwd_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(cwd_path) and cwd_path not in tried_paths:
        return os.path.abspath(cwd_path)
    
    # File not found
    error_msg = f"File not found: {filename}\n"
    error_msg += "Searched in:\n"
    for path in tried_paths:
        error_msg += f"  - {path}\n"
    if cwd_path not in tried_paths:
        error_msg += f"  - {cwd_path}\n"
    
    raise FileNotFoundError(error_msg)


def find_evaluation_questions_file(
    filename: str,
    script_dir: Optional[str] = None,
) -> str:
    """Find evaluation questions CSV file in standard locations
    
    Args:
        filename: Name of the evaluation questions file
        script_dir: Base directory (defaults to project root)
    
    Returns:
        Path to the file if found
        
    Raises:
        FileNotFoundError: If file is not found
    """
    search_dirs = [
        "reports/evaluation/data",  # Standard location
        "",  # Root directory
        "data",  # Data directory
    ]
    return find_file(filename, search_dirs=search_dirs, script_dir=script_dir)


def find_env_file(
    filename: str = ".env",
    script_dir: Optional[str] = None,
) -> str:
    """Find environment file (.env or local_env) in standard locations
    
    Args:
        filename: Name of the env file (default: ".env")
        script_dir: Base directory (defaults to project root)
    
    Returns:
        Path to the file if found
        
    Raises:
        FileNotFoundError: If file is not found
    """
    search_dirs = [
        "",  # Root directory
        ".",  # Current directory
    ]
    
    # Try the specified filename first
    try:
        return find_file(filename, search_dirs=search_dirs, script_dir=script_dir)
    except FileNotFoundError:
        # If .env not found, try local_env as fallback
        if filename == ".env":
            try:
                return find_file("local_env", search_dirs=search_dirs, script_dir=script_dir)
            except FileNotFoundError:
                pass
        
        # Re-raise original error
        raise


def find_config_file(
    filename: str = "config.yaml",
    script_dir: Optional[str] = None,
) -> str:
    """Find config file in standard locations
    
    Args:
        filename: Name of the config file (default: "config.yaml")
        script_dir: Base directory (defaults to project root)
    
    Returns:
        Path to the file if found
        
    Raises:
        FileNotFoundError: If file is not found
    """
    search_dirs = [
        "",  # Root directory
        ".",  # Current directory
    ]
    return find_file(filename, search_dirs=search_dirs, script_dir=script_dir)


