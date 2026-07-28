"""
mcp_server.py
--------------
Custom Model Context Protocol (MCP) Server providing local file reading,
directory listing, and system context tools for the Agentic Codebase &
Knowledge QA System.

Run standalone (stdio transport) with:
    python mcp_server.py

This server can be attached to any MCP-compatible client (e.g. Claude Desktop,
a LangGraph MCP client adapter, etc.) by pointing the client config at this
script.
"""

import os
import platform
import sys
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("codebase-context-server")

ALLOWED_EXTENSIONS = {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".cfg"}
EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "lancedb_store"}


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read and return the contents of a local text file.

    Args:
        file_path: Absolute or relative path to the file to read.
    """
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found: {file_path}"
        if not os.path.isfile(file_path):
            return f"ERROR: Path is not a file: {file_path}"

        file_size = os.path.getsize(file_path)
        if file_size > 2_000_000:
            return f"ERROR: File too large to read ({file_size} bytes): {file_path}"

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return content
    except PermissionError:
        return f"ERROR: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"ERROR: Failed to read file '{file_path}': {str(e)}"


@mcp.tool()
def list_directory(dir_path: str = ".") -> str:
    """List files and subdirectories within a given directory path (non-recursive).

    Args:
        dir_path: Path to the directory to list. Defaults to current directory.
    """
    try:
        if not os.path.exists(dir_path):
            return f"ERROR: Directory not found: {dir_path}"
        if not os.path.isdir(dir_path):
            return f"ERROR: Path is not a directory: {dir_path}"

        entries = []
        for entry in sorted(os.listdir(dir_path)):
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                entries.append(f"[DIR]  {entry}")
            else:
                size = os.path.getsize(full_path)
                entries.append(f"[FILE] {entry} ({size} bytes)")

        if not entries:
            return f"Directory '{dir_path}' is empty."
        return "\n".join(entries)
    except PermissionError:
        return f"ERROR: Permission denied listing directory: {dir_path}"
    except Exception as e:
        return f"ERROR: Failed to list directory '{dir_path}': {str(e)}"


@mcp.tool()
def list_source_files(dir_path: str = ".") -> str:
    """Recursively list all source/config files (.py, .md, .json, etc.) under a directory.

    Args:
        dir_path: Root directory to search recursively.
    """
    try:
        if not os.path.isdir(dir_path):
            return f"ERROR: Directory not found: {dir_path}"

        matches = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext in ALLOWED_EXTENSIONS:
                    matches.append(os.path.join(root, fname))

        if not matches:
            return f"No source files found under '{dir_path}'."
        return "\n".join(matches)
    except Exception as e:
        return f"ERROR: Failed to walk directory '{dir_path}': {str(e)}"


@mcp.tool()
def get_system_context() -> str:
    """Return basic system/environment context (OS, Python version, cwd, timestamp)."""
    try:
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": sys.version.split()[0],
            "cwd": os.getcwd(),
            "timestamp": datetime.now().isoformat(),
        }
        return "\n".join(f"{k}: {v}" for k, v in info.items())
    except Exception as e:
        return f"ERROR: Failed to gather system context: {str(e)}"


if __name__ == "__main__":
    # Runs the MCP server over stdio so it can be attached to any MCP client.
    mcp.run(transport="stdio")
