"""Document management: upload, list, delete."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend.components import ui  # noqa: E402
from frontend.components.document_manager import render_documents, render_upload  # noqa: E402

ui.setup_page("Documents", "📚", "Upload files to index them, and manage what the assistant can search.")
render_upload()
render_documents()
