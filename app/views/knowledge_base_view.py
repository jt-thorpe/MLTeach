from app.views.knowledge_base_view_ui import Ui_kb_window
from PyQt6.QtWidgets import QMainWindow


class KnowledgeBaseView(QMainWindow):
    """KnowledgeBase view."""

    def __init__(self):
        """Initialize KnowledgeBase view"""
        super().__init__()

        self._ui = Ui_kb_window()
        self._ui.setupUi(self)

    def set_up_connections(self):
        """Connect widgets to view methods."""
        pass

    def clean_up_connections(self):
        """Disconnect widgets from view methods."""
        pass
