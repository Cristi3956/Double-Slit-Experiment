"""Main entry point script for the Double-Slit Experiment Simulation.

To run the application:
    python run.py
"""

from __future__ import annotations

import sys
import os
import warnings

# Suppress harmless CuPy CUDA_PATH detection warning (CUDA runtime is bundled via pip wheels)
warnings.filterwarnings(
    "ignore",
    message="CUDA path could not be detected",
    category=UserWarning,
    module="cupy",
)

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> int:
    """Initializes Qt runtime, applies high-DPI scaling, and launches main window."""
    # Ensure high DPI attribute support
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Double-Slit Simulation Laboratory")
    app.setOrganizationName("Advanced Methods Course")

    window = MainWindow()
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
