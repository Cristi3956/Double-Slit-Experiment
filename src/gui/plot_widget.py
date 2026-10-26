"""High-performance 1D intensity cross-section plotting widget.

This module provides a real-time scientific graphing widget powered by pyqtgraph.
It overlays:
1. Analytical Theoretical Curve:
   Exact Fraunhofer sinc^2 * cos^2 solution (warm amber curve).
2. GPU Numerical Simulation Profile:
   Computed 1D horizontal center slice from the 3D Helmholtz wave solver (cyan curve).
3. Quantum Monte Carlo Histogram:
   Spatial distribution of discrete detected photons across observation bins (emerald curve).
4. Physical Figures of Merit & Metrics Banner:
   Live indicators for Young's fringe spacing (Delta_x), central envelope width,
   Michelson visibility contrast (V), Pearson correlation (R), and Fresnel number (N_F).
5. Interactive Mouse Crosshair:
   Hover readout displaying exact millimeter position x and normalized irradiance I(x).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.config import DarkPalette, LightPalette, SimulationMode, UIColors
from src.physics.analytical import AnalyticalMetrics


# Configure global pyqtgraph scientific appearance
pg.setConfigOptions(antialias=True, useOpenGL=False)


class IntensityPlotWidget(QWidget):
    """Scientific 1D cross-section graph widget with theoretical overlay and metrics."""

    def __init__(self, parent=None) -> None:
        """Initializes the plotting widget and layout."""
        super().__init__(parent)

        self._setup_ui()
        self._setup_plot_items()
        self._setup_crosshair()

    # ==========================================================================
    # UI Layout Construction
    # ==========================================================================

    def _setup_ui(self) -> None:
        """Constructs layout containing the graph, metrics dashboard, and legend."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Top Metrics Banner Bar
        self.metrics_bar = QFrame(self)
        self.metrics_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {UIColors.SURFACE_CARD};
                border: 1px solid {UIColors.BORDER_COLOR};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }}
        """)
        metrics_layout = QHBoxLayout(self.metrics_bar)
        metrics_layout.setContentsMargins(8, 4, 8, 4)
        metrics_layout.setSpacing(16)

        self.lbl_fringe_spacing = QLabel("Δx (Fringe): —", self)
        self.lbl_envelope_width = QLabel("Envelope 2x₀: —", self)
        self.lbl_visibility = QLabel("Visibility V: —", self)
        self.lbl_correlation = QLabel("Correlation R: —", self)
        self.lbl_fresnel_num = QLabel("Regime: —", self)

        # Color-coded labels
        self.lbl_fringe_spacing.setStyleSheet(f"color: {UIColors.ACCENT_CYAN}; font-weight: 600;")
        self.lbl_envelope_width.setStyleSheet(f"color: {UIColors.TEXT_PRIMARY};")
        self.lbl_visibility.setStyleSheet(f"color: {UIColors.ACCENT_AMBER}; font-weight: 600;")
        self.lbl_correlation.setStyleSheet(f"color: {UIColors.ACCENT_GREEN}; font-weight: 600;")
        self.lbl_fresnel_num.setStyleSheet(f"color: {UIColors.TEXT_SECONDARY};")

        metrics_layout.addWidget(self.lbl_fringe_spacing)
        metrics_layout.addWidget(self.lbl_envelope_width)
        metrics_layout.addWidget(self.lbl_visibility)
        metrics_layout.addWidget(self.lbl_correlation)
        metrics_layout.addWidget(self.lbl_fresnel_num)
        metrics_layout.addStretch()

        # Crosshair Coordinate Readout Label
        self.lbl_crosshair = QLabel("x: 0.00 mm | I: 0.000", self)
        self.lbl_crosshair.setStyleSheet(f"color: {UIColors.TEXT_MUTED}; font-family: monospace; font-size: 11px;")
        metrics_layout.addWidget(self.lbl_crosshair)

        main_layout.addWidget(self.metrics_bar)

        # 2. PyQtGraph Viewport
        self.plot_widget = pg.PlotWidget(parent=self)
        self.plot_widget.setBackground(UIColors.PLOT_BACKGROUND)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_widget.setMouseEnabled(x=True, y=False)  # Lock y-axis in [0, 1] for stability

        # Axis Labels & Styling
        font = QFont("Segoe UI", 9)
        self.plot_item.getAxis("bottom").setLabel("Detector Screen Position x", units="mm")
        self.plot_item.getAxis("left").setLabel("Normalized Irradiance I(x)", units="")
        self.plot_item.getAxis("bottom").tickFont = font
        self.plot_item.getAxis("left").tickFont = font
        self.plot_item.getAxis("bottom").setTextPen(QColor(UIColors.TEXT_SECONDARY))
        self.plot_item.getAxis("left").setTextPen(QColor(UIColors.TEXT_SECONDARY))

        # Scientific Grid Lines
        self.plot_item.showGrid(x=True, y=True, alpha=0.22)
        self.plot_item.setYRange(-0.02, 1.08, padding=0.0)

        # Embedded Interactive Legend
        self.legend = self.plot_item.addLegend(offset=(15, 15))
        self.legend.setLabelTextColor(QColor(UIColors.TEXT_PRIMARY))
        self.legend.setBrush(QColor(18, 22, 30, 210))
        self.legend.setPen(QColor(UIColors.BORDER_COLOR))

        main_layout.addWidget(self.plot_widget, stretch=1)

    # ==========================================================================
    # Plot Curves Initialization
    # ==========================================================================

    def _setup_plot_items(self) -> None:
        """Initializes plot curves for theory, simulation, and quantum hits."""
        # 1. Analytical Theoretical Curve (Amber Gold, solid)
        pen_theory = pg.mkPen(
            color=UIColors.PLOT_THEORETICAL,
            width=2.0,
            style=Qt.PenStyle.SolidLine,
        )
        self.curve_theory = self.plot_item.plot(
            name="Analytical Theory (Fraunhofer sinc²·cos²)",
            pen=pen_theory,
        )

        # 2. GPU Numerical Wave Simulation Curve (Cyan, solid)
        pen_sim = pg.mkPen(
            color=UIColors.PLOT_SIMULATED,
            width=2.2,
            style=Qt.PenStyle.SolidLine,
        )
        self.curve_sim = self.plot_item.plot(
            name="GPU Wave Solver (3D Helmholtz)",
            pen=pen_sim,
        )

        # 3. Quantum Photon Histogram (Emerald Green)
        pen_quantum = pg.mkPen(
            color=UIColors.PLOT_HISTOGRAM,
            width=1.8,
            style=Qt.PenStyle.SolidLine,
        )
        self.curve_quantum = self.plot_item.plot(
            name="Quantum Monte Carlo (Photons)",
            pen=pen_quantum,
            fillLevel=0.0,
            brush=pg.mkBrush(16, 185, 129, 45),  # Semi-transparent emerald fill
        )

        # 4. Vertical Indicators for First Diffraction Minima (+- x_min)
        pen_marker = pg.mkPen(color=QColor(245, 158, 11, 100), width=1.2, style=Qt.PenStyle.DashLine)
        self.line_min_pos = pg.InfiniteLine(angle=90, movable=False, pen=pen_marker)
        self.line_min_neg = pg.InfiniteLine(angle=90, movable=False, pen=pen_marker)
        self.plot_item.addItem(self.line_min_pos)
        self.plot_item.addItem(self.line_min_neg)
        self.line_min_pos.setVisible(False)
        self.line_min_neg.setVisible(False)

    # ==========================================================================
    # Interactive Crosshair
    # ==========================================================================

    def _setup_crosshair(self) -> None:
        """Creates mouse hover crosshair and connects mouse move signals."""
        crosshair_pen = pg.mkPen(color=QColor(160, 175, 200, 160), width=1.0, style=Qt.PenStyle.DashLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.plot_item.addItem(self.v_line, ignoreBounds=True)
        self.plot_item.addItem(self.h_line, ignoreBounds=True)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        """Updates crosshair line positions and coordinate text readout."""
        if not self.plot_item.sceneBoundingRect().contains(scene_pos):
            return

        mouse_point = self.plot_item.vb.mapSceneToView(scene_pos)
        x_mm = float(mouse_point.x())
        y_val = float(mouse_point.y())

        self.v_line.setPos(x_mm)
        self.h_line.setPos(y_val)

        self.lbl_crosshair.setText(f"x: {x_mm:+.2f} mm | I: {max(0.0, min(1.0, y_val)):.3f}")

    # ==========================================================================
    # Public Data Update API
    # ==========================================================================

    def update_data(
        self,
        x_coords_m: np.ndarray,
        simulated_1d: np.ndarray,
        theoretical_1d: np.ndarray,
        metrics: AnalyticalMetrics,
        error_stats: Dict[str, float],
        mode: SimulationMode = SimulationMode.CLASSICAL_WAVE,
        quantum_bins_m: Optional[np.ndarray] = None,
        quantum_hist: Optional[np.ndarray] = None,
    ) -> None:
        """Updates all graph curves and metrics banner.

        Args:
            x_coords_m: 1D horizontal spatial coordinates in meters.
            simulated_1d: 1D normalized intensity array from wave solver.
            theoretical_1d: 1D normalized intensity array from analytical theory.
            metrics: AnalyticalMetrics instance from physical configuration.
            error_stats: Dictionary containing 'correlation', 'rmse', 'mse'.
            mode: SimulationMode (classical continuous wave vs. quantum photons).
            quantum_bins_m: Optional 1D bin centers for quantum histogram.
            quantum_hist: Optional 1D normalized quantum hit counts.
        """
        # Convert meters to millimeters for human-readable plotting
        x_mm = x_coords_m * 1e3

        # 1. Update Theoretical and Numerical Curves
        self.curve_theory.setData(x_mm, theoretical_1d)
        self.curve_sim.setData(x_mm, simulated_1d)

        # 2. Update Quantum Histogram
        if mode == SimulationMode.QUANTUM_MONTE_CARLO and quantum_bins_m is not None and quantum_hist is not None:
            self.curve_quantum.setVisible(True)
            bins_mm = quantum_bins_m * 1e3
            self.curve_quantum.setData(bins_mm, quantum_hist)
        else:
            self.curve_quantum.setVisible(False)

        # 3. Update Diffraction Minimum Markers (+- x_min)
        half_env_mm = (metrics.envelope_width_m / 2.0) * 1e3
        if half_env_mm > 0.0:
            self.line_min_pos.setPos(half_env_mm)
            self.line_min_neg.setPos(-half_env_mm)
            self.line_min_pos.setVisible(True)
            self.line_min_neg.setVisible(True)
        else:
            self.line_min_pos.setVisible(False)
            self.line_min_neg.setVisible(False)

        # 4. Update Top Metrics Dashboard Banner
        if metrics.fringe_spacing_m > 0.0:
            self.lbl_fringe_spacing.setText(f"Δx (Fringe): {metrics.fringe_spacing_m * 1e3:.2f} mm")
        else:
            self.lbl_fringe_spacing.setText("Δx: N/A (Single Slit)")

        self.lbl_envelope_width.setText(f"Envelope 2x₀: {metrics.envelope_width_m * 1e3:.1f} mm")
        self.lbl_visibility.setText(f"Visibility V: {metrics.fringe_visibility:.3f}")

        corr = error_stats.get("correlation", 1.0)
        self.lbl_correlation.setText(f"Correlation R: {corr * 100.0:.1f}%")
        self.lbl_fresnel_num.setText(f"NF: {metrics.fresnel_number:.3f} ({metrics.regime_label})")

    def set_theme(self, theme: str = "dark") -> None:
        """Updates plot background, axes, grid, curves, and banner colors for the theme."""
        is_dark = (theme.lower() == "dark")
        pal = DarkPalette if is_dark else LightPalette

        self.plot_widget.setBackground(pal.plot_bg)
        axis_text_pen = QColor("#0f172a" if not is_dark else pal.text_secondary)
        # Black axis and grid lines in light mode as requested by user
        axis_line_pen = pg.mkPen(color="#000000" if not is_dark else pal.border_color, width=1.0)

        self.plot_item.getAxis("bottom").setTextPen(axis_text_pen)
        self.plot_item.getAxis("left").setTextPen(axis_text_pen)
        self.plot_item.getAxis("bottom").setPen(axis_line_pen)
        self.plot_item.getAxis("left").setPen(axis_line_pen)

        # Scientific Grid Lines (black with 38% alpha on soft light blue in light mode)
        self.plot_item.showGrid(x=True, y=True, alpha=0.38 if not is_dark else 0.22)

        # 1. Update Curve Pens with high-contrast colors
        pen_theory = pg.mkPen(
            color=pal.plot_theoretical,
            width=2.2 if not is_dark else 2.0,
            style=Qt.PenStyle.SolidLine,
        )
        self.curve_theory.setPen(pen_theory)

        pen_sim = pg.mkPen(
            color=pal.plot_simulated,
            width=2.5 if not is_dark else 2.0,
            style=Qt.PenStyle.SolidLine,
        )
        self.curve_sim.setPen(pen_sim)

        pen_quantum = pg.mkPen(
            color=pal.plot_histogram,
            width=2.0,
        )
        brush_quantum = pg.mkBrush(
            color=QColor(4, 120, 87, 70) if not is_dark else QColor(16, 185, 129, 45)
        )
        self.curve_quantum.setPen(pen_quantum)
        self.curve_quantum.setBrush(brush_quantum)

        # 2. Update Metrics Banner
        self.metrics_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {pal.surface_card};
                border: 1px solid {pal.border_color};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QLabel {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }}
        """)
        self.lbl_fringe_spacing.setStyleSheet(f"color: {pal.accent_cyan}; font-weight: 600;")
        self.lbl_envelope_width.setStyleSheet(f"color: {pal.text_primary};")
        self.lbl_visibility.setStyleSheet(f"color: {pal.accent_amber}; font-weight: 600;")
        self.lbl_correlation.setStyleSheet(f"color: {pal.accent_green}; font-weight: 600;")
        self.lbl_fresnel_num.setStyleSheet(f"color: {pal.text_secondary};")
        self.lbl_crosshair.setStyleSheet(f"color: {pal.text_muted}; font-family: monospace; font-size: 11px;")

        # 3. Update Legend Box
        self.legend.setLabelTextColor(QColor(pal.text_primary))
        if is_dark:
            self.legend.setBrush(QColor(18, 22, 30, 210))
            self.legend.setPen(QColor(pal.border_color))
        else:
            self.legend.setBrush(QColor(237, 243, 248, 235))
            self.legend.setPen(QColor(pal.border_color))

    def auto_range(self) -> None:
        """Resets plot viewport zoom to fit all data snugly."""
        self.plot_item.enableAutoRange(axis=pg.ViewBox.XAxis)
        self.plot_item.setYRange(-0.02, 1.08, padding=0.0)


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = IntensityPlotWidget()
    widget.setWindowTitle("Intensity Plot (Standalone)")
    widget.resize(800, 400)
    widget.show()
    sys.exit(app.exec())
