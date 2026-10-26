"""Main application window for the Double-Slit Experiment Simulation.

This module integrates all scientific components into a cohesive PyQt6 workstation:
1. Control Panel: Parameter sliders, presets, and mode toggles.
2. 3D Virtual Optical Bench: Hardware-accelerated OpenGL apparatus visualization.
3. 2D Detector Viewport: High-resolution irradiance texture and phosphor hits.
4. 1D Intensity Plot: Real-time curve comparison with Fraunhofer analytical theory.
5. Quantum Simulation Loop: 60 FPS timer driving stochastic photon generation.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.config import (
    DarkPalette,
    LightPalette,
    QT_STYLE_SHEET,
    SimulationConfig,
    SimulationMode,
    UIColors,
    get_stylesheet,
)
from src.gui.control_panel import ControlPanel
from src.gui.plot_widget import IntensityPlotWidget
from src.physics.analytical import (
    calculate_analytical_metrics,
    compare_simulation_to_theory,
    compute_theoretical_1d_slice,
)
from src.physics.quantum_engine import QuantumAccumulator, QuantumEngine
from src.physics.wave_engine import WaveEngine, WaveResult
from src.rendering.gl_apparatus import GLApparatusWidget
from src.rendering.gl_detector import GLDetectorWidget


class MainWindow(QMainWindow):
    """Main window hosting the complete double-slit simulation laboratory."""

    def __init__(self) -> None:
        """Initializes main window, UI layouts, physics engines, and simulation timers."""
        super().__init__()

        self.setWindowTitle("Double-Slit Experiment & Quantum Wave Optics Laboratory")
        self.resize(1480, 940)
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(QT_STYLE_SHEET)

        # 1. Instantiate Physics & Rendering Engines
        self.wave_engine = WaveEngine()
        self.quantum_engine = QuantumEngine()
        self.quantum_accumulator = QuantumAccumulator(max_retained_hits=150_000, histogram_bins=200)

        # Active Data Caches
        self.latest_config: SimulationConfig = SimulationConfig()
        self.latest_wave_result: Optional[WaveResult] = None
        self.latest_theoretical_1d: Optional[np.ndarray] = None
        self.latest_metrics = None
        self.latest_error_stats = {"correlation": 1.0, "rmse": 0.0, "mse": 0.0}
        self.laser_enabled: bool = True
        self.current_theme: str = "dark"

        # 2. Construct User Interface Hierarchy
        self._setup_ui()
        self._setup_menus_and_toolbars()
        self._setup_status_bar()

        # Apply initial theme & button state
        self.gl_apparatus.set_theme(self.current_theme)
        self.gl_detector.set_theme(self.current_theme)
        self.plot_widget.set_theme(self.current_theme)
        self._update_laser_button_style(self.laser_enabled)

        # 3. Quantum Simulation Timer (60 FPS)
        self.sim_timer = QTimer(self)
        self.sim_timer.setInterval(16)  # ~60 FPS
        self.sim_timer.timeout.connect(self._on_quantum_timer_tick)
        self.last_tick_time: float = time.perf_counter()

        # 4. Keyboard Shortcuts (F11 Fullscreen Toggle)
        self.shortcut_fullscreen = QShortcut(QKeySequence("F11"), self)
        self.shortcut_fullscreen.activated.connect(self._toggle_fullscreen)

        # 5. Initial Physics Propagation
        self._on_parameters_changed()

    def _toggle_fullscreen(self) -> None:
        """Toggles between fullscreen and maximized/normal display."""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def showEvent(self, event) -> None:
        """Configures initial splitter proportions to match the user's preferred layout."""
        super().showEvent(event)
        if not hasattr(self, "_has_initialized_layout"):
            self._has_initialized_layout = True
            # Control panel fixed comfortable width ~330px
            self.main_h_splitter.setSizes([330, max(800, self.width() - 330)])
            # Vertical: 3D Optical Bench ~72% height, 1D graph ~28% height
            total_h = self.right_v_splitter.height()
            top_h = int(total_h * 0.72)
            bot_h = max(150, total_h - top_h)
            self.right_v_splitter.setSizes([top_h, bot_h])

    # ==========================================================================
    # UI Hierarchy Setup
    # ==========================================================================

    def _setup_ui(self) -> None:
        """Constructs layout splitters, dockable panels, and viewports."""
        central_container = QWidget(self)
        self.setCentralWidget(central_container)

        root_layout = QHBoxLayout(central_container)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # Main Horizontal Splitter: [Control Panel | Viewports & Graphs]
        self.main_h_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left Column: Parameter Control Panel
        self.control_panel = ControlPanel(self)
        self.control_panel.setMinimumWidth(320)
        self.control_panel.setMaximumWidth(460)
        self.control_panel.parameters_changed.connect(self._on_parameters_changed)
        self.control_panel.reset_photons_requested.connect(self._on_reset_photons)
        self.main_h_splitter.addWidget(self.control_panel)

        # Right Column: Vertical Splitter [Viewports | 1D Intensity Plot]
        self.right_v_splitter = QSplitter(Qt.Orientation.Vertical, self)

        # Upper Area: Tabbed Viewports (3D Apparatus Bench vs. 2D Screen Detector)
        self.viewport_tabs = QTabWidget(self)

        # Tab 1: 3D Interactive Laboratory Bench
        self.gl_apparatus = GLApparatusWidget(self)
        self.viewport_tabs.addTab(self.gl_apparatus, "3D Optical Bench")

        # Tab 2: 2D Screen Detector (Zoomable Irradiance & Phosphor Hits)
        self.gl_detector = GLDetectorWidget(self)
        self.viewport_tabs.addTab(self.gl_detector, "2D Detector Screen")

        self.right_v_splitter.addWidget(self.viewport_tabs)

        # Lower Area: 1D Intensity Cross-Section Plot (Compact, ~28% vertical share)
        self.plot_widget = IntensityPlotWidget(self)
        self.plot_widget.setMinimumHeight(150)
        self.right_v_splitter.addWidget(self.plot_widget)

        # Set vertical proportions: ~72% 3D Viewport, ~28% 1D Graph
        self.right_v_splitter.setStretchFactor(0, 5)
        self.right_v_splitter.setStretchFactor(1, 2)

        self.main_h_splitter.addWidget(self.right_v_splitter)
        self.main_h_splitter.setStretchFactor(0, 0)
        self.main_h_splitter.setStretchFactor(1, 1)

        root_layout.addWidget(self.main_h_splitter)

    # ==========================================================================
    # Menus, Actions, and Status Bar
    # ==========================================================================

    def _setup_menus_and_toolbars(self) -> None:
        """Sets up application menubar and quick action shortcuts."""
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        act_export_csv = QAction("Export 1D Intensity Data to CSV...", self)
        act_export_csv.setShortcut(QKeySequence("Ctrl+S"))
        act_export_csv.triggered.connect(self._export_csv)
        menu_file.addAction(act_export_csv)

        menu_file.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # View Menu
        menu_view = menubar.addMenu("&View")

        act_reset_3d = QAction("Reset 3D Camera Overview", self)
        act_reset_3d.setShortcut(QKeySequence("R"))
        act_reset_3d.triggered.connect(self.gl_apparatus.reset_camera)
        menu_view.addAction(act_reset_3d)

        act_reset_2d = QAction("Reset 2D Detector Zoom/Pan", self)
        act_reset_2d.triggered.connect(self.gl_detector.reset_view)
        menu_view.addAction(act_reset_2d)

        menu_view.addSeparator()

        act_toggle_beam = QAction("Toggle Volumetric Diffraction Beam", self)
        act_toggle_beam.setCheckable(True)
        act_toggle_beam.setChecked(True)
        act_toggle_beam.toggled.connect(self._toggle_beam_volume)
        menu_view.addAction(act_toggle_beam)

        # Help Menu
        menu_help = menubar.addMenu("&Help")
        act_about = QAction("About Double-Slit Laboratory...", self)
        act_about.triggered.connect(self._show_about_dialog)
        menu_help.addAction(act_about)

        # Top-Right Header Corner Controls: Laser Power + Dark/Light Theme
        corner_widget = QWidget(self)
        corner_layout = QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 2, 10, 2)
        corner_layout.setSpacing(8)

        # 1. Laser Power Toggle Button
        self.btn_laser_power = QPushButton("Laser: ON", self)
        self.btn_laser_power.setCheckable(True)
        self.btn_laser_power.setChecked(True)
        self.btn_laser_power.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_laser_power.toggled.connect(self._toggle_laser_power)
        self._update_laser_button_style(True)
        corner_layout.addWidget(self.btn_laser_power)

        # 2. Dark / Light Mode Toggle Button
        self.btn_theme_toggle = QPushButton("Theme: Dark", self)
        self.btn_theme_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme_toggle.setStyleSheet("""
            QPushButton {
                font-weight: 600;
                padding: 4px 12px;
                border-radius: 5px;
            }
        """)
        self.btn_theme_toggle.clicked.connect(self._toggle_theme)
        corner_layout.addWidget(self.btn_theme_toggle)

        menubar.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)

    def _setup_status_bar(self) -> None:
        """Initializes status bar with device identifier and performance stats."""
        status = QStatusBar(self)
        self.setStatusBar(status)

        self.lbl_status_device = QLabel(f"Device: {self.wave_engine.device_name}", self)
        self.lbl_status_device.setStyleSheet(f"color: {DarkPalette.accent_cyan}; font-weight: 600; padding: 0 8px;")

        self.lbl_status_perf = QLabel("Solve: — ms | 60 FPS", self)
        self.lbl_status_perf.setStyleSheet(f"color: {DarkPalette.text_secondary}; padding: 0 8px;")

        status.addWidget(self.lbl_status_device)
        status.addWidget(self.lbl_status_perf)

    def _update_laser_button_style(self, is_on: bool) -> None:
        """Applies green active / red standby styling to laser power toggle button."""
        is_light = (self.current_theme == "light")
        if is_on:
            self.btn_laser_power.setText("Laser: ON")
            text_col = "#047857" if is_light else "#10b981"
            bg_col = "rgba(4, 120, 87, 0.15)" if is_light else "rgba(16, 185, 129, 0.22)"
            hover_bg = "rgba(4, 120, 87, 0.25)" if is_light else "rgba(16, 185, 129, 0.38)"
            self.btn_laser_power.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_col};
                    border: 1px solid {text_col};
                    color: {text_col};
                    font-weight: 700;
                    padding: 4px 12px;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {hover_bg};
                }}
            """)
        else:
            self.btn_laser_power.setText("Laser: OFF")
            text_col = "#b91c1c" if is_light else "#ef4444"
            bg_col = "rgba(185, 28, 28, 0.15)" if is_light else "rgba(239, 68, 68, 0.22)"
            hover_bg = "rgba(185, 28, 28, 0.25)" if is_light else "rgba(239, 68, 68, 0.38)"
            self.btn_laser_power.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_col};
                    border: 1px solid {text_col};
                    color: {text_col};
                    font-weight: 700;
                    padding: 4px 12px;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: {hover_bg};
                }}
            """)

    def _toggle_laser_power(self, checked: bool) -> None:
        """Toggles optical laser power emission state with Star Wars animation."""
        self.laser_enabled = checked
        self._update_laser_button_style(checked)

        # Update 3D Apparatus and 2D Screen
        self.gl_apparatus.set_laser_enabled(checked)
        self.gl_detector.set_laser_enabled(checked)

        if checked:
            if getattr(self, "_dirty_while_off", False) or self.latest_wave_result is None:
                self._dirty_while_off = False
                self._on_parameters_changed()
            else:
                if self.latest_wave_result is not None:
                    self.plot_widget.update_data(
                        x_coords_m=self.latest_wave_result.x_coordinates,
                        simulated_1d=self.latest_wave_result.intensity_1d,
                        theoretical_1d=self.latest_theoretical_1d,
                        metrics=self.latest_metrics,
                        error_stats=self.latest_error_stats,
                        mode=self.latest_config.mode,
                    )
            self.lbl_status_perf.setText(f"Laser: ON | Emitting {self.latest_config.wavelength*1e9:.0f} nm")
        else:
            if self.latest_wave_result is not None:
                zero_1d = np.zeros_like(self.latest_wave_result.intensity_1d)
                self.plot_widget.update_data(
                    x_coords_m=self.latest_wave_result.x_coordinates,
                    simulated_1d=zero_1d,
                    theoretical_1d=zero_1d,
                    metrics=self.latest_metrics,
                    error_stats={"correlation": 0.0, "rmse": 0.0, "mse": 0.0},
                    mode=self.latest_config.mode,
                )
            self.lbl_status_perf.setText("Laser: OFF | Zero Emission")

    def _toggle_theme(self) -> None:
        """Toggles between scientific dark mode and clean light mode."""
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.btn_theme_toggle.setText(f"Theme: {self.current_theme.capitalize()}")

        # Update global stylesheet
        new_qss = get_stylesheet(self.current_theme)
        QApplication.instance().setStyleSheet(new_qss)
        self.setStyleSheet(new_qss)

        # Update plot widget and 3D / 2D viewports
        self.plot_widget.set_theme(self.current_theme)
        self.gl_apparatus.set_theme(self.current_theme)
        self.gl_detector.set_theme(self.current_theme)

        pal = LightPalette if self.current_theme == "light" else DarkPalette
        self.lbl_status_device.setStyleSheet(f"color: {pal.accent_cyan}; font-weight: 600; padding: 0 8px;")
        self.lbl_status_perf.setStyleSheet(f"color: {pal.text_secondary}; padding: 0 8px;")

        # Update theme button styling
        if self.current_theme == "light":
            self.btn_theme_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #edf3f8;
                    border: 1px solid #94a3b8;
                    color: #0f172a;
                    font-weight: 600;
                    padding: 4px 12px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #dbe6f0;
                    border-color: #1d4ed8;
                }
            """)
        else:
            self.btn_theme_toggle.setStyleSheet("""
                QPushButton {
                    background-color: #1a1f2b;
                    border: 1px solid #2e3748;
                    color: #f0f4f8;
                    font-weight: 600;
                    padding: 4px 12px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #222938;
                    border-color: #4d7cfe;
                }
            """)

        # Refresh laser button styling
        self._update_laser_button_style(self.laser_enabled)

    # ==========================================================================
    # Core Simulation Pipeline & Event Handlers
    # ==========================================================================

    def _on_parameters_changed(self) -> None:
        """Executes GPU wave propagation and refreshes all viewports."""
        config = self.control_panel.get_configuration()
        self.latest_config = config

        if not self.laser_enabled:
            self._dirty_while_off = True
            return

        # 1. Run GPU Optical Wave Propagation (Helmholtz Solver)
        res = self.wave_engine.compute_propagation(config)
        self.latest_wave_result = res

        # 2. Calculate Analytical Ground Truth and Metrics
        metrics = calculate_analytical_metrics(config)
        self.latest_metrics = metrics

        theo_1d = compute_theoretical_1d_slice(config, res.x_coordinates)
        self.latest_theoretical_1d = theo_1d

        # 3. Compute Statistical Agreement (MSE, Correlation)
        error_stats = compare_simulation_to_theory(res.intensity_1d, theo_1d)
        self.latest_error_stats = error_stats

        # 4. Refresh 3D Apparatus Viewport
        self.gl_apparatus.update_parameters(
            wavelength_nm=config.wavelength * 1e9,
            distance_l_m=config.distance_l,
            slit_width_m=config.slit_width,
            slit_separation_m=config.slit_separation,
            slit_height_m=config.slit_height,
            screen_span_m=config.screen_span,
            aperture_type=config.aperture_type,
            which_way_active=config.which_way_detected,
            mode=config.mode,
            num_slits=config.num_slits,
        )
        self.gl_apparatus.set_screen_texture(res.intensity_2d)

        # 5. Refresh 2D Detector Viewport
        self.gl_detector.set_simulation_mode(config.mode)
        self.gl_detector.set_wave_field(
            intensity_2d=res.intensity_2d,
            wavelength_nm=config.wavelength * 1e9,
            physical_span_m=config.screen_span,
        )

        # 6. Refresh 1D Plot Widget
        q_bins, q_hist = None, None
        if config.mode == SimulationMode.QUANTUM_MONTE_CARLO:
            q_bins, q_hist = self.quantum_accumulator.get_normalized_1d_histogram()

        self.plot_widget.update_data(
            x_coords_m=res.x_coordinates,
            simulated_1d=res.intensity_1d,
            theoretical_1d=theo_1d,
            metrics=metrics,
            error_stats=error_stats,
            mode=config.mode,
            quantum_bins_m=q_bins,
            quantum_hist=q_hist,
        )

        # 7. Update Status Bar Performance Readout
        active_fps = self.gl_apparatus.current_fps if self.viewport_tabs.currentIndex() == 0 else self.gl_detector.current_fps
        self.lbl_status_perf.setText(
            f"Solve Time: {res.execution_time_ms:.2f} ms ({res.propagation_method}) | FPS: {active_fps:.0f} | Grid: {config.grid_size}²"
        )

        # 8. Manage Quantum Simulation Timer
        if config.mode == SimulationMode.QUANTUM_MONTE_CARLO:
            if not self.sim_timer.isActive():
                self.sim_timer.start()
        else:
            if self.sim_timer.isActive():
                self.sim_timer.stop()

    def _on_quantum_timer_tick(self) -> None:
        """Generates stochastic photon hits per timer tick according to emission rate."""
        if not self.laser_enabled or self.latest_wave_result is None:
            return

        cfg = self.latest_config
        now = time.perf_counter()
        dt = now - self.last_tick_time
        self.last_tick_time = now

        # Calculate number of photons to emit in this time slice dt
        photons_to_emit = int(round(cfg.photon_rate * dt))
        photons_to_emit = max(1, min(5000, photons_to_emit))

        # Sample stochastic photon detections on GPU
        batch = self.quantum_engine.sample_batch(
            count=photons_to_emit,
            intensity_2d=self.latest_wave_result.intensity_2d,
            x_coords=self.latest_wave_result.x_coordinates,
            y_coords=self.latest_wave_result.y_coordinates,
            which_way_active=cfg.which_way_detected,
        )

        # Accumulate into virtual detector screen
        self.quantum_accumulator.add_batch(batch, cfg.screen_span)

        # Spawn and advance in-flight 3D photons in the 3D optical bench
        self.gl_apparatus.spawn_in_flight_photons(
            hit_x_m=batch.x_hits,
            hit_y_m=batch.y_hits,
            slit_tags=batch.slit_tags,
        )
        self.gl_apparatus.advance_in_flight_photons(dt)

        # Refresh 2D Detector with active point cloud
        hit_x, hit_y, tags = self.quantum_accumulator.get_active_hits()
        self.gl_detector.set_quantum_hits(
            hit_x=hit_x,
            hit_y=hit_y,
            hit_tags=tags,
            total_count=self.quantum_accumulator.total_photons,
            wavelength_nm=cfg.wavelength * 1e9,
            physical_span_m=cfg.screen_span,
            which_way_active=cfg.which_way_detected,
        )

        # Refresh 1D Plot with updated photon histogram
        q_bins, q_hist = self.quantum_accumulator.get_normalized_1d_histogram()
        self.plot_widget.update_data(
            x_coords_m=self.latest_wave_result.x_coordinates,
            simulated_1d=self.latest_wave_result.intensity_1d,
            theoretical_1d=self.latest_theoretical_1d,
            metrics=self.latest_metrics,
            error_stats=self.latest_error_stats,
            mode=cfg.mode,
            quantum_bins_m=q_bins,
            quantum_hist=q_hist,
        )

        # Update Status Bar with active FPS and photon count
        active_fps = self.gl_apparatus.current_fps if self.viewport_tabs.currentIndex() == 0 else self.gl_detector.current_fps
        self.lbl_status_perf.setText(
            f"Solve: {self.latest_wave_result.execution_time_ms:.1f} ms | FPS: {active_fps:.0f} | Photons: {self.quantum_accumulator.total_photons:,}"
        )

    def _on_reset_photons(self) -> None:
        """Clears accumulated quantum particle hits and resets histogram."""
        self.quantum_accumulator.reset(self.latest_config.screen_span)
        self.gl_apparatus.clear_quantum_hits()
        self.gl_detector.set_quantum_hits(
            hit_x=np.empty(0, dtype=np.float32),
            hit_y=np.empty(0, dtype=np.float32),
            hit_tags=np.empty(0, dtype=np.int8),
            total_count=0,
            wavelength_nm=self.latest_config.wavelength * 1e9,
            physical_span_m=self.latest_config.screen_span,
            which_way_active=self.latest_config.which_way_detected,
        )
        if self.latest_wave_result is not None:
            q_bins, q_hist = self.quantum_accumulator.get_normalized_1d_histogram()
            self.plot_widget.update_data(
                x_coords_m=self.latest_wave_result.x_coordinates,
                simulated_1d=self.latest_wave_result.intensity_1d,
                theoretical_1d=self.latest_theoretical_1d,
                metrics=self.latest_metrics,
                error_stats=self.latest_error_stats,
                mode=self.latest_config.mode,
                quantum_bins_m=q_bins,
                quantum_hist=q_hist,
            )

    def _toggle_beam_volume(self, checked: bool) -> None:
        """Toggles visibility of the 3D diffraction light cone."""
        self.gl_apparatus.show_beam_volume = checked
        self.gl_apparatus.update()

    # ==========================================================================
    # Export and Dialog Routines
    # ==========================================================================

    def _export_csv(self) -> None:
        """Exports 1D spatial coordinates and intensity values to CSV file."""
        if self.latest_wave_result is None or self.latest_theoretical_1d is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 1D Intensity Profile",
            "double_slit_intensity_data.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not file_path:
            return

        x_mm = self.latest_wave_result.x_coordinates * 1e3
        sim_i = self.latest_wave_result.intensity_1d
        theo_i = self.latest_theoretical_1d

        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_position_mm", "simulated_intensity_norm", "theoretical_intensity_norm"])
            for xm, si, ti in zip(x_mm, sim_i, theo_i):
                writer.writerow([f"{xm:.6f}", f"{si:.6f}", f"{ti:.6f}"])

        QMessageBox.information(
            self,
            "Export Successful",
            f"Successfully exported {len(x_mm)} data points to:\n{file_path}",
        )

    def _show_about_dialog(self) -> None:
        """Displays academic about dialog."""
        QMessageBox.about(
            self,
            "About Double-Slit Laboratory",
            "<h3>Double-Slit Experiment Simulation</h3>"
            "<p><b>Course:</b> Metode Avansate (Advanced Methods)</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li><b>GPU Physics Engine:</b> 3D Helmholtz diffraction solver via 2D FFT (CuPy on NVIDIA RTX GPU).</li>"
            "<li><b>Quantum Monte Carlo:</b> Born's rule particle-by-particle detector accumulation.</li>"
            "<li><b>Decoherence:</b> Which-way detector quantum state collapse demonstration.</li>"
            "<li><b>3D OpenGL Bench:</b> Interactive apparatus with orbit camera and live textured screen.</li>"
            "<li><b>Analytical Verification:</b> Live Pearson correlation against Fraunhofer sinc²·cos² theory.</li>"
            "</ul>",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
