"""2D OpenGL detector screen viewport with live texture and photon hit rendering.

This module implements a hardware-accelerated QOpenGLWidget that renders:
1. Continuous 2D Irradiance Field:
   Uploaded as an RGBA texture dynamically tinted to the physical laser wavelength
   or rendered in scientific false-color mode.
2. Quantum Monte Carlo Particle Hits:
   Point-cloud rendering of discrete photon impacts using OpenGL additive alpha
   blending (phosphor luminescence effect). When which-way detection is active,
   photons are color-coded according to slit origin (Slit 1 vs. Slit 2).
3. Interactive Navigation:
   Interactive zoom (mouse scroll wheel) and pan (left-click drag) to inspect
   individual microscopic fringes, with physical millimeter scale bars and HUD.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from src.config import SimulationMode
from src.rendering.color_utils import wavelength_to_rgb


class GLDetectorWidget(QOpenGLWidget):
    """OpenGL viewport displaying the 2D detector screen and photon hits."""

    # Signal emitted when user adjusts pan/zoom
    viewport_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        """Initializes the OpenGL detector widget."""
        super().__init__(parent)

        # Rendering Data State
        self.mode: SimulationMode = SimulationMode.CLASSICAL_WAVE
        self.wavelength_nm: float = 532.0
        self.physical_span_m: float = 25e-3
        self.which_way_active: bool = False
        self.total_photons_count: int = 0
        self.laser_enabled: bool = True

        # Texture Buffer (Continuous Field)
        self._texture_id: int = 0
        self._tex_width: int = 0
        self._tex_height: int = 0
        self._intensity_matrix: Optional[np.ndarray] = None
        self._texture_dirty: bool = False

        # Quantum Particle Buffers
        self._hit_x: Optional[np.ndarray] = None
        self._hit_y: Optional[np.ndarray] = None
        self._hit_tags: Optional[np.ndarray] = None

        # Camera / Pan & Zoom State
        self.zoom_factor: float = 1.0
        self.pan_offset_x: float = 0.0
        self.pan_offset_y: float = 0.0
        self._is_panning: bool = False
        self._last_mouse_pos: QPointF = QPointF()

        # Appearance Settings
        self.use_false_color: bool = False
        self.point_size: float = 2.5
        self.show_scale_bar: bool = True
        self.show_crosshair: bool = True
        self.current_theme: str = "dark"

        # Real-Time Measured FPS
        import time as _time
        self._fps_last_time: float = _time.perf_counter()
        self._fps_frames: int = 0
        self.current_fps: float = 60.0

        # Widget Focus and Mouse Tracking
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ==========================================================================
    # Public Data Update API
    # ==========================================================================

    def set_wave_field(
        self,
        intensity_2d: np.ndarray,
        wavelength_nm: float,
        physical_span_m: float,
    ) -> None:
        """Updates the continuous wave intensity texture.

        Args:
            intensity_2d: 2D float32 array normalized to [0, 1].
            wavelength_nm: Laser wavelength in nanometers.
            physical_span_m: Physical span of the screen in meters.
        """
        self._intensity_matrix = intensity_2d
        self.wavelength_nm = wavelength_nm
        self.physical_span_m = physical_span_m
        self._texture_dirty = True
        self.update()

    def set_quantum_hits(
        self,
        hit_x: np.ndarray,
        hit_y: np.ndarray,
        hit_tags: np.ndarray,
        total_count: int,
        wavelength_nm: float,
        physical_span_m: float,
        which_way_active: bool,
    ) -> None:
        """Updates the quantum particle hit arrays for point-cloud rendering.

        Args:
            hit_x: 1D float32 array of x positions in meters.
            hit_y: 1D float32 array of y positions in meters.
            hit_tags: 1D int8 array of slit tags (0=coherent, 1=slit1, 2=slit2).
            total_count: Total cumulative photons detected.
            wavelength_nm: Optical wavelength in nm.
            physical_span_m: Physical span of the screen in meters.
            which_way_active: Whether which-way tagging is active.
        """
        self._hit_x = hit_x
        self._hit_y = hit_y
        self._hit_tags = hit_tags
        self.total_photons_count = total_count
        self.wavelength_nm = wavelength_nm
        self.physical_span_m = physical_span_m
        self.which_way_active = which_way_active
        self.update()

    def set_simulation_mode(self, mode: SimulationMode) -> None:
        """Switches between continuous wave field and quantum particle view."""
        self.mode = mode
        self.update()

    def set_laser_enabled(self, enabled: bool) -> None:
        """Enables or disables laser illumination on the detector."""
        self.laser_enabled = enabled
        self.update()

    def set_theme(self, theme: str) -> None:
        """Configures 2D detector screen viewport colors and HUD styling for theme."""
        self.current_theme = theme.lower()
        if self.isValid():
            self.makeCurrent()
            if self.current_theme == "light":
                GL.glClearColor(0.85, 0.89, 0.94, 1.0)
            else:
                GL.glClearColor(0.05, 0.07, 0.10, 1.0)
            self.doneCurrent()
        self.update()

    def reset_view(self) -> None:
        """Resets camera zoom and pan offsets to default center."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.viewport_changed.emit()
        self.update()

    # ==========================================================================
    # OpenGL Lifecycle Methods
    # ==========================================================================

    def initializeGL(self) -> None:
        """Initializes OpenGL context, texture objects, and blend modes."""
        if self.current_theme == "light":
            GL.glClearColor(0.85, 0.89, 0.94, 1.0)
        else:
            GL.glClearColor(0.05, 0.07, 0.10, 1.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)  # Additive blending for luminescence

        # Generate OpenGL texture object
        self._texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def resizeGL(self, width: int, height: int) -> None:
        """Handles viewport dimension changes."""
        GL.glViewport(0, 0, width, height)

    def paintGL(self) -> None:
        """Executes OpenGL rendering pipeline."""
        # Calculate Real-Time Measured FPS
        import time as _time
        now = _time.perf_counter()
        dt_fps = now - self._fps_last_time
        self._fps_frames += 1
        if dt_fps >= 0.4:
            self.current_fps = float(self._fps_frames) / max(1e-4, dt_fps)
            self._fps_frames = 0
            self._fps_last_time = now

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Setup 2D Orthographic Projection Matrix with Pan & Zoom
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        w = float(max(1, self.width()))
        h = float(max(1, self.height()))
        aspect = w / h

        # Half-extents in normalized coordinate space [-1, 1]
        half_x = 1.0 / self.zoom_factor
        half_y = 1.0 / self.zoom_factor

        if aspect >= 1.0:
            half_x *= aspect
        else:
            half_y /= aspect

        # Apply Pan offset
        left = -half_x - self.pan_offset_x
        right = half_x - self.pan_offset_x
        bottom = -half_y - self.pan_offset_y
        top = half_y - self.pan_offset_y

        GL.glOrtho(left, right, bottom, top, -1.0, 1.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        # 1. Render Screen Background Quad
        self._render_screen_frame()

        # 2. Render View Modes (only when laser is enabled)
        if self.laser_enabled:
            if self.mode == SimulationMode.CLASSICAL_WAVE:
                self._render_wave_texture()
            else:
                self._render_quantum_point_cloud()

        # 3. Render Crosshair overlay in OpenGL
        if self.show_crosshair:
            self._render_crosshair()

    # ==========================================================================
    # Rendering Helpers
    # ==========================================================================

    def _render_screen_frame(self) -> None:
        """Renders dark background aperture plate boundary."""
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glColor4f(0.02, 0.03, 0.05, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2f(-1.0, -1.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()

        # Frame Border
        GL.glLineWidth(1.5)
        if self.current_theme == "light":
            GL.glColor4f(0.35, 0.42, 0.55, 1.0)
        else:
            GL.glColor4f(0.2, 0.25, 0.35, 1.0)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex2f(-1.0, -1.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()

    def _render_wave_texture(self) -> None:
        """Uploads and maps the continuous 2D wavefield texture onto the screen quad."""
        if self._intensity_matrix is None:
            return

        # Upload texture to GPU if data was refreshed
        if self._texture_dirty:
            self._upload_texture()
            self._texture_dirty = False

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glColor4f(1.0, 1.0, 1.0, 1.0)

        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex2f(-1.0, -1.0)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex2f(1.0, -1.0)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex2f(1.0, 1.0)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex2f(-1.0, 1.0)
        GL.glEnd()

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

    def _upload_texture(self) -> None:
        """Formats the 2D float array into an RGBA texture tinted to laser color."""
        if self._intensity_matrix is None:
            return

        mat = self._intensity_matrix
        h, w = mat.shape

        # Base emission color corresponding to laser wavelength
        base_r, base_g, base_b = wavelength_to_rgb(self.wavelength_nm)

        # Create RGBA uint8 buffer
        rgba = np.empty((h, w, 4), dtype=np.uint8)

        if self.use_false_color:
            # High-contrast scientific colormap (Cyan to Yellow to White)
            # R channel: activates at mid-to-high intensity
            rgba[:, :, 0] = np.clip(mat * 255.0 * 1.5 - 64.0, 0.0, 255.0).astype(np.uint8)
            # G channel: primary intensity
            rgba[:, :, 1] = np.clip(mat * 255.0 * 1.2, 0.0, 255.0).astype(np.uint8)
            # B channel: deep blues/cyans at lower levels
            rgba[:, :, 2] = np.clip((1.0 - mat) * 128.0 + mat * 255.0, 0.0, 255.0).astype(np.uint8)
            rgba[:, :, 3] = (mat * 255.0).astype(np.uint8)
        else:
            # Physically accurate true monochromatic laser emission
            rgba[:, :, 0] = np.clip(mat * base_r * 255.0, 0.0, 255.0).astype(np.uint8)
            rgba[:, :, 1] = np.clip(mat * base_g * 255.0, 0.0, 255.0).astype(np.uint8)
            rgba[:, :, 2] = np.clip(mat * base_b * 255.0, 0.0, 255.0).astype(np.uint8)
            # Alpha controls phosphor brightness
            rgba[:, :, 3] = 255

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        if (h, w) != (self._tex_height, self._tex_width):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba.tobytes()
            )
            self._tex_width = w
            self._tex_height = h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba.tobytes()
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _render_quantum_point_cloud(self) -> None:
        """Renders accumulated discrete photon clicks with additive blending."""
        if self._hit_x is None or len(self._hit_x) == 0:
            return

        GL.glPointSize(self.point_size)
        GL.glEnable(GL.GL_POINT_SMOOTH)

        half_span = 0.5 * self.physical_span_m
        if half_span <= 0.0:
            return

        # Map physical meters [-half_span, half_span] to normalized quad [-1.0, 1.0]
        norm_x = self._hit_x / half_span
        norm_y = self._hit_y / half_span

        # Mask points outside screen boundary
        valid = (np.abs(norm_x) <= 1.0) & (np.abs(norm_y) <= 1.0)
        px = norm_x[valid]
        py = norm_y[valid]
        tags = self._hit_tags[valid] if self._hit_tags is not None else None

        base_r, base_g, base_b = wavelength_to_rgb(self.wavelength_nm)

        GL.glBegin(GL.GL_POINTS)
        if self.which_way_active and tags is not None:
            # Quantum Path Tagging: color-code photons by slit
            # Slit 1 (Left) -> Cyan (0.0, 0.8, 1.0)
            # Slit 2 (Right) -> Amber (1.0, 0.6, 0.1)
            for x, y, tag in zip(px, py, tags):
                if tag == 1:
                    GL.glColor4f(0.0, 0.85, 1.0, 0.65)
                elif tag == 2:
                    GL.glColor4f(1.0, 0.60, 0.1, 0.65)
                else:
                    GL.glColor4f(base_r, base_g, base_b, 0.65)
                GL.glVertex2f(float(x), float(y))
        else:
            # Monochromatic coherent photon clicks with phosphor luminescence
            GL.glColor4f(base_r, base_g, base_b, 0.55)
            for x, y in zip(px, py):
                GL.glVertex2f(float(x), float(y))

        GL.glEnd()

    def _render_crosshair(self) -> None:
        """Draws faint central alignment crosshair on the detector."""
        GL.glLineWidth(1.0)
        GL.glColor4f(1.0, 1.0, 1.0, 0.15)

        ch_size = 0.08
        GL.glBegin(GL.GL_LINES)
        # Horizontal crosshair line
        GL.glVertex2f(-ch_size, 0.0)
        GL.glVertex2f(ch_size, 0.0)
        # Vertical crosshair line
        GL.glVertex2f(0.0, -ch_size)
        GL.glVertex2f(0.0, ch_size)
        glEnd = GL.glEnd()

    # ==========================================================================
    # Heads-Up Display (HUD) & 2D Painter Overlay
    # ==========================================================================

    def paintEvent(self, event) -> None:
        """Paints OpenGL content, then renders HUD overlays via QPainter."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Physical Scale Bar Overlay
        if self.show_scale_bar:
            self._draw_scale_bar(painter)

        # 2. Status Readout Overlay
        self._draw_hud_text(painter)

        painter.end()

    def _draw_scale_bar(self, painter: QPainter) -> None:
        """Draws a calibrated physical millimeter scale bar at the bottom-left."""
        # Calculate visual pixels per millimeter
        screen_h = float(self.height())
        screen_w = float(self.width())
        aspect = screen_w / max(1.0, screen_h)

        half_ext = 1.0 / self.zoom_factor
        unit_height_pixels = screen_h / (2.0 * half_ext)
        # Half-extent covers half_span_m
        pixels_per_meter = unit_height_pixels / (0.5 * self.physical_span_m)
        pixels_per_mm = pixels_per_meter * 1e-3

        # Choose scale bar length (e.g. 1 mm, 2 mm, 5 mm, or 10 mm)
        target_bar_pixels = 80.0
        candidate_lengths_mm = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        chosen_mm = candidate_lengths_mm[0]
        for c in candidate_lengths_mm:
            if c * pixels_per_mm <= 150.0:
                chosen_mm = c

        bar_pixel_width = chosen_mm * pixels_per_mm

        # Position at bottom-left corner
        x0 = 20
        y0 = self.height() - 25
        is_light = (self.current_theme == "light")

        pen = QPen(QColor(15, 23, 42) if is_light else QColor(220, 230, 245), 2)
        painter.setPen(pen)
        painter.drawLine(int(x0), int(y0), int(x0 + bar_pixel_width), int(y0))
        painter.drawLine(int(x0), int(y0 - 4), int(x0), int(y0 + 4))
        painter.drawLine(int(x0 + bar_pixel_width), int(y0 - 4), int(x0 + bar_pixel_width), int(y0 + 4))

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        text = f"{chosen_mm:g} mm"
        painter.drawText(int(x0 + bar_pixel_width / 2.0 - 18), int(y0 - 6), text)

    def _draw_hud_text(self, painter: QPainter) -> None:
        """Draws informational HUD badges at top corners."""
        painter.setFont(QFont("Segoe UI", 10))
        is_light = (self.current_theme == "light")

        # Top-Left: Mode & Zoom Badge
        badge_rect_w = 215
        badge_rect_h = 44
        if is_light:
            painter.fillRect(15, 15, badge_rect_w, badge_rect_h, QColor(237, 243, 248, 235))
            painter.setPen(QColor(148, 163, 184))
            painter.drawRoundedRect(15, 15, badge_rect_w, badge_rect_h, 6, 6)

            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(3, 105, 161))  # High-contrast Sky-700
            painter.drawText(25, 32, "DETECTOR SCREEN")

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(4, 120, 87))  # High-contrast Emerald-700
            painter.drawText(160, 32, f"{self.current_fps:.0f} FPS")

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(30, 41, 59))  # Dark Slate
            if not self.laser_enabled:
                mode_str = "Laser Power: OFF"
            else:
                mode_str = "Classical Wave" if self.mode == SimulationMode.CLASSICAL_WAVE else "Quantum Photons"
            painter.drawText(25, 50, f"{mode_str} | Zoom: {self.zoom_factor:.1f}x")

            # Top-Right: Photon Count Badge (Quantum Mode)
            if self.mode == SimulationMode.QUANTUM_MONTE_CARLO:
                count_w = 175
                count_h = 42
                x_right = self.width() - count_w - 15
                painter.fillRect(x_right, 15, count_w, count_h, QColor(237, 243, 248, 235))
                painter.setPen(QColor(148, 163, 184))
                painter.drawRoundedRect(x_right, 15, count_w, count_h, 6, 6)

                painter.setPen(QColor(4, 120, 87))
                painter.drawText(x_right + 12, 32, "PHOTONS DETECTED")
                painter.setPen(QColor(15, 23, 42))
                painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                painter.drawText(x_right + 12, 49, f"{self.total_photons_count:,}")
        else:
            painter.fillRect(15, 15, badge_rect_w, badge_rect_h, QColor(18, 22, 30, 200))
            painter.setPen(QColor(46, 55, 72))
            painter.drawRoundedRect(15, 15, badge_rect_w, badge_rect_h, 6, 6)

            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(0, 210, 255))
            painter.drawText(25, 32, "DETECTOR SCREEN")

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(16, 185, 129))
            painter.drawText(160, 32, f"{self.current_fps:.0f} FPS")

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(160, 175, 200))
            if not self.laser_enabled:
                mode_str = "Laser Power: OFF"
            else:
                mode_str = "Classical Wave" if self.mode == SimulationMode.CLASSICAL_WAVE else "Quantum Photons"
            painter.drawText(25, 50, f"{mode_str} | Zoom: {self.zoom_factor:.1f}x")

            # Top-Right: Photon Count Badge (Quantum Mode)
            if self.mode == SimulationMode.QUANTUM_MONTE_CARLO:
                count_w = 170
                count_h = 42
                x_right = self.width() - count_w - 15
                painter.fillRect(x_right, 15, count_w, count_h, QColor(18, 22, 30, 200))
                painter.setPen(QColor(46, 55, 72))
                painter.drawRoundedRect(x_right, 15, count_w, count_h, 6, 6)

                painter.setPen(QColor(16, 185, 129))
                painter.drawText(x_right + 12, 32, "PHOTONS DETECTED")
                painter.setPen(QColor(240, 244, 248))
                painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                painter.drawText(x_right + 12, 49, f"{self.total_photons_count:,}")

    # ==========================================================================
    # Mouse & Keyboard Interaction Handlers
    # ==========================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Begins panning on left mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handles pan translation on mouse drag."""
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()

            # Convert pixel delta to normalized view space
            screen_h = float(max(1, self.height()))
            norm_delta_x = (delta.x() / screen_h) * (2.0 / self.zoom_factor)
            norm_delta_y = (delta.y() / screen_h) * (2.0 / self.zoom_factor)

            self.pan_offset_x += norm_delta_x
            self.pan_offset_y -= norm_delta_y  # Inverted y in Qt vs OpenGL

            self.viewport_changed.emit()
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Ends pan interaction."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handles zoom in/out with mouse scroll wheel."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_factor = min(30.0, self.zoom_factor * 1.15)
        elif delta < 0:
            self.zoom_factor = max(0.5, self.zoom_factor / 1.15)

        self.viewport_changed.emit()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double click resets zoom and pan."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = GLDetectorWidget()
    widget.setWindowTitle("2D Detector Screen (Standalone Viewport)")
    widget.resize(700, 700)
    widget.show()
    sys.exit(app.exec())
