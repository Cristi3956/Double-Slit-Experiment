"""3D interactive OpenGL apparatus viewport representing the physical laboratory bench.

This module implements an interactive 3D virtual optics bench featuring:
1. Optical Rail & Base:
   Anodized laboratory rail along the optical axis (z) with metric alignment markings.
2. Solid Laser Source:
   He-Ne / Diode laser head with a fully capped solid cylindrical body, rear power
   controls, and front glowing emission aperture.
3. Solid Slit Aperture Mount:
   Optical post and solid plate at z = 0 with distinct physical slit openings.
   Displays which-way detector sensor blocks and active indicator LEDs.
4. Volumetric Diffraction Beam (Classical Wave Mode):
   Semi-transparent glowing wave frustum from slits to screen with interference fringe shading.
5. In-Flight Quantum Photons (Quantum Monte Carlo Mode):
   Animated particles traveling in 3D flight from the slits through space to the
   detector screen. Particles leave glowing motion trails and produce impact flashes.
   When which-way detection is active, photons are visibly emitted from Slit 1 (Cyan)
   or Slit 2 (Amber) and follow separate trajectory corridors.
6. Solid Detector Screen:
   6-sided solid target screen at z = L, textured on the front face with the live
   2D diffraction pattern or accumulated quantum photon hits.
7. Multi-Pass Depth Architecture:
   Ensures robust 24-bit Z-buffer depth testing across camera rotations so that
   closer objects correctly occlude distant objects from any viewpoint.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QMouseEvent, QPainter, QWheelEvent
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from src.config import ApertureType, SimulationMode
from src.rendering.color_utils import wavelength_to_rgb


@dataclass
class Photon3D:
    """Represents an individual photon in 3D flight between slits and screen.

    Attributes:
        start_x, start_y, start_z: 3D coordinates at aperture plane (z=0).
        target_x, target_y, target_z: 3D coordinates on detector screen (z=z_screen).
        progress: Flight progress in range [0.0, 1.0].
        speed: Flight advance rate per second.
        slit_tag: 0 = coherent superposition, 1 = slit 1 (left), 2 = slit 2 (right).
        color: RGB tuple for particle glow.
    """

    start_x: float
    start_y: float
    start_z: float
    target_x: float
    target_y: float
    target_z: float
    progress: float
    speed: float
    slit_tag: int
    color: Tuple[float, float, float]


class GLApparatusWidget(QOpenGLWidget):
    """3D OpenGL laboratory bench visualization widget."""

    camera_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        """Initializes the 3D apparatus viewport."""
        super().__init__(parent)

        # Simulation Parameters
        self.wavelength_nm: float = 532.0
        self.distance_l_m: float = 0.8
        self.slit_width_m: float = 25e-6
        self.slit_separation_m: float = 120e-6
        self.slit_height_m: float = 2.5e-3
        self.screen_span_m: float = 25e-3
        self.aperture_type: ApertureType = ApertureType.DOUBLE_SLIT
        self.num_slits: int = 2
        self.which_way_active: bool = False
        self.mode: SimulationMode = SimulationMode.CLASSICAL_WAVE

        # Laser Ignition & Power-Down State (Star Wars Effect)
        self.laser_enabled: bool = True
        self.ignition_progress: float = 1.0  # 0.0 = completely off, 1.0 = fully active
        self._target_laser_power: bool = True
        import time as _time
        self._ignition_last_time: float = _time.perf_counter()
        self._ignition_timer: QTimer = QTimer(self)
        self._ignition_timer.setInterval(16)  # ~60 FPS animation tick
        self._ignition_timer.timeout.connect(self._on_ignition_tick)

        # Theme Configuration
        self.current_theme: str = "dark"

        # Live Screen Texture
        self._screen_texture_id: int = 0
        self._intensity_2d: Optional[np.ndarray] = None
        self._texture_dirty: bool = False
        self._tex_w: int = 0
        self._tex_h: int = 0

        # Quantum In-Flight Photons & Accumulated 3D Screen Hits
        self.in_flight_photons: List[Photon3D] = []
        self.max_in_flight: int = 350
        self.screen_hits_3d: List[Tuple[float, float, Tuple[float, float, float]]] = []
        self.max_screen_hits: int = 25_000

        # 3D Orbit Camera State (Configured to show laser, slits, and projected screen face clearly)
        self.camera_yaw: float = -44.0      # Azimuth rotation angle in degrees
        self.camera_pitch: float = 19.0     # Elevation angle in degrees
        self.camera_distance: float = 4.0   # Distance from orbit target in 3D units
        self.target_x: float = 0.06
        self.target_y: float = -0.01
        self.target_z: float = 0.12         # Centered near middle of bench

        # Real-Time FPS Measurement
        import time as _time
        self._fps_last_time: float = _time.perf_counter()
        self._fps_frames: int = 0
        self.current_fps: float = 60.0

        # Mouse Interaction State
        self._mouse_mode: Optional[str] = None
        self._last_mouse_pos: QPointF = QPointF()

        # Display Options
        self.show_beam_volume: bool = True
        self.show_grid: bool = True
        self.show_hud: bool = True

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ==========================================================================
    # Public Update API
    # ==========================================================================

    def update_parameters(
        self,
        wavelength_nm: float,
        distance_l_m: float,
        slit_width_m: float,
        slit_separation_m: float,
        slit_height_m: float,
        screen_span_m: float,
        aperture_type: ApertureType,
        which_way_active: bool,
        mode: SimulationMode,
        num_slits: int = 2,
    ) -> None:
        """Updates physical apparatus dimensions and optical parameters."""
        self.wavelength_nm = wavelength_nm
        self.distance_l_m = distance_l_m
        self.slit_width_m = slit_width_m
        self.slit_separation_m = slit_separation_m
        self.slit_height_m = slit_height_m
        self.screen_span_m = screen_span_m
        self.aperture_type = aperture_type
        self.which_way_active = which_way_active
        self.num_slits = num_slits

        # If switching modes, clear in-flight particles
        if self.mode != mode:
            self.mode = mode
            self.in_flight_photons.clear()

        self.update()

    def _get_visual_slit_geometry(self) -> Tuple[float, float, float, float]:
        """Computes dynamically scaled 3D visual aperture dimensions from physical parameters.

        Returns:
            Tuple of (slit_w_vis, slit_h_vis, slit_sep_vis, pinhole_r_vis) in 3D scene units.
        """
        # Slit Width a: physical range [2, 200] µm
        a_um = self.slit_width_m * 1e6
        norm_a = max(0.0, min(1.0, (a_um - 2.0) / (200.0 - 2.0)))
        slit_w_vis = 0.003 + 0.035 * (norm_a ** 0.55)

        # Slit Height h: physical range [0.1, 10.0] mm
        h_mm = self.slit_height_m * 1e3
        norm_h = max(0.0, min(1.0, (h_mm - 0.1) / (10.0 - 0.1)))
        slit_h_vis = 0.035 + 0.195 * (norm_h ** 0.5)

        # Slit Separation d: physical range [10, 1000] µm
        d_um = self.slit_separation_m * 1e6
        norm_d = max(0.0, min(1.0, (d_um - 10.0) / (1000.0 - 10.0)))
        slit_sep_vis = 0.016 + 0.145 * (norm_d ** 0.5)
        # Ensure clear visual gap between adjacent slits
        slit_sep_vis = max(slit_sep_vis, slit_w_vis + 0.008)

        # Circular Pinhole Radius: physical range [2, 200] µm
        pinhole_r_vis = 0.008 + 0.035 * (norm_a ** 0.55)

        return slit_w_vis, slit_h_vis, slit_sep_vis, pinhole_r_vis

    def set_laser_enabled(self, enabled: bool) -> None:
        """Turns the physical laser source on or off with Star Wars animation."""
        self.laser_enabled = enabled
        self._target_laser_power = enabled
        import time as _time
        self._ignition_last_time = _time.perf_counter()
        if not self._ignition_timer.isActive():
            self._ignition_timer.start()
        self.update()

    def _on_ignition_tick(self) -> None:
        """Advances Star Wars style laser ignition blade extension or power-down retraction."""
        import time as _time
        now = _time.perf_counter()
        dt = max(0.001, min(0.1, now - self._ignition_last_time))
        self._ignition_last_time = now

        if self._target_laser_power:
            # Ignition: extends forward in ~280 ms
            self.ignition_progress += dt / 0.28
            if self.ignition_progress >= 1.0:
                self.ignition_progress = 1.0
                self._ignition_timer.stop()
        else:
            # Power-down: retracts backward in ~220 ms
            self.ignition_progress -= dt / 0.22
            if self.ignition_progress <= 0.0:
                self.ignition_progress = 0.0
                self._ignition_timer.stop()
                self.in_flight_photons.clear()

        self.update()

    def set_theme(self, theme: str) -> None:
        """Configures 3D viewport colors, clear color, and HUD styling for theme."""
        self.current_theme = theme.lower()
        if self.isValid():
            self.makeCurrent()
            if self.current_theme == "light":
                GL.glClearColor(0.85, 0.89, 0.94, 1.0)
            else:
                GL.glClearColor(0.06, 0.08, 0.12, 1.0)
            self.doneCurrent()
        self.update()

    def set_screen_texture(self, intensity_2d: np.ndarray) -> None:
        """Uploads the 2D diffraction irradiance pattern to the 3D detector screen."""
        self._intensity_2d = intensity_2d
        self._texture_dirty = True
        self.update()

    def clear_quantum_hits(self) -> None:
        """Clears all in-flight photons and 3D accumulated hits."""
        self.in_flight_photons.clear()
        self.screen_hits_3d.clear()
        self.update()

    def spawn_in_flight_photons(
        self,
        hit_x_m: np.ndarray,
        hit_y_m: np.ndarray,
        slit_tags: np.ndarray,
    ) -> None:
        """Spawns 3D in-flight photons from the slits towards screen impact points.

        Args:
            hit_x_m: 1D array of horizontal impact coordinates on screen in meters.
            hit_y_m: 1D array of vertical impact coordinates on screen in meters.
            slit_tags: 1D array of slit origins (0 = coherent, 1 = slit1, 2 = slit2).
        """
        if len(hit_x_m) == 0 or self.ignition_progress < 0.82:
            return

        # Limit active particles to prevent visual overload
        space_available = self.max_in_flight - len(self.in_flight_photons)
        if space_available <= 0:
            return

        # Screen 3D properties
        z_screen = max(0.4, min(2.8, self.distance_l_m * 1.8))
        screen_w = 0.58
        screen_h = 0.58
        half_span = max(1e-4, 0.5 * self.screen_span_m)

        # Dynamic Slit physical origins in 3D
        slit_w_vis, slit_h_vis, slit_sep_vis, pinhole_r_vis = self._get_visual_slit_geometry()
        x_slit1 = -slit_sep_vis / 2.0
        x_slit2 = slit_sep_vis / 2.0
        half_w = max(0.001, slit_w_vis * 0.45)
        half_h = max(0.005, slit_h_vis * 0.45)

        base_r, base_g, base_b = wavelength_to_rgb(self.wavelength_nm)

        # Pick a subset if hit batch is large
        step = max(1, len(hit_x_m) // space_available)
        indices = range(0, len(hit_x_m), step)[:space_available]

        for i in indices:
            xm = hit_x_m[i]
            ym = hit_y_m[i]
            tag = int(slit_tags[i]) if i < len(slit_tags) else 0

            # Target 3D coordinates on screen (z_screen front face is at z_screen - 0.015)
            # Physical meters -> 3D screen geometry:
            target_3d_x = float(np.clip((xm / half_span) * (screen_w * 0.46), -screen_w * 0.48, screen_w * 0.48))
            target_3d_y = float(np.clip((ym / half_span) * (screen_h * 0.46), -screen_h * 0.48, screen_h * 0.48))
            target_3d_z = z_screen - 0.015

            # Origin 3D coordinates at aperture plane (emerging from front face at z = 0.01)
            if self.aperture_type == ApertureType.SINGLE_SLIT:
                start_x = random.uniform(-half_w * 1.5, half_w * 1.5)
                start_y = random.uniform(-half_h, half_h)
                color = (base_r, base_g, base_b)
            elif self.aperture_type == ApertureType.CIRCULAR_PINHOLE:
                pr = pinhole_r_vis * math.sqrt(random.random())
                pth = random.uniform(0.0, 2.0 * math.pi)
                start_x = pr * math.cos(pth)
                start_y = pr * math.sin(pth)
                color = (base_r, base_g, base_b)
            elif self.aperture_type == ApertureType.MULTI_SLIT:
                num = max(2, min(8, getattr(self, "num_slits", 5)))
                pitch = min(slit_sep_vis * 0.6, 0.24 / max(1, num - 1))
                x_start = -((num - 1) * pitch) / 2.0
                k_slit = random.randint(0, num - 1)
                start_x = x_start + k_slit * pitch + random.uniform(-half_w * 0.7, half_w * 0.7)
                start_y = random.uniform(-half_h, half_h)
                color = (base_r, base_g, base_b)
            elif self.which_way_active and tag == 1:
                # Slit 1 (Left) origin -> Cyan photon
                start_x = x_slit1 + random.uniform(-half_w, half_w)
                start_y = random.uniform(-half_h, half_h)
                color = (0.0, 0.85, 1.0)
            elif self.which_way_active and tag == 2:
                # Slit 2 (Right) origin -> Amber photon
                start_x = x_slit2 + random.uniform(-half_w, half_w)
                start_y = random.uniform(-half_h, half_h)
                color = (1.0, 0.60, 0.1)
            else:
                # Coherent superposition: emerges randomly from either slit corridor
                if random.random() < 0.5:
                    start_x = x_slit1 + random.uniform(-half_w, half_w)
                else:
                    start_x = x_slit2 + random.uniform(-half_w, half_w)
                start_y = random.uniform(-half_h, half_h)
                color = (base_r, base_g, base_b)

            # Flight speed: takes 0.6 to 1.1 seconds to cross the bench
            speed = random.uniform(1.1, 1.8)

            p = Photon3D(
                start_x=start_x,
                start_y=start_y,
                start_z=0.01,
                target_x=target_3d_x,
                target_y=target_3d_y,
                target_z=target_3d_z,
                progress=0.0,
                speed=speed,
                slit_tag=tag,
                color=color,
            )
            self.in_flight_photons.append(p)

    def advance_in_flight_photons(self, dt: float) -> None:
        """Advances positions of all active 3D in-flight photons.

        Args:
            dt: Time elapsed since last frame in seconds.
        """
        if self.mode != SimulationMode.QUANTUM_MONTE_CARLO:
            return

        surviving: List[Photon3D] = []
        for p in self.in_flight_photons:
            p.progress += p.speed * dt
            if p.progress >= 1.0:
                # Photon reached screen: register impact on 3D screen surface
                if len(self.screen_hits_3d) >= self.max_screen_hits:
                    # Circular overwrite
                    self.screen_hits_3d.pop(0)
                self.screen_hits_3d.append((p.target_x, p.target_y, p.color))
            else:
                surviving.append(p)

        self.in_flight_photons = surviving
        self.update()

    def reset_camera(self) -> None:
        """Resets camera to show the laser, slits, and projected screen face clearly."""
        self.camera_yaw = -44.0
        self.camera_pitch = 19.0
        self.camera_distance = 4.0
        self.target_x = 0.06
        self.target_y = -0.01
        self.target_z = 0.12
        self.camera_changed.emit()
        self.update()

    # ==========================================================================
    # OpenGL Lifecycle
    # ==========================================================================

    def initializeGL(self) -> None:
        """Configures OpenGL rendering state, depth testing, and lighting."""
        if self.current_theme == "light":
            GL.glClearColor(0.85, 0.89, 0.94, 1.0)
        else:
            GL.glClearColor(0.06, 0.08, 0.12, 1.0)
        GL.glClearDepth(1.0)

        # Enable Z-buffer Depth Testing
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_TRUE)

        # Enable Alpha Blending
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        # Smooth lines and points
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glHint(GL.GL_POINT_SMOOTH_HINT, GL.GL_NICEST)

        # Generate Screen Texture
        self._screen_texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._screen_texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def resizeGL(self, width: int, height: int) -> None:
        """Configures 3D perspective projection on resize."""
        GL.glViewport(0, 0, width, height)

    def paintGL(self) -> None:
        """Executes full 3D scene rendering pass with strict depth test enforcement."""
        # Calculate Real-Time Measured FPS
        import time as _time
        now = _time.perf_counter()
        dt_fps = now - self._fps_last_time
        self._fps_frames += 1
        if dt_fps >= 0.4:
            self.current_fps = float(self._fps_frames) / max(1e-4, dt_fps)
            self._fps_frames = 0
            self._fps_last_time = now

        # Crucial: Reset OpenGL states that QPainter alters during 2D HUD painting
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_TEXTURE_2D)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # 1. Setup Perspective Projection Matrix
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        fov_deg = 45.0
        aspect = float(max(1, self.width())) / float(max(1, self.height()))
        z_near = 0.1
        z_far = 50.0

        half_h = z_near * math.tan(math.radians(fov_deg) / 2.0)
        half_w = half_h * aspect
        GL.glFrustum(-half_w, half_w, -half_h, half_h, z_near, z_far)

        # 2. Camera ModelView Matrix
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        # Orbit transformation
        GL.glTranslatef(0.0, 0.0, -self.camera_distance)
        GL.glRotatef(self.camera_pitch, 1.0, 0.0, 0.0)
        GL.glRotatef(self.camera_yaw, 0.0, 1.0, 0.0)
        GL.glTranslatef(-self.target_x, -self.target_y, -self.target_z)

        # Setup directional lighting
        self._setup_lighting()

        # 3. Setup Spatial Coordinates on Bench
        z_laser = -0.95
        z_slits = 0.0
        z_screen = max(0.4, min(2.8, self.distance_l_m * 1.8))

        r_col, g_col, b_col = wavelength_to_rgb(self.wavelength_nm)

        # ----------------------------------------------------------------------
        # PASS 1: RENDER ALL SOLID / OPAQUE OBJECTS (Depth writes ENABLED)
        # ----------------------------------------------------------------------
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)

        # Floor grid and optical rail
        if self.show_grid:
            self._render_lab_grid()
        self._render_optical_rail(z_min=z_laser - 0.25, z_max=3.0)

        # Solid Laser Housing (Cylinder + Front Cap + Rear Solid Cap)
        self._render_laser_head(z_pos=z_laser, r=r_col, g=g_col, b=b_col)

        # Solid Slit Mount & Plate (with transmitted laser glow when powered)
        self._render_slit_mount(z_pos=z_slits, r=r_col, g=g_col, b=b_col)

        # Solid Detector Screen (Front Face Textured, Solid Metal Back & Sides)
        self._render_detector_screen(z_pos=z_screen, r=r_col, g=g_col, b=b_col)

        # ----------------------------------------------------------------------
        # PASS 2: RENDER TRANSPARENT LIGHT VOLUMES / PHOTONS (Depth writes DISABLED)
        # ----------------------------------------------------------------------
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_BLEND)

        if self.ignition_progress > 0.001:
            # Incident Collimated Laser Beam (Laser Head -> Slits Back Face)
            self._render_incident_beam(
                z_start=z_laser + 0.38, z_end=z_slits - 0.01, r=r_col, g=g_col, b=b_col
            )

            # Diffraction cone and quantum particles emerge when beam reaches slits (ignition_progress >= 0.70)
            if self.ignition_progress >= 0.70:
                raw_c = (self.ignition_progress - 0.70) / 0.30
                cone_factor = min(1.0, max(0.0, 3.0 * (raw_c ** 2) - 2.0 * (raw_c ** 3)))

                # Mode A: Classical Volumetric Light Frustum (Emerges from Slits Front Face)
                if self.mode == SimulationMode.CLASSICAL_WAVE and self.show_beam_volume:
                    self._render_diffraction_cone(
                        z_start=z_slits + 0.012, z_end=z_screen - 0.015, r=r_col, g=g_col, b=b_col,
                        alpha_factor=cone_factor,
                    )

                # Mode B: In-Flight Quantum Photons (Particles Traveling in Space)
                if self.mode == SimulationMode.QUANTUM_MONTE_CARLO:
                    self._render_in_flight_photons(alpha_factor=cone_factor)
                    self._render_screen_hits_3d(z_screen=z_screen, alpha_factor=cone_factor)

        # Restore depth writes
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_LIGHTING)

    # ==========================================================================
    # Lighting Configuration
    # ==========================================================================

    def _setup_lighting(self) -> None:
        """Sets up key and ambient lighting to give 3D depth perception."""
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        light_pos = [1.5, 3.0, 2.0, 0.0]
        light_ambient = [0.25, 0.28, 0.35, 1.0]
        light_diffuse = [0.75, 0.78, 0.85, 1.0]
        light_specular = [0.4, 0.4, 0.4, 1.0]

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_pos)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, light_ambient)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, light_specular)

    # ==========================================================================
    # In-Flight 3D Quantum Photons Renderer
    # ==========================================================================

    def _render_in_flight_photons(self, alpha_factor: float = 1.0) -> None:
        """Renders individual photons flying from slits to detector screen."""
        if not self.in_flight_photons or alpha_factor <= 0.001:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)  # Additive blending for luminescence

        # 1. Render Photon Motion Trails (Comet Tails)
        GL.glLineWidth(1.0)
        GL.glBegin(GL.GL_LINES)
        for p in self.in_flight_photons:
            u = p.progress
            u_tail = max(0.0, u - 0.08)  # Tail follows behind head

            # Current head position
            hx = (1.0 - u) * p.start_x + u * p.target_x
            hy = (1.0 - u) * p.start_y + u * p.target_y
            hz = (1.0 - u) * p.start_z + u * p.target_z

            # Tail position
            tx = (1.0 - u_tail) * p.start_x + u_tail * p.target_x
            ty = (1.0 - u_tail) * p.start_y + u_tail * p.target_y
            tz = (1.0 - u_tail) * p.start_z + u_tail * p.target_z

            r, g, b = p.color
            # Faint tail vertex
            GL.glColor4f(r, g, b, 0.04 * alpha_factor)
            GL.glVertex3f(tx, ty, tz)
            # Bright head vertex
            GL.glColor4f(r, g, b, 0.70 * alpha_factor)
            GL.glVertex3f(hx, hy, hz)
        GL.glEnd()

        # 2. Render Glowing Photon Particle Heads (Subtle & Crisp)
        GL.glPointSize(2.2)
        GL.glBegin(GL.GL_POINTS)
        for p in self.in_flight_photons:
            u = p.progress
            hx = (1.0 - u) * p.start_x + u * p.target_x
            hy = (1.0 - u) * p.start_y + u * p.target_y
            hz = (1.0 - u) * p.start_z + u * p.target_z

            r, g, b = p.color
            GL.glColor4f(r, g, b, 0.95 * alpha_factor)
            GL.glVertex3f(hx, hy, hz)
        GL.glEnd()

        # White core dot
        GL.glPointSize(1.0)
        GL.glColor4f(1.0, 1.0, 1.0, 0.95 * alpha_factor)
        GL.glBegin(GL.GL_POINTS)
        for p in self.in_flight_photons:
            u = p.progress
            hx = (1.0 - u) * p.start_x + u * p.target_x
            hy = (1.0 - u) * p.start_y + u * p.target_y
            hz = (1.0 - u) * p.start_z + u * p.target_z
            GL.glVertex3f(hx, hy, hz)
        GL.glEnd()

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LIGHTING)

    def _render_screen_hits_3d(self, z_screen: float, alpha_factor: float = 1.0) -> None:
        """Renders accumulated photon impact dots directly on the 3D screen surface."""
        if not self.screen_hits_3d or alpha_factor <= 0.001:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)

        # Front surface of screen is at z_screen - 0.015
        z_face = z_screen - 0.016

        GL.glPointSize(1.8)
        GL.glBegin(GL.GL_POINTS)
        for x, y, col in self.screen_hits_3d:
            r, g, b = col
            GL.glColor4f(r, g, b, 0.70 * alpha_factor)
            GL.glVertex3f(x, y, z_face)
        GL.glEnd()

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LIGHTING)

    # ==========================================================================
    # Solid 3D Geometry Renderers (Pass 1)
    # ==========================================================================

    def _render_lab_grid(self) -> None:
        """Renders subtle laboratory floor grid (black in light mode for clear contrast)."""
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.0)
        if self.current_theme == "light":
            GL.glColor4f(0.04, 0.04, 0.06, 0.85)
        else:
            GL.glColor4f(0.12, 0.16, 0.22, 0.7)

        grid_min = -2.0
        grid_max = 4.0
        step = 0.4
        y_level = -0.35

        GL.glBegin(GL.GL_LINES)
        x = -2.0
        while x <= 2.0:
            GL.glVertex3f(x, y_level, grid_min)
            GL.glVertex3f(x, y_level, grid_max)
            x += step

        z = grid_min
        while z <= grid_max:
            GL.glVertex3f(-2.0, y_level, z)
            GL.glVertex3f(2.0, y_level, z)
            z += step
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _render_optical_rail(self, z_min: float, z_max: float) -> None:
        """Renders solid extruded aluminum optical rail along z-axis."""
        rail_w = 0.22
        rail_h = 0.08
        y_rail = -0.32
        is_light = (self.current_theme == "light")

        # Rail Top Face
        GL.glNormal3f(0.0, 1.0, 0.0)
        if is_light:
            GL.glColor4f(0.74, 0.79, 0.86, 1.0)
        else:
            GL.glColor4f(0.20, 0.24, 0.30, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-rail_w / 2, y_rail + rail_h, z_min)
        GL.glVertex3f(rail_w / 2, y_rail + rail_h, z_min)
        GL.glVertex3f(rail_w / 2, y_rail + rail_h, z_max)
        GL.glVertex3f(-rail_w / 2, y_rail + rail_h, z_max)
        GL.glEnd()

        # Rail Sides
        if is_light:
            GL.glColor4f(0.60, 0.65, 0.73, 1.0)
        else:
            GL.glColor4f(0.14, 0.17, 0.22, 1.0)
        GL.glBegin(GL.GL_QUADS)
        # Left side
        GL.glNormal3f(-1.0, 0.0, 0.0)
        GL.glVertex3f(-rail_w / 2, y_rail, z_min)
        GL.glVertex3f(-rail_w / 2, y_rail + rail_h, z_min)
        GL.glVertex3f(-rail_w / 2, y_rail + rail_h, z_max)
        GL.glVertex3f(-rail_w / 2, y_rail, z_max)
        # Right side
        GL.glNormal3f(1.0, 0.0, 0.0)
        GL.glVertex3f(rail_w / 2, y_rail, z_min)
        GL.glVertex3f(rail_w / 2, y_rail + rail_h, z_min)
        GL.glVertex3f(rail_w / 2, y_rail + rail_h, z_max)
        GL.glVertex3f(rail_w / 2, y_rail, z_max)
        GL.glEnd()

        # Millimeter Ticks along Rail Edge
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.0)
        if is_light:
            GL.glColor4f(0.04, 0.04, 0.06, 0.95)
        else:
            GL.glColor4f(0.45, 0.55, 0.70, 0.8)
        GL.glBegin(GL.GL_LINES)
        z = z_min
        while z <= z_max:
            GL.glVertex3f(rail_w / 2, y_rail + rail_h + 0.001, z)
            GL.glVertex3f(rail_w / 2 - 0.025, y_rail + rail_h + 0.001, z)
            z += 0.1
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _render_laser_head(self, z_pos: float, r: float, g: float, b: float) -> None:
        """Renders fully enclosed solid cylindrical laser head with front and rear caps."""
        y_beam = 0.0
        radius = 0.08
        length = 0.38
        segments = 32

        # Mounting Post connecting laser to rail
        self._render_mount_post(0.0, y_beam, z_pos + length / 2)

        # 1. Cylinder Outer Shell with Surface Normals for Smooth Curved Lighting
        GL.glColor4f(0.18, 0.22, 0.28, 1.0)
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            nx = math.cos(theta)
            ny = math.sin(theta)
            cx = radius * nx
            cy = y_beam + radius * ny

            GL.glNormal3f(nx, ny, 0.0)
            GL.glVertex3f(cx, cy, z_pos)
            GL.glVertex3f(cx, cy, z_pos + length)
        GL.glEnd()

        # 2. SOLID REAR CAP (z = z_pos, Normal pointing towards -Z)
        GL.glNormal3f(0.0, 0.0, -1.0)
        GL.glColor4f(0.12, 0.14, 0.18, 1.0)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0.0, y_beam, z_pos)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * (segments - i) / segments
            cx = radius * math.cos(theta)
            cy = y_beam + radius * math.sin(theta)
            GL.glVertex3f(cx, cy, z_pos)
        GL.glEnd()

        # Rear Details: Power socket on rear cap
        GL.glColor4f(0.28, 0.32, 0.40, 1.0)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0.0, y_beam, z_pos - 0.002)
        socket_r = 0.03
        for i in range(17):
            theta = 2.0 * math.pi * i / 16
            GL.glVertex3f(socket_r * math.cos(theta), y_beam + socket_r * math.sin(theta), z_pos - 0.002)
        GL.glEnd()

        # 3. SOLID FRONT CAP (z = z_pos + length, Normal pointing towards +Z)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glColor4f(0.12, 0.14, 0.18, 1.0)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0.0, y_beam, z_pos + length)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            cx = radius * math.cos(theta)
            cy = y_beam + radius * math.sin(theta)
            GL.glVertex3f(cx, cy, z_pos + length)
        GL.glEnd()

        # Front Glowing Emission Aperture (powered when ignition_progress > 0)
        GL.glDisable(GL.GL_LIGHTING)
        prog = self.ignition_progress
        z_nozzle = z_pos + length + 0.002
        ap_r = 0.025
        if prog > 0.001:
            GL.glColor4f(r, g, b, min(1.0, prog * 1.4))
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3f(0.0, y_beam, z_nozzle)
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                cx = ap_r * math.cos(theta)
                cy = y_beam + ap_r * math.sin(theta)
                GL.glVertex3f(cx, cy, z_nozzle)
            GL.glEnd()

            # Star Wars Plasma Surge at Nozzle when transitioning
            if 0.005 < prog < 0.995:
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                surge_alpha = 0.9 * math.sin(prog * math.pi)
                flare_r = ap_r * (1.6 + 0.5 * math.sin(prog * math.pi))
                GL.glBegin(GL.GL_TRIANGLE_FAN)
                GL.glColor4f(1.0, 1.0, 1.0, surge_alpha)
                GL.glVertex3f(0.0, y_beam, z_nozzle + 0.001)
                GL.glColor4f(r, g, b, 0.0)
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    GL.glVertex3f(flare_r * math.cos(theta), y_beam + flare_r * math.sin(theta), z_nozzle + 0.001)
                GL.glEnd()
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        else:
            GL.glColor4f(0.14, 0.16, 0.20, 1.0)
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3f(0.0, y_beam, z_nozzle)
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                cx = ap_r * math.cos(theta)
                cy = y_beam + ap_r * math.sin(theta)
                GL.glVertex3f(cx, cy, z_nozzle)
            GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _render_slit_mount(self, z_pos: float, r: float = 0.0, g: float = 1.0, b: float = 0.5) -> None:
        """Renders solid 3D aperture mount and plate with distinct slit openings."""
        self._render_mount_post(0.0, 0.0, z_pos)

        plate_w = 0.36
        plate_h = 0.36
        th = 0.02

        z_front = z_pos + th / 2.0
        z_back = z_pos - th / 2.0
        is_light = (self.current_theme == "light")

        # Render 6-Faced Solid Box for Aperture Plate
        if is_light:
            GL.glColor4f(0.24, 0.28, 0.36, 1.0)
        else:
            GL.glColor4f(0.14, 0.16, 0.22, 1.0)

        # Front Face (+Z)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_front)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_front)
        # Back Face (-Z)
        GL.glNormal3f(0.0, 0.0, -1.0)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_back)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_back)
        # Left Side (-X)
        GL.glNormal3f(-1.0, 0.0, 0.0)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_front)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_back)
        # Right Side (+X)
        GL.glNormal3f(1.0, 0.0, 0.0)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_back)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_front)
        # Top Side (+Y)
        GL.glNormal3f(0.0, 1.0, 0.0)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_back)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_back)
        GL.glEnd()

        # Dynamic Visual Geometry scaled from current physical parameters
        slit_w_vis, slit_h_vis, slit_sep_vis, pinhole_r_vis = self._get_visual_slit_geometry()

        if self.aperture_type == ApertureType.DOUBLE_SLIT:
            x1 = -slit_sep_vis / 2.0
            x2 = slit_sep_vis / 2.0
            self._render_slit_rect_3d(x1, 0.0, slit_w_vis, slit_h_vis, z_front, z_back, r, g, b)
            self._render_slit_rect_3d(x2, 0.0, slit_w_vis, slit_h_vis, z_front, z_back, r, g, b)

            if self.which_way_active:
                sensor_y = slit_h_vis / 2.0 + 0.025
                self._render_sensor_module(x1, sensor_y, z_front)
                self._render_sensor_module(x2, sensor_y, z_front)

        elif self.aperture_type == ApertureType.SINGLE_SLIT:
            self._render_slit_rect_3d(0.0, 0.0, slit_w_vis * 1.5, slit_h_vis, z_front, z_back, r, g, b)

        elif self.aperture_type == ApertureType.MULTI_SLIT:
            num = max(2, min(8, getattr(self, "num_slits", 5)))
            pitch = min(slit_sep_vis * 0.6, 0.24 / max(1, num - 1))
            x_start = -((num - 1) * pitch) / 2.0
            for k in range(num):
                xk = x_start + k * pitch
                self._render_slit_rect_3d(xk, 0.0, slit_w_vis * 0.75, slit_h_vis, z_front, z_back, r, g, b)

        elif self.aperture_type == ApertureType.CIRCULAR_PINHOLE:
            self._render_pinhole_3d(0.0, 0.0, pinhole_r_vis, z_front, z_back, r, g, b)

        # Outer Rim Frame Highlight on both front (+Z) and back (-Z) sides
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.5)
        if is_light:
            GL.glColor4f(0.15, 0.20, 0.30, 1.0)
        else:
            GL.glColor4f(0.35, 0.42, 0.55, 1.0)
        # Front Rim
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_front)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_front)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_front)
        GL.glEnd()
        # Back Rim
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(-plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(plate_w / 2, -plate_h / 2, z_back)
        GL.glVertex3f(plate_w / 2, plate_h / 2, z_back)
        GL.glVertex3f(-plate_w / 2, plate_h / 2, z_back)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _render_slit_rect_3d(
        self, cx: float, cy: float, w: float, h: float, z_front: float, z_back: float,
        r: float = 0.0, g: float = 1.0, b: float = 0.5
    ) -> None:
        """Renders physical rectangular slit opening with glowing illuminated borders on both sides."""
        # Dynamic transmitted laser glow through opening (smooth S-curve when laser reaches slits)
        if self.ignition_progress > 0.70:
            raw_t = (self.ignition_progress - 0.70) / 0.30
            trans = 3.0 * (raw_t ** 2) - 2.0 * (raw_t ** 3)
            r_hole = min(1.0, 0.04 + 0.75 * r * trans)
            g_hole = min(1.0, 0.05 + 0.75 * g * trans)
            b_hole = min(1.0, 0.07 + 0.75 * b * trans)
        else:
            r_hole, g_hole, b_hole = 0.04, 0.05, 0.07

        GL.glColor4f(r_hole, g_hole, b_hole, 1.0)
        GL.glBegin(GL.GL_QUADS)
        # Front Cutout (+Z)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_front + 0.001)
        GL.glVertex3f(cx + w / 2, cy - h / 2, z_front + 0.001)
        GL.glVertex3f(cx + w / 2, cy + h / 2, z_front + 0.001)
        GL.glVertex3f(cx - w / 2, cy + h / 2, z_front + 0.001)
        # Back Cutout (-Z)
        GL.glNormal3f(0.0, 0.0, -1.0)
        GL.glVertex3f(cx + w / 2, cy - h / 2, z_back - 0.001)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_back - 0.001)
        GL.glVertex3f(cx - w / 2, cy + h / 2, z_back - 0.001)
        GL.glVertex3f(cx + w / 2, cy + h / 2, z_back - 0.001)
        GL.glEnd()

        # Cyan illuminated aperture border on BOTH sides (Front + Back) & Interior Tunnel
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.5)
        GL.glColor4f(0.0, 0.85, 1.0, 0.85)

        # 1. Front Illuminated Rect (+Z Face)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_front + 0.003)
        GL.glVertex3f(cx + w / 2, cy - h / 2, z_front + 0.003)
        GL.glVertex3f(cx + w / 2, cy + h / 2, z_front + 0.003)
        GL.glVertex3f(cx - w / 2, cy + h / 2, z_front + 0.003)
        GL.glEnd()

        # 2. Back Illuminated Rect (-Z Face)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_back - 0.003)
        GL.glVertex3f(cx + w / 2, cy - h / 2, z_back - 0.003)
        GL.glVertex3f(cx + w / 2, cy + h / 2, z_back - 0.003)
        GL.glVertex3f(cx - w / 2, cy + h / 2, z_back - 0.003)
        GL.glEnd()

        # 3. Slit Interior Corner Edges connecting front to back
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_front + 0.003)
        GL.glVertex3f(cx - w / 2, cy - h / 2, z_back - 0.003)

        GL.glVertex3f(cx + w / 2, cy - h / 2, z_front + 0.003)
        GL.glVertex3f(cx + w / 2, cy - h / 2, z_back - 0.003)

        GL.glVertex3f(cx + w / 2, cy + h / 2, z_front + 0.003)
        GL.glVertex3f(cx + w / 2, cy + h / 2, z_back - 0.003)

        GL.glVertex3f(cx - w / 2, cy + h / 2, z_front + 0.003)
        GL.glVertex3f(cx - w / 2, cy + h / 2, z_back - 0.003)
        GL.glEnd()

        GL.glEnable(GL.GL_LIGHTING)

    def _render_pinhole_3d(
        self, cx: float, cy: float, radius: float, z_front: float, z_back: float,
        r: float = 0.0, g: float = 1.0, b: float = 0.5
    ) -> None:
        """Renders circular pinhole aperture opening with illuminated rings on both sides."""
        # Dynamic transmitted laser glow through circular aperture
        if self.ignition_progress > 0.70:
            raw_t = (self.ignition_progress - 0.70) / 0.30
            trans = 3.0 * (raw_t ** 2) - 2.0 * (raw_t ** 3)
            r_hole = min(1.0, 0.04 + 0.75 * r * trans)
            g_hole = min(1.0, 0.05 + 0.75 * g * trans)
            b_hole = min(1.0, 0.07 + 0.75 * b * trans)
        else:
            r_hole, g_hole, b_hole = 0.04, 0.05, 0.07

        GL.glColor4f(r_hole, g_hole, b_hole, 1.0)
        # Front circle cutout
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glVertex3f(cx, cy, z_front + 0.001)
        for i in range(17):
            th = 2.0 * math.pi * i / 16
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_front + 0.001)
        GL.glEnd()

        # Back circle cutout
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glNormal3f(0.0, 0.0, -1.0)
        GL.glVertex3f(cx, cy, z_back - 0.001)
        for i in range(17):
            th = 2.0 * math.pi * (16 - i) / 16
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_back - 0.001)
        GL.glEnd()

        # Cyan illuminated aperture rings on BOTH sides
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.5)
        GL.glColor4f(0.0, 0.85, 1.0, 0.85)

        # Front ring
        GL.glBegin(GL.GL_LINE_LOOP)
        for i in range(16):
            th = 2.0 * math.pi * i / 16
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_front + 0.003)
        GL.glEnd()

        # Back ring
        GL.glBegin(GL.GL_LINE_LOOP)
        for i in range(16):
            th = 2.0 * math.pi * i / 16
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_back - 0.003)
        GL.glEnd()

        # Connecting cylinder lines along depth
        GL.glBegin(GL.GL_LINES)
        for i in range(8):
            th = 2.0 * math.pi * i / 8
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_front + 0.003)
            GL.glVertex3f(cx + radius * math.cos(th), cy + radius * math.sin(th), z_back - 0.003)
        GL.glEnd()

        GL.glEnable(GL.GL_LIGHTING)

    def _render_sensor_module(self, cx: float, cy: float, z_front: float) -> None:
        """Renders which-way detector sensor block with warning LED."""
        bw = 0.04
        bh = 0.035
        bd = 0.025

        GL.glColor4f(0.24, 0.28, 0.36, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glVertex3f(cx - bw / 2, cy - bh / 2, z_front + bd)
        GL.glVertex3f(cx + bw / 2, cy - bh / 2, z_front + bd)
        GL.glVertex3f(cx + bw / 2, cy + bh / 2, z_front + bd)
        GL.glVertex3f(cx - bw / 2, cy + bh / 2, z_front + bd)
        GL.glEnd()

        GL.glDisable(GL.GL_LIGHTING)
        GL.glColor4f(1.0, 0.2, 0.2, 1.0)
        led_r = 0.007
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(cx, cy, z_front + bd + 0.002)
        for i in range(9):
            th = 2.0 * math.pi * i / 8
            GL.glVertex3f(cx + led_r * math.cos(th), cy + led_r * math.sin(th), z_front + bd + 0.002)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _render_detector_screen(
        self, z_pos: float, r: float, g: float, b: float
    ) -> None:
        """Renders 6-sided solid detector target screen."""
        self._render_mount_post(0.0, 0.0, z_pos)

        screen_w = 0.58
        screen_h = 0.58
        th = 0.03

        z_front = z_pos - th / 2.0
        z_back = z_pos + th / 2.0

        # Upload Live Screen Texture if data is ready
        if self._texture_dirty and self._intensity_2d is not None:
            self._upload_screen_texture(r, g, b)
            self._texture_dirty = False

        # Screen illumination smoothly fades based on laser beam reaching detector
        if self.ignition_progress <= 0.70:
            screen_fade = 0.0
        else:
            raw_fade = (self.ignition_progress - 0.70) / 0.30
            # Hermite smoothstep S-curve
            screen_fade = min(1.0, max(0.0, 3.0 * (raw_fade ** 2) - 2.0 * (raw_fade ** 3)))

        # 1. FRONT FACE (FACING -Z TOWARDS SLITS): TEXTURED WITH LIVE DIFFRACTION PATTERN
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._screen_texture_id)
        GL.glColor4f(screen_fade, screen_fade, screen_fade, 1.0)
        GL.glNormal3f(0.0, 0.0, -1.0)

        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0.0, 0.0)
        GL.glVertex3f(screen_w / 2, -screen_h / 2, z_front)
        GL.glTexCoord2f(1.0, 0.0)
        GL.glVertex3f(-screen_w / 2, -screen_h / 2, z_front)
        GL.glTexCoord2f(1.0, 1.0)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_front)
        GL.glTexCoord2f(0.0, 1.0)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_front)
        GL.glEnd()

        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)

        is_light = (self.current_theme == "light")

        # 2. SOLID OPAQUE BACK AND SIDES
        if is_light:
            GL.glColor4f(0.28, 0.32, 0.40, 1.0)
        else:
            GL.glColor4f(0.12, 0.15, 0.20, 1.0)
        GL.glBegin(GL.GL_QUADS)
        # Solid Back Face (+Z)
        GL.glNormal3f(0.0, 0.0, 1.0)
        GL.glVertex3f(-screen_w / 2, -screen_h / 2, z_back)
        GL.glVertex3f(screen_w / 2, -screen_h / 2, z_back)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_back)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_back)

        # Left Side (-X)
        GL.glNormal3f(-1.0, 0.0, 0.0)
        GL.glVertex3f(-screen_w / 2, -screen_h / 2, z_back)
        GL.glVertex3f(-screen_w / 2, -screen_h / 2, z_front)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_front)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_back)

        # Right Side (+X)
        GL.glNormal3f(1.0, 0.0, 0.0)
        GL.glVertex3f(screen_w / 2, -screen_h / 2, z_front)
        GL.glVertex3f(screen_w / 2, -screen_h / 2, z_back)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_back)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_front)

        # Top Side (+Y)
        GL.glNormal3f(0.0, 1.0, 0.0)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_front)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_front)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_back)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_back)
        GL.glEnd()

        # Solid Screen Frame Outer Highlight
        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(1.5)
        if is_light:
            GL.glColor4f(0.20, 0.25, 0.35, 1.0)
        else:
            GL.glColor4f(0.40, 0.48, 0.60, 1.0)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex3f(-screen_w / 2, -screen_h / 2, z_front)
        GL.glVertex3f(screen_w / 2, -screen_h / 2, z_front)
        GL.glVertex3f(screen_w / 2, screen_h / 2, z_front)
        GL.glVertex3f(-screen_w / 2, screen_h / 2, z_front)
        GL.glEnd()
        GL.glEnable(GL.GL_LIGHTING)

    def _upload_screen_texture(self, r: float, g: float, b: float) -> None:
        """Converts 2D intensity array into RGBA texture tinted to laser color."""
        if self._intensity_2d is None:
            return

        mat = self._intensity_2d
        h, w = mat.shape

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = np.clip(mat * r * 255.0 + 8.0, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 1] = np.clip(mat * g * 255.0 + 10.0, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 2] = np.clip(mat * b * 255.0 + 14.0, 0.0, 255.0).astype(np.uint8)
        rgba[:, :, 3] = 255

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._screen_texture_id)
        if (h, w) != (self._tex_h, self._tex_w):
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba.tobytes()
            )
            self._tex_w = w
            self._tex_h = h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rgba.tobytes()
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _render_mount_post(self, x: float, y_top: float, z: float) -> None:
        """Renders solid stainless steel mounting post connecting component to rail."""
        y_bottom = -0.32
        radius = 0.02
        segments = 16
        is_light = (self.current_theme == "light")

        if is_light:
            GL.glColor4f(0.45, 0.50, 0.58, 1.0)
        else:
            GL.glColor4f(0.50, 0.55, 0.65, 1.0)
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            nx = math.cos(theta)
            nz = math.sin(theta)
            cx = x + radius * nx
            cz = z + radius * nz

            GL.glNormal3f(nx, 0.0, nz)
            GL.glVertex3f(cx, y_bottom, cz)
            GL.glVertex3f(cx, y_top, cz)
        GL.glEnd()

        clamp_w = 0.11
        clamp_l = 0.09
        clamp_h = 0.03
        if is_light:
            GL.glColor4f(0.32, 0.36, 0.44, 1.0)
        else:
            GL.glColor4f(0.20, 0.24, 0.32, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glNormal3f(0.0, 1.0, 0.0)
        GL.glVertex3f(x - clamp_w / 2, y_bottom + clamp_h, z - clamp_l / 2)
        GL.glVertex3f(x + clamp_w / 2, y_bottom + clamp_h, z - clamp_l / 2)
        GL.glVertex3f(x + clamp_w / 2, y_bottom + clamp_h, z + clamp_l / 2)
        GL.glVertex3f(x - clamp_w / 2, y_bottom + clamp_h, z + clamp_l / 2)
        GL.glEnd()

    # ==========================================================================
    # Transparent Light Volumes (Pass 2)
    # ==========================================================================

    def _render_incident_beam(
        self, z_start: float, z_end: float, r: float, g: float, b: float
    ) -> None:
        """Renders collimated cylindrical laser beam with Star Wars lightsaber ignition/retraction."""
        prog = self.ignition_progress
        if prog <= 0.001:
            return

        GL.glDisable(GL.GL_LIGHTING)
        beam_r = 0.012
        segments = 16

        # Smooth easing for Star Wars blade extension/retraction
        smooth_p = 3.0 * (prog ** 2) - 2.0 * (prog ** 3)
        z_tip = z_start + smooth_p * (z_end - z_start)

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)

        # 1. Outer Cylindrical Plasma Beam Sheath
        sheath_alpha = 0.38 * min(1.0, prog * 1.5)
        GL.glColor4f(r, g, b, sheath_alpha)
        GL.glBegin(GL.GL_QUAD_STRIP)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            cx = beam_r * math.cos(theta)
            cy = beam_r * math.sin(theta)
            GL.glVertex3f(cx, cy, z_start)
            GL.glVertex3f(cx, cy, z_tip)
        GL.glEnd()

        # 2. Super-Intense Pure White Laser Core Line
        GL.glLineWidth(3.0)
        GL.glColor4f(1.0, 1.0, 1.0, min(1.0, prog * 1.6))
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(0.0, 0.0, z_start)
        GL.glVertex3f(0.0, 0.0, z_tip)
        GL.glEnd()

        # 3. STAR WARS LEADING PLASMA TIP / BLAST HEAD
        tip_radius = beam_r * (1.8 + 0.6 * math.sin(prog * math.pi))
        # Inner white plasma core disk
        GL.glColor4f(1.0, 1.0, 1.0, 0.95)
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glVertex3f(0.0, 0.0, z_tip + 0.002)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            GL.glVertex3f(beam_r * 0.9 * math.cos(theta), beam_r * 0.9 * math.sin(theta), z_tip + 0.002)
        GL.glEnd()

        # Outer plasma corona
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        GL.glColor4f(1.0, 1.0, 1.0, 0.85)
        GL.glVertex3f(0.0, 0.0, z_tip + 0.003)
        GL.glColor4f(r, g, b, 0.0)
        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            GL.glVertex3f(tip_radius * math.cos(theta), tip_radius * math.sin(theta), z_tip + 0.003)
        GL.glEnd()

        # Crackling Star Wars Energy Sparks at Tip during transition
        if 0.01 < prog < 0.98:
            GL.glLineWidth(1.5)
            import time as _t
            t_phase = _t.perf_counter() * 25.0
            GL.glBegin(GL.GL_LINES)
            for k in range(6):
                ang = k * (math.pi / 3.0) + math.sin(t_phase + k) * 0.3
                sp_len = tip_radius * (1.2 + 0.5 * math.cos(t_phase * 1.3 + k))
                GL.glColor4f(1.0, 1.0, 1.0, 0.9)
                GL.glVertex3f(0.0, 0.0, z_tip)
                GL.glColor4f(r, g, b, 0.0)
                GL.glVertex3f(sp_len * math.cos(ang), sp_len * math.sin(ang), z_tip - 0.01)
            GL.glEnd()

        # 4. Energy Impact Flare on Slits Back Face (when beam reaches slits, prog >= 0.70)
        if prog >= 0.70:
            raw_f = (prog - 0.70) / 0.30
            impact_f = 3.0 * (raw_f ** 2) - 2.0 * (raw_f ** 3)
            impact_r = 0.045 * impact_f
            GL.glColor4f(1.0, 1.0, 1.0, 0.9 * impact_f)
            GL.glBegin(GL.GL_TRIANGLE_FAN)
            GL.glVertex3f(0.0, 0.0, z_end - 0.001)
            GL.glColor4f(r, g, b, 0.0)
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                GL.glVertex3f(impact_r * math.cos(theta), impact_r * math.sin(theta), z_end - 0.001)
            GL.glEnd()

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LIGHTING)

    def _render_diffraction_cone(
        self, z_start: float, z_end: float, r: float, g: float, b: float, alpha_factor: float = 1.0
    ) -> None:
        """Renders volumetric expanding light cone from slits to screen."""
        if alpha_factor <= 0.001:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)

        slit_w_vis, slit_h_vis, slit_sep_vis, pinhole_r_vis = self._get_visual_slit_geometry()
        if self.aperture_type == ApertureType.DOUBLE_SLIT:
            w_start = max(0.04, slit_sep_vis + slit_w_vis * 1.5)
            h_start = max(0.04, slit_h_vis)
        elif self.aperture_type == ApertureType.SINGLE_SLIT:
            w_start = max(0.02, slit_w_vis * 2.0)
            h_start = max(0.04, slit_h_vis)
        elif self.aperture_type == ApertureType.CIRCULAR_PINHOLE:
            w_start = max(0.02, pinhole_r_vis * 2.2)
            h_start = max(0.02, pinhole_r_vis * 2.2)
        elif self.aperture_type == ApertureType.MULTI_SLIT:
            num = max(2, min(8, getattr(self, "num_slits", 5)))
            pitch = min(slit_sep_vis * 0.6, 0.24 / max(1, num - 1))
            span = (num - 1) * pitch
            w_start = max(0.04, span + slit_w_vis * 2.0)
            h_start = max(0.04, slit_h_vis)
        else:
            w_start = 0.06
            h_start = 0.12

        w_end = 0.44
        h_end = 0.44

        # 4-Sided Volumetric Frustum Sheath
        GL.glColor4f(r, g, b, 0.08 * alpha_factor)
        GL.glBegin(GL.GL_QUADS)
        # Top
        GL.glVertex3f(-w_start / 2, h_start / 2, z_start)
        GL.glVertex3f(w_start / 2, h_start / 2, z_start)
        GL.glVertex3f(w_end / 2, h_end / 2, z_end)
        GL.glVertex3f(-w_end / 2, h_end / 2, z_end)
        # Bottom
        GL.glVertex3f(-w_start / 2, -h_start / 2, z_start)
        GL.glVertex3f(w_start / 2, -h_start / 2, z_start)
        GL.glVertex3f(w_end / 2, -h_end / 2, z_end)
        GL.glVertex3f(-w_end / 2, -h_end / 2, z_end)
        # Left
        GL.glVertex3f(-w_start / 2, -h_start / 2, z_start)
        GL.glVertex3f(-w_start / 2, h_start / 2, z_start)
        GL.glVertex3f(-w_end / 2, h_end / 2, z_end)
        GL.glVertex3f(-w_end / 2, -h_end / 2, z_end)
        # Right
        GL.glVertex3f(w_start / 2, -h_start / 2, z_start)
        GL.glVertex3f(w_start / 2, h_start / 2, z_start)
        GL.glVertex3f(w_end / 2, h_end / 2, z_end)
        GL.glVertex3f(w_end / 2, -h_end / 2, z_end)
        GL.glEnd()

        # Raylines connecting corners
        GL.glLineWidth(1.0)
        GL.glColor4f(r, g, b, 0.3 * alpha_factor)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(-w_start / 2, 0.0, z_start)
        GL.glVertex3f(-w_end / 2, 0.0, z_end)
        GL.glVertex3f(w_start / 2, 0.0, z_start)
        GL.glVertex3f(w_end / 2, 0.0, z_end)
        GL.glEnd()

        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_LIGHTING)

    # ==========================================================================
    # Heads-Up Display (HUD)
    # ==========================================================================

    def paintEvent(self, event) -> None:
        """Renders 3D OpenGL scene and overlays 2D HUD text via QPainter."""
        super().paintEvent(event)

        if not self.show_hud:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        is_light = (self.current_theme == "light")

        # Top-Left HUD Badge
        hud_w = 250
        hud_h = 68
        if is_light:
            painter.fillRect(15, 15, hud_w, hud_h, QColor(237, 243, 248, 235))
            painter.setPen(QColor(148, 163, 184))
            painter.drawRoundedRect(15, 15, hud_w, hud_h, 6, 6)

            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(3, 105, 161))  # High-contrast Sky-700
            painter.drawText(25, 33, "3D OPTICAL BENCH")

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(4, 120, 87))  # High-contrast Emerald-700
            painter.drawText(185, 33, f"{self.current_fps:.0f} FPS")

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(30, 41, 59))  # Dark slate
            painter.drawText(25, 50, f"Screen Distance L: {self.distance_l_m:.2f} m")

            if self.ignition_progress <= 0.001:
                mode_str = "Laser Power: OFF (Standby)"
            elif self.ignition_progress < 1.0:
                mode_str = "Laser: Igniting..." if self._target_laser_power else "Laser: Powering down..."
            else:
                mode_str = "Classical Wave" if self.mode == SimulationMode.CLASSICAL_WAVE else f"Quantum In-Flight ({len(self.in_flight_photons)} particles)"
            painter.drawText(25, 66, mode_str)
        else:
            painter.fillRect(15, 15, hud_w, hud_h, QColor(18, 22, 30, 200))
            painter.setPen(QColor(46, 55, 72))
            painter.drawRoundedRect(15, 15, hud_w, hud_h, 6, 6)

            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(0, 210, 255))
            painter.drawText(25, 33, "3D OPTICAL BENCH")

            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(QColor(16, 185, 129))
            painter.drawText(185, 33, f"{self.current_fps:.0f} FPS")

            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(160, 175, 200))
            painter.drawText(25, 50, f"Screen Distance L: {self.distance_l_m:.2f} m")

            if self.ignition_progress <= 0.001:
                mode_str = "Laser Power: OFF (Standby)"
            elif self.ignition_progress < 1.0:
                mode_str = "Laser: Igniting..." if self._target_laser_power else "Laser: Powering down..."
            else:
                mode_str = "Classical Wave" if self.mode == SimulationMode.CLASSICAL_WAVE else f"Quantum In-Flight ({len(self.in_flight_photons)} particles)"
            painter.drawText(25, 66, mode_str)

        # Bottom-Right Navigation Hint
        hint_w = 265
        hint_h = 28
        hx = self.width() - hint_w - 15
        hy = self.height() - hint_h - 15
        if is_light:
            painter.fillRect(hx, hy, hint_w, hint_h, QColor(237, 243, 248, 225))
            painter.setPen(QColor(148, 163, 184))
            painter.drawRoundedRect(hx, hy, hint_w, hint_h, 5, 5)
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(51, 65, 85))
            painter.drawText(hx + 10, hy + 18, "Orbit: Left-Drag | Pan: Right-Drag | Zoom: Scroll")
        else:
            painter.fillRect(hx, hy, hint_w, hint_h, QColor(18, 22, 30, 180))
            painter.setPen(QColor(46, 55, 72))
            painter.drawRoundedRect(hx, hy, hint_w, hint_h, 5, 5)
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(140, 155, 180))
            painter.drawText(hx + 10, hy + 18, "Orbit: Left-Drag | Pan: Right-Drag | Zoom: Scroll")

        painter.end()

    # ==========================================================================
    # Mouse Orbit & Pan Handlers
    # ==========================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Detects mouse press for orbit (left) or pan (right)."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_mode = "ORBIT"
            self._last_mouse_pos = event.position()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._mouse_mode = "PAN"
            self._last_mouse_pos = event.position()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handles 3D camera orbit rotation or panning."""
        if self._mouse_mode is None:
            return

        delta = event.position() - self._last_mouse_pos
        self._last_mouse_pos = event.position()

        if self._mouse_mode == "ORBIT":
            self.camera_yaw += float(delta.x()) * 0.4
            self.camera_pitch += float(delta.y()) * 0.4
            self.camera_pitch = max(-85.0, min(85.0, self.camera_pitch))

        elif self._mouse_mode == "PAN":
            pan_speed = 0.003 * (self.camera_distance / 4.0)
            rad_yaw = math.radians(self.camera_yaw)

            dx = float(delta.x()) * pan_speed
            dy = float(delta.y()) * pan_speed

            self.target_x -= dx * math.cos(rad_yaw)
            self.target_z -= dx * math.sin(rad_yaw)
            self.target_y += dy

        self.camera_changed.emit()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Ends mouse interaction."""
        self._mouse_mode = None
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zooms camera in/out along line of sight."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.camera_distance = max(0.8, self.camera_distance * 0.9)
        elif delta < 0:
            self.camera_distance = min(15.0, self.camera_distance * 1.1)

        self.camera_changed.emit()
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double click resets camera to default view."""
        self.reset_camera()
        event.accept()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = GLApparatusWidget()
    widget.setWindowTitle("3D Apparatus Bench (Standalone Viewport)")
    widget.resize(960, 680)
    widget.show()
    sys.exit(app.exec())
