"""Configuration module for the Double-Slit Experiment Simulation.

This module defines physical constants, parameter bounds, presets, dataclasses,
and visual styling options used across both the GPU physics engine and the
PyQt6 user interface.

All physical quantities in the physics engine adhere strictly to SI units:
    - Lengths, positions, wavelengths: meters (m)
    - Energies: Joules (J)
    - Times: seconds (s)
    - Frequencies: Hertz (Hz)
Conversion helpers and GUI slider scalers are provided to translate between
display units (nanometers, micrometers, millimeters) and SI units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


# ==============================================================================
# 1. Fundamental Physical Constants (SI Units)
# ==============================================================================

SPEED_OF_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum c, in meters per second (m/s)."""

PLANCK_CONSTANT: float = 6.626_070_15e-34
"""Planck's constant h, in Joule-seconds (J*s)."""

HBAR: float = PLANCK_CONSTANT / (2.0 * 3.141592653589793)
"""Reduced Planck's constant h-bar = h / (2*pi), in Joule-seconds (J*s)."""


# ==============================================================================
# 2. Simulation Enumerations
# ==============================================================================

class SimulationMode(Enum):
    """Execution mode of the simulation pipeline."""

    CLASSICAL_WAVE = "Classical Wave (Angular Spectrum)"
    """Continuous wave optics using 3D Helmholtz propagation via 2D FFT."""

    QUANTUM_MONTE_CARLO = "Quantum Monte Carlo (Photons)"
    """Probabilistic single-particle hit buildup sampled from |psi(x, y)|^2."""


class ApertureType(Enum):
    """Geometric profile of the aperture plate."""

    DOUBLE_SLIT = "Double Slit"
    """Two identical rectangular slits separated by distance d."""

    SINGLE_SLIT = "Single Slit"
    """A single rectangular slit of width a."""

    MULTI_SLIT = "Diffraction Grating"
    """Array of N equally spaced slits."""

    CIRCULAR_PINHOLE = "Circular Pinhole"
    """Circular opening of diameter a (Airy disk diffraction)."""


# ==============================================================================
# 3. Numeric Parameter Specification and Ranges
# ==============================================================================

@dataclass(frozen=True)
class ParameterRange:
    """Defines numeric boundaries, display scaling, and metadata for a parameter.

    Attributes:
        min_val: Minimum permitted value in display units.
        max_val: Maximum permitted value in display units.
        default_val: Default starting value in display units.
        step: Minimum increment step for UI sliders/spinboxes.
        decimals: Number of decimal places to display in the UI.
        display_unit: Text abbreviation of the unit (e.g. 'nm', 'um', 'mm', 'm').
        to_si_scale: Multiplier to convert display unit to SI meters (e.g. 1e-9 for nm).
        description: Human-readable explanation of the physical parameter.
    """

    min_val: float
    max_val: float
    default_val: float
    step: float
    decimals: int
    display_unit: str
    to_si_scale: float
    description: str

    def to_si(self, display_value: float) -> float:
        """Converts a value from GUI display units to SI meters."""
        return display_value * self.to_si_scale

    def from_si(self, si_value: float) -> float:
        """Converts a value from SI meters to GUI display units."""
        return si_value / self.to_si_scale


# Global parameter bounds used by control widgets
PARAMETER_RANGES: Dict[str, ParameterRange] = {
    "wavelength": ParameterRange(
        min_val=380.0,
        max_val=750.0,
        default_val=532.0,
        step=1.0,
        decimals=0,
        display_unit="nm",
        to_si_scale=1e-9,
        description="Wavelength of coherent illumination (visible spectrum: 380-750 nm)",
    ),
    "slit_width": ParameterRange(
        min_val=2.0,
        max_val=200.0,
        default_val=20.0,
        step=1.0,
        decimals=1,
        display_unit="µm",
        to_si_scale=1e-6,
        description="Width of each individual slit aperture along x-axis",
    ),
    "slit_separation": ParameterRange(
        min_val=10.0,
        max_val=1000.0,
        default_val=100.0,
        step=5.0,
        decimals=1,
        display_unit="µm",
        to_si_scale=1e-6,
        description="Center-to-center distance between adjacent slits",
    ),
    "slit_height": ParameterRange(
        min_val=0.1,
        max_val=10.0,
        default_val=2.0,
        step=0.1,
        decimals=2,
        display_unit="mm",
        to_si_scale=1e-3,
        description="Vertical height of the rectangular slits along y-axis",
    ),
    "distance_l": ParameterRange(
        min_val=0.05,
        max_val=2.5,
        default_val=0.5,
        step=0.01,
        decimals=3,
        display_unit="m",
        to_si_scale=1.0,
        description="Optical propagation distance from slit plate to detector screen",
    ),
    "beam_radius": ParameterRange(
        min_val=0.2,
        max_val=5.0,
        default_val=1.2,
        step=0.1,
        decimals=2,
        display_unit="mm",
        to_si_scale=1e-3,
        description="Waist radius (w0) of the incident Gaussian laser beam",
    ),
    "screen_span": ParameterRange(
        min_val=5.0,
        max_val=60.0,
        default_val=25.0,
        step=1.0,
        decimals=1,
        display_unit="mm",
        to_si_scale=1e-3,
        description="Physical width and height of the square observation window",
    ),
    "photon_rate": ParameterRange(
        min_val=10.0,
        max_val=10000.0,
        default_val=500.0,
        step=50.0,
        decimals=0,
        display_unit="ph/s",
        to_si_scale=1.0,
        description="Emission rate of individual photons in Monte Carlo quantum mode",
    ),
}


# ==============================================================================
# 4. Simulation State & Parameter Container
# ==============================================================================

@dataclass
class SimulationConfig:
    """Complete parameter bundle required to compute the physical diffraction field.

    All stored attributes are maintained in SI standard units (meters, seconds)
    to prevent dimensional inconsistency across mathematical routines.
    """

    # Optical Source
    wavelength: float = 532e-9
    """Optical wavelength in meters (default: 532 nm green diode)."""

    beam_radius: float = 1.2e-3
    """1/e^2 beam waist radius in meters (default: 1.2 mm)."""

    # Aperture Geometry
    aperture_type: ApertureType = ApertureType.DOUBLE_SLIT
    """Aperture geometry (single slit, double slit, multi-slit, circular)."""

    slit_width: float = 20e-6
    """Slit width 'a' in meters (default: 20 µm)."""

    slit_separation: float = 100e-6
    """Slit separation 'd' in meters (default: 100 µm)."""

    slit_height: float = 2.0e-3
    """Slit height 'h' in meters (default: 2.0 mm)."""

    num_slits: int = 2
    """Number of slits when aperture_type is MULTI_SLIT or DOUBLE_SLIT."""

    # Propagation Geometry
    distance_l: float = 0.5
    """Propagation distance from aperture to screen in meters (default: 0.5 m)."""

    screen_span: float = 25e-3
    """Full width and height of the detector window in meters (default: 25 mm)."""

    # Grid Resolution
    grid_size: int = 1024
    """Number of grid sample points along each spatial axis (e.g. 1024 x 1024)."""

    # Simulation Mode & Quantum State
    mode: SimulationMode = SimulationMode.CLASSICAL_WAVE
    """Active simulation paradigm (continuous wave vs. quantum Monte Carlo)."""

    which_way_detected: bool = False
    """When True, quantum state collapse removes phase coherence between slits."""

    photon_rate: int = 500
    """Number of photon detection events generated per second in Monte Carlo mode."""

    def get_wavenumber(self) -> float:
        """Returns the optical angular wavenumber k = 2 * pi / lambda."""
        return (2.0 * 3.141592653589793) / self.wavelength

    def get_fringe_spacing(self) -> float:
        """Calculates theoretical Young's fringe spacing Delta_x = lambda * L / d.

        Returns:
            Fringe spacing in meters, or 0.0 if not applicable.
        """
        if self.slit_separation <= 0:
            return 0.0
        return (self.wavelength * self.distance_l) / self.slit_separation

    def get_central_envelope_width(self) -> float:
        """Calculates the full width between first diffraction minima: 2 * lambda * L / a.

        Returns:
            Central diffraction envelope width in meters.
        """
        if self.slit_width <= 0:
            return 0.0
        return 2.0 * (self.wavelength * self.distance_l) / self.slit_width


# ==============================================================================
# 5. Experimental Demonstration Presets
# ==============================================================================

@dataclass(frozen=True)
class ExperimentPreset:
    """Pre-configured experimental scenario for quick classroom demonstration."""

    name: str
    description: str
    wavelength_nm: float
    slit_width_um: float
    slit_separation_um: float
    slit_height_mm: float
    distance_l_m: float
    aperture_type: ApertureType
    num_slits: int = 2
    which_way_detected: bool = False


PRESETS: Dict[str, ExperimentPreset] = {
    "green_laser_standard": ExperimentPreset(
        name="Standard Green Laser (532 nm)",
        description="Standard textbook double-slit interference with crisp visible fringes.",
        wavelength_nm=532.0,
        slit_width_um=25.0,
        slit_separation_um=120.0,
        slit_height_mm=2.5,
        distance_l_m=0.8,
        aperture_type=ApertureType.DOUBLE_SLIT,
    ),
    "red_hene_laser": ExperimentPreset(
        name="He-Ne Red Laser (632.8 nm)",
        description="Classic helium-neon gas laser demonstration with wider fringe separation.",
        wavelength_nm=632.8,
        slit_width_um=30.0,
        slit_separation_um=150.0,
        slit_height_mm=3.0,
        distance_l_m=1.0,
        aperture_type=ApertureType.DOUBLE_SLIT,
    ),
    "single_slit_diffraction": ExperimentPreset(
        name="Single Slit Diffraction",
        description="Pure Fraunhofer single-slit envelope illustrating central and secondary maxima.",
        wavelength_nm=532.0,
        slit_width_um=40.0,
        slit_separation_um=0.0,
        slit_height_mm=3.0,
        distance_l_m=0.6,
        aperture_type=ApertureType.SINGLE_SLIT,
        num_slits=1,
    ),
    "diffraction_grating": ExperimentPreset(
        name="Diffraction Grating (N = 5)",
        description="Multi-slit grating showing ultra-sharp principal interference peaks.",
        wavelength_nm=450.0,
        slit_width_um=15.0,
        slit_separation_um=80.0,
        slit_height_mm=2.0,
        distance_l_m=0.7,
        aperture_type=ApertureType.MULTI_SLIT,
        num_slits=5,
    ),
    "fresnel_near_field": ExperimentPreset(
        name="Fresnel Near-Field Regime",
        description="Near-field diffraction showing geometric shadow and Fresnel fringe ripples.",
        wavelength_nm=532.0,
        slit_width_um=80.0,
        slit_separation_um=300.0,
        slit_height_mm=4.0,
        distance_l_m=0.08,
        aperture_type=ApertureType.DOUBLE_SLIT,
    ),
    "quantum_decoherence": ExperimentPreset(
        name="Quantum Decoherence (Which-Way Active)",
        description="Path detector active: wavefunction collapse destroys the interference fringes.",
        wavelength_nm=532.0,
        slit_width_um=25.0,
        slit_separation_um=120.0,
        slit_height_mm=2.5,
        distance_l_m=0.8,
        aperture_type=ApertureType.DOUBLE_SLIT,
        which_way_detected=True,
    ),
}


# ==============================================================================
# 6. UI Theme & Scientific Styling Palette
# ==============================================================================

class DarkPalette:
    """Scientific dark theme palette."""

    is_dark: bool = True
    bg_main: str = "#12151c"
    surface_card: str = "#1a1f2b"
    surface_input: str = "#222938"
    border_color: str = "#2e3748"
    border_active: str = "#4d7cfe"

    text_primary: str = "#f0f4f8"
    text_secondary: str = "#94a3b8"
    text_muted: str = "#64748b"

    accent_cyan: str = "#00d2ff"
    accent_green: str = "#10b981"
    accent_violet: str = "#8b5cf6"
    accent_amber: str = "#f59e0b"
    accent_red: str = "#ef4444"

    plot_bg: str = "#0d1117"
    plot_grid: str = "#21262d"
    plot_theoretical: str = "#f59e0b"
    plot_simulated: str = "#00d2ff"
    plot_histogram: str = "#10b981"

    arrow_up_file: str = "arrow_up_dark.png"
    arrow_down_file: str = "arrow_down_dark.png"
    arrow_up_hover_file: str = "arrow_up_hover.png"
    arrow_down_hover_file: str = "arrow_down_hover.png"

    # High-contrast scrollbar properties
    scrollbar_track: str = "#161b24"
    scrollbar_handle: str = "#475569"
    scrollbar_handle_hover: str = "#00d2ff"
    scrollbar_handle_pressed: str = "#38bdf8"

    # Compatibility aliases
    BACKGROUND_DARK: str = "#12151c"
    SURFACE_CARD: str = "#1a1f2b"
    SURFACE_INPUT: str = "#222938"
    BORDER_COLOR: str = "#2e3748"
    BORDER_ACTIVE: str = "#4d7cfe"
    TEXT_PRIMARY: str = "#f0f4f8"
    TEXT_SECONDARY: str = "#94a3b8"
    TEXT_MUTED: str = "#64748b"
    ACCENT_CYAN: str = "#00d2ff"
    ACCENT_GREEN: str = "#10b981"
    ACCENT_VIOLET: str = "#8b5cf6"
    ACCENT_AMBER: str = "#f59e0b"
    ACCENT_RED: str = "#ef4444"
    PLOT_BACKGROUND: str = "#0d1117"
    PLOT_GRID: str = "#21262d"
    PLOT_THEORETICAL: str = "#f59e0b"
    PLOT_SIMULATED: str = "#00d2ff"
    PLOT_HISTOGRAM: str = "#10b981"


class LightPalette:
    """Scientific eye-friendly light blue palette."""

    is_dark: bool = False
    bg_main: str = "#dbe6f0"
    surface_card: str = "#edf3f8"
    surface_input: str = "#f8fbfe"
    border_color: str = "#94a3b8"
    border_active: str = "#1d4ed8"

    text_primary: str = "#0f172a"
    text_secondary: str = "#334155"
    text_muted: str = "#64748b"

    accent_cyan: str = "#0369a1"
    accent_green: str = "#047857"
    accent_violet: str = "#6d28d9"
    accent_amber: str = "#b45309"
    accent_red: str = "#b91c1c"

    plot_bg: str = "#edf3f8"
    plot_grid: str = "#000000"
    plot_theoretical: str = "#b45309"
    plot_simulated: str = "#1d4ed8"
    plot_histogram: str = "#047857"

    arrow_up_file: str = "arrow_up_light.png"
    arrow_down_file: str = "arrow_down_light.png"
    arrow_up_hover_file: str = "arrow_up_light_hover.png"
    arrow_down_hover_file: str = "arrow_down_light_hover.png"

    # High-contrast scrollbar properties
    scrollbar_track: str = "#c8d7e6"
    scrollbar_handle: str = "#64748b"
    scrollbar_handle_hover: str = "#1d4ed8"
    scrollbar_handle_pressed: str = "#1e40af"

    # Compatibility aliases
    BACKGROUND_DARK: str = "#dbe6f0"
    SURFACE_CARD: str = "#edf3f8"
    SURFACE_INPUT: str = "#f8fbfe"
    BORDER_COLOR: str = "#94a3b8"
    BORDER_ACTIVE: str = "#1d4ed8"
    TEXT_PRIMARY: str = "#0f172a"
    TEXT_SECONDARY: str = "#334155"
    TEXT_MUTED: str = "#475569"
    ACCENT_CYAN: str = "#0369a1"
    ACCENT_GREEN: str = "#047857"
    ACCENT_VIOLET: str = "#6d28d9"
    ACCENT_AMBER: str = "#b45309"
    ACCENT_RED: str = "#b91c1c"
    PLOT_BACKGROUND: str = "#edf3f8"
    PLOT_GRID: str = "#000000"
    PLOT_THEORETICAL: str = "#b45309"
    PLOT_SIMULATED: str = "#1d4ed8"
    PLOT_HISTOGRAM: str = "#047857"


# Default backwards-compatible alias
UIColors = DarkPalette

from pathlib import Path as _Path
_RESOURCE_DIR = _Path(__file__).resolve().parent / "resources"


def get_stylesheet(theme: str = "dark") -> str:
    """Generates a complete Qt stylesheet for dark or light theme."""
    pal = LightPalette if theme.lower() == "light" else DarkPalette

    arrow_up = (_RESOURCE_DIR / pal.arrow_up_file).as_posix()
    arrow_down = (_RESOURCE_DIR / pal.arrow_down_file).as_posix()
    arrow_up_hover = (_RESOURCE_DIR / pal.arrow_up_hover_file).as_posix()
    arrow_down_hover = (_RESOURCE_DIR / pal.arrow_down_hover_file).as_posix()

    return f"""
QMainWindow {{
    background-color: {pal.bg_main};
}}
QWidget {{
    color: {pal.text_primary};
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {pal.surface_card};
    border: 1px solid {pal.border_color};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: {pal.accent_cyan};
}}
QLabel {{
    color: {pal.text_secondary};
}}
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {pal.surface_input};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {pal.border_active};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {pal.text_primary};
    border: 2px solid {pal.border_active};
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {pal.accent_cyan};
}}
QDoubleSpinBox, QSpinBox {{
    background-color: {pal.surface_input};
    border: 1px solid {pal.border_color};
    border-radius: 5px;
    padding: 3px 2px 3px 6px;
    color: {pal.text_primary};
    selection-background-color: {pal.border_active};
}}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 17px;
    border-left: 1px solid {pal.border_color};
    border-bottom: 1px solid {pal.border_color};
    background-color: {pal.surface_card};
}}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 17px;
    border-left: 1px solid {pal.border_color};
    background-color: {pal.surface_card};
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: {pal.border_color};
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    image: url({arrow_up});
    width: 8px;
    height: 5px;
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    image: url({arrow_down});
    width: 8px;
    height: 5px;
}}
QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{
    image: url({arrow_up_hover});
}}
QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{
    image: url({arrow_down_hover});
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {pal.border_active};
}}
QComboBox {{
    background-color: {pal.surface_input};
    border: 1px solid {pal.border_color};
    border-radius: 5px;
    padding: 4px 26px 4px 8px;
    color: {pal.text_primary};
    font-weight: 500;
}}
QComboBox:hover {{
    border-color: {pal.border_active};
}}
QComboBox:focus {{
    border: 1px solid {pal.border_active};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {pal.border_color};
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background-color: {pal.surface_card};
}}
QComboBox::drop-down:hover {{
    background-color: {pal.border_color};
}}
QComboBox::down-arrow {{
    image: url({arrow_down});
    width: 8px;
    height: 5px;
}}
QComboBox::down-arrow:hover {{
    image: url({arrow_down_hover});
}}
QComboBox QAbstractItemView, QListView {{
    background-color: {pal.surface_card};
    border: 1px solid {pal.border_color};
    border-radius: 6px;
    color: {pal.text_primary};
    selection-background-color: {pal.border_active};
    selection-color: #ffffff;
    padding: 4px;
    outline: none;
}}
QComboBox QAbstractItemView::item, QListView::item {{
    min-height: 24px;
    padding: 4px 8px;
    border-radius: 4px;
    color: {pal.text_primary};
    background-color: {pal.surface_card};
}}
QComboBox QAbstractItemView::item:selected, QComboBox QAbstractItemView::item:hover,
QListView::item:selected, QListView::item:hover {{
    background-color: {pal.border_active};
    color: #ffffff;
}}
QPushButton {{
    background-color: {pal.surface_input};
    border: 1px solid {pal.border_color};
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
    color: {pal.text_primary};
}}
QPushButton:hover {{
    background-color: {pal.border_color};
    border-color: {pal.border_active};
}}
QPushButton:pressed {{
    background-color: {pal.border_active};
    color: #ffffff;
}}
QRadioButton {{
    spacing: 8px;
    color: {pal.text_primary};
    font-weight: 500;
}}
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 1px solid {pal.border_color};
    background-color: {pal.surface_input};
}}
QRadioButton::indicator:hover {{
    border-color: {pal.border_active};
}}
QRadioButton::indicator:checked {{
    background-color: {pal.border_active};
    border: 3px solid {pal.surface_card};
}}
QScrollBar:vertical {{
    background: {pal.scrollbar_track};
    width: 13px;
    margin: 2px 2px 2px 2px;
    border-radius: 5px;
    border: 1px solid {pal.border_color};
}}
QScrollBar::handle:vertical {{
    background: {pal.scrollbar_handle};
    min-height: 28px;
    border-radius: 4px;
    margin: 1px;
}}
QScrollBar::handle:vertical:hover {{
    background: {pal.scrollbar_handle_hover};
}}
QScrollBar::handle:vertical:pressed {{
    background: {pal.scrollbar_handle_pressed};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: {pal.scrollbar_track};
    height: 13px;
    margin: 2px 2px 2px 2px;
    border-radius: 5px;
    border: 1px solid {pal.border_color};
}}
QScrollBar::handle:horizontal {{
    background: {pal.scrollbar_handle};
    min-width: 28px;
    border-radius: 4px;
    margin: 1px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {pal.scrollbar_handle_hover};
}}
QScrollBar::handle:horizontal:pressed {{
    background: {pal.scrollbar_handle_pressed};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}
QSplitter::handle {{
    background-color: {pal.border_color};
}}
QSplitter::handle:hover {{
    background-color: {pal.border_active};
}}
QToolTip {{
    background-color: {pal.surface_card};
    color: {pal.text_primary};
    border: 1px solid {pal.border_color};
    border-radius: 4px;
    padding: 4px 8px;
}}
QCheckBox {{
    spacing: 8px;
    color: {pal.text_primary};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {pal.border_color};
    background-color: {pal.surface_input};
}}
QCheckBox::indicator:checked {{
    background-color: {pal.border_active};
    border-color: {pal.border_active};
}}
QStatusBar {{
    background-color: {pal.bg_main};
    border-top: 1px solid {pal.border_color};
    color: {pal.text_secondary};
}}
QMenuBar {{
    background-color: {pal.bg_main};
    border-bottom: 1px solid {pal.border_color};
    color: {pal.text_primary};
}}
QMenuBar::item {{
    background-color: transparent;
    padding: 5px 10px;
}}
QMenuBar::item:selected {{
    background-color: {pal.surface_card};
    border-radius: 4px;
}}
QMenu {{
    background-color: {pal.surface_card};
    border: 1px solid {pal.border_color};
    border-radius: 6px;
    color: {pal.text_primary};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 20px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {pal.border_active};
    color: #ffffff;
}}
QTabWidget::pane {{
    border: 1px solid {pal.border_color};
    border-radius: 6px;
    background-color: {pal.bg_main};
}}
QTabBar::tab {{
    background-color: {pal.surface_card};
    color: {pal.text_secondary};
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {pal.border_active};
    color: #ffffff;
}}
"""


# Common Qt Stylesheet string (default dark mode)
QT_STYLE_SHEET: str = get_stylesheet("dark")
