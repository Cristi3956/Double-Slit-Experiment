"""Interactive parameter control panel widget for the Double-Slit Laboratory.

This module provides a comprehensive PyQt6 control panel widget with:
1. Synchronized Dual Inputs:
   Every physical parameter is controlled via a smooth high-resolution QSlider
   synchronized bidirectionally with a high-precision QDoubleSpinBox.
2. Parameter Sections:
   - Coherent Laser Source: Wavelength lambda (with live RGB color swatch), Beam Waist w0.
   - Aperture Geometry: Type (Double, Single, Multi-Slit, Pinhole), Slit Width a,
     Separation d, Height h, Number of Slits N.
   - Optical Propagation: Distance to Screen L, Observation Screen Span.
   - Simulation & Quantum State: Classical Wave vs. Quantum Photons,
     Which-Way Detector toggle, Photon Emission Rate, Reset Hits.
3. Demonstration Presets:
   One-click presets (Green Laser, He-Ne Red, Single Slit, Grating, Near-Field, Decoherence).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.config import (
    PARAMETER_RANGES,
    PRESETS,
    ApertureType,
    ParameterRange,
    SimulationConfig,
    SimulationMode,
    UIColors,
)
from src.rendering.color_utils import wavelength_to_hex


class DualInputControl(QWidget):
    """Synchronized slider and spinbox widget for a physical parameter."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        range_def: ParameterRange,
        parent=None,
    ) -> None:
        """Initializes a dual slider-spinbox input row.

        Args:
            range_def: ParameterRange instance defining min, max, default, and units.
        """
        super().__init__(parent)
        self.range_def: ParameterRange = range_def
        self._block_signals: bool = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        # High-resolution integer slider scaling: 1000 steps across range
        self._slider_steps: int = 1000
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, self._slider_steps)

        # Precision Numeric SpinBox
        self.spinbox = QDoubleSpinBox(self)
        self.spinbox.setRange(range_def.min_val, range_def.max_val)
        self.spinbox.setSingleStep(range_def.step)
        self.spinbox.setDecimals(range_def.decimals)
        self.spinbox.setSuffix(f" {range_def.display_unit}")
        self.spinbox.setFixedWidth(98)

        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spinbox)

        # Connect synchronization signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

        # Set initial default value
        self.set_display_value(range_def.default_val)

    def get_display_value(self) -> float:
        """Returns the current value in GUI display units."""
        return float(self.spinbox.value())

    def get_si_value(self) -> float:
        """Returns the current value converted to standard SI meters."""
        return self.range_def.to_si(self.get_display_value())

    def set_display_value(self, val: float, emit_signal: bool = False) -> None:
        """Updates both slider and spinbox without feedback loop."""
        val = max(self.range_def.min_val, min(self.range_def.max_val, val))
        self._block_signals = True

        self.spinbox.setValue(val)
        # Map value to slider integer position [0, _slider_steps]
        frac = (val - self.range_def.min_val) / (self.range_def.max_val - self.range_def.min_val)
        slider_int = int(round(frac * self._slider_steps))
        self.slider.setValue(slider_int)

        self._block_signals = False
        if emit_signal:
            self.value_changed.emit(val)

    def _on_slider_changed(self, int_val: int) -> None:
        if self._block_signals:
            return
        frac = float(int_val) / float(self._slider_steps)
        val = self.range_def.min_val + frac * (self.range_def.max_val - self.range_def.min_val)
        self._block_signals = True
        self.spinbox.setValue(val)
        self._block_signals = False
        self.value_changed.emit(val)

    def _on_spinbox_changed(self, val: float) -> None:
        if self._block_signals:
            return
        frac = (val - self.range_def.min_val) / (self.range_def.max_val - self.range_def.min_val)
        slider_int = int(round(frac * self._slider_steps))
        self._block_signals = True
        self.slider.setValue(slider_int)
        self._block_signals = False
        self.value_changed.emit(val)


class ControlPanel(QWidget):
    """Full-featured scientific control panel for the Double-Slit Laboratory."""

    parameters_changed = pyqtSignal()
    reset_photons_requested = pyqtSignal()
    preset_selected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        """Initializes the control panel widget."""
        super().__init__(parent)

        self._updating_preset: bool = False
        self.current_config = SimulationConfig()

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Constructs scrollable control sections."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Preset Scenarios Group
        layout.addWidget(self._create_presets_group())

        # 2. Optical Source Group
        layout.addWidget(self._create_source_group())

        # 3. Slit Aperture Geometry Group
        layout.addWidget(self._create_aperture_group())

        # 4. Propagation & Screen Geometry Group
        layout.addWidget(self._create_propagation_group())

        # 5. Simulation Mode & Quantum Measurement Group
        layout.addWidget(self._create_quantum_group())

        layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

    # ==========================================================================
    # Group Box Builders
    # ==========================================================================

    def _create_presets_group(self) -> QGroupBox:
        """Builds preset demonstration dropdown."""
        grp = QGroupBox("Experimental Presets", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(10, 14, 10, 10)

        self.combo_presets = QComboBox(self)
        self.combo_presets.setView(QListView())
        self.combo_presets.addItem("Custom Configuration...", None)
        for key, p in PRESETS.items():
            self.combo_presets.addItem(f"{p.name}", key)

        self.combo_presets.currentIndexChanged.connect(self._on_preset_selected)
        layout.addWidget(self.combo_presets)

        self.lbl_preset_desc = QLabel("Adjust parameters freely or choose a preset.", self)
        self.lbl_preset_desc.setWordWrap(True)
        self.lbl_preset_desc.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_preset_desc)

        return grp

    def _create_source_group(self) -> QGroupBox:
        """Builds Laser source controls (Wavelength, Beam Radius)."""
        grp = QGroupBox("Coherent Laser Source", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(6)

        # Wavelength Row with Live Color Swatch
        wl_header = QHBoxLayout()
        wl_label = QLabel("Wavelength (λ):", self)
        self.swatch_wavelength = QLabel(self)
        self.swatch_wavelength.setFixedSize(22, 14)
        self.swatch_wavelength.setStyleSheet(f"border-radius: 3px; background-color: {wavelength_to_hex(532.0)};")
        wl_header.addWidget(wl_label)
        wl_header.addWidget(self.swatch_wavelength)
        wl_header.addStretch()
        layout.addLayout(wl_header)

        self.ctrl_wavelength = DualInputControl(PARAMETER_RANGES["wavelength"], self)
        self.ctrl_wavelength.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_wavelength)

        # Beam Radius
        layout.addWidget(QLabel("Gaussian Beam Waist (w₀):", self))
        self.ctrl_beam_radius = DualInputControl(PARAMETER_RANGES["beam_radius"], self)
        self.ctrl_beam_radius.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_beam_radius)

        return grp

    def _create_aperture_group(self) -> QGroupBox:
        """Builds aperture geometry controls (Type, Width, Separation, Height)."""
        grp = QGroupBox("Slit Aperture Geometry", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(6)

        # Aperture Type Selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Aperture Type:", self))
        self.combo_aperture = QComboBox(self)
        self.combo_aperture.setView(QListView())
        for ap in ApertureType:
            self.combo_aperture.addItem(ap.value, ap)
        self.combo_aperture.currentIndexChanged.connect(self._on_aperture_type_changed)
        type_row.addWidget(self.combo_aperture)
        layout.addLayout(type_row)

        # Slit Width (a)
        layout.addWidget(QLabel("Slit Width (a):", self))
        self.ctrl_slit_width = DualInputControl(PARAMETER_RANGES["slit_width"], self)
        self.ctrl_slit_width.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_slit_width)

        # Slit Separation (d)
        self.lbl_slit_sep = QLabel("Slit Separation (d):", self)
        layout.addWidget(self.lbl_slit_sep)
        self.ctrl_slit_sep = DualInputControl(PARAMETER_RANGES["slit_separation"], self)
        self.ctrl_slit_sep.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_slit_sep)

        # Slit Height (h)
        self.lbl_slit_height = QLabel("Slit Height (h):", self)
        layout.addWidget(self.lbl_slit_height)
        self.ctrl_slit_height = DualInputControl(PARAMETER_RANGES["slit_height"], self)
        self.ctrl_slit_height.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_slit_height)

        # Number of Slits (for Grating)
        self.row_num_slits = QWidget(self)
        ns_layout = QHBoxLayout(self.row_num_slits)
        ns_layout.setContentsMargins(0, 0, 0, 0)
        ns_layout.addWidget(QLabel("Number of Slits (N):", self))
        self.spin_num_slits = QSpinBox(self)
        self.spin_num_slits.setRange(2, 12)
        self.spin_num_slits.setValue(2)
        self.spin_num_slits.setFixedWidth(70)
        self.spin_num_slits.valueChanged.connect(self._on_parameter_changed)
        ns_layout.addWidget(self.spin_num_slits)
        layout.addWidget(self.row_num_slits)
        self.row_num_slits.setVisible(False)

        return grp

    def _create_propagation_group(self) -> QGroupBox:
        """Builds propagation distance and observation screen span controls."""
        grp = QGroupBox("Propagation Geometry", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(6)

        # Distance L
        layout.addWidget(QLabel("Screen Distance (L):", self))
        self.ctrl_distance_l = DualInputControl(PARAMETER_RANGES["distance_l"], self)
        self.ctrl_distance_l.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_distance_l)

        # Screen Span
        layout.addWidget(QLabel("Observation Window Span:", self))
        self.ctrl_screen_span = DualInputControl(PARAMETER_RANGES["screen_span"], self)
        self.ctrl_screen_span.value_changed.connect(self._on_parameter_changed)
        layout.addWidget(self.ctrl_screen_span)

        return grp

    def _create_quantum_group(self) -> QGroupBox:
        """Builds Simulation Mode, Which-Way measurement, and photon rate controls."""
        grp = QGroupBox("Simulation & Quantum State", self)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)

        # Mode Selection Radio Buttons
        mode_row = QHBoxLayout()
        self.radio_wave = QRadioButton("Continuous Wave", self)
        self.radio_quantum = QRadioButton("Quantum Photons", self)
        self.radio_wave.setChecked(True)

        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.addButton(self.radio_wave)
        self.btn_group_mode.addButton(self.radio_quantum)
        self.radio_wave.toggled.connect(self._on_mode_toggled)

        mode_row.addWidget(self.radio_wave)
        mode_row.addWidget(self.radio_quantum)
        layout.addLayout(mode_row)

        # Which-Way Path Detector Toggle
        self.chk_which_way = QCheckBox("Which-Way Path Detector (Decoherence)", self)
        self.chk_which_way.setStyleSheet("font-weight: 600;")
        self.chk_which_way.toggled.connect(self._on_parameter_changed)
        layout.addWidget(self.chk_which_way)

        self.lbl_decoherence_note = QLabel("Path detection collapses quantum state: fringes vanish.", self)
        self.lbl_decoherence_note.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl_decoherence_note)
        self.lbl_decoherence_note.setVisible(False)

        # Quantum Controls Container (Emission Rate, Clear Hits)
        self.quantum_box = QWidget(self)
        q_layout = QVBoxLayout(self.quantum_box)
        q_layout.setContentsMargins(0, 4, 0, 0)
        q_layout.setSpacing(6)

        q_layout.addWidget(QLabel("Photon Emission Rate:", self))
        self.ctrl_photon_rate = DualInputControl(PARAMETER_RANGES["photon_rate"], self)
        self.ctrl_photon_rate.value_changed.connect(self._on_parameter_changed)
        q_layout.addWidget(self.ctrl_photon_rate)

        # Reset / Clear Hits Button
        self.btn_clear_photons = QPushButton("Clear Accumulated Photons", self)
        self.btn_clear_photons.clicked.connect(self.reset_photons_requested.emit)
        q_layout.addWidget(self.btn_clear_photons)

        layout.addWidget(self.quantum_box)
        self.quantum_box.setVisible(False)

        return grp

    # ==========================================================================
    # Event Handlers & State Assembly
    # ==========================================================================

    def _on_parameter_changed(self) -> None:
        """Gathers updated parameters, updates color swatch, and emits signal."""
        if self._updating_preset:
            return

        # Update Live Color Swatch
        wl = self.ctrl_wavelength.get_display_value()
        self.swatch_wavelength.setStyleSheet(f"border-radius: 3px; background-color: {wavelength_to_hex(wl)};")

        # Update Decoherence Notice Visibility
        is_which_way = self.chk_which_way.isChecked()
        self.lbl_decoherence_note.setVisible(is_which_way)

        self.current_config = self.get_configuration()
        self.parameters_changed.emit()

    def _on_aperture_type_changed(self) -> None:
        """Adapts UI control visibility depending on aperture geometry."""
        ap_type = self.combo_aperture.currentData()

        is_multi = (ap_type == ApertureType.MULTI_SLIT)
        is_single = (ap_type == ApertureType.SINGLE_SLIT)
        is_pinhole = (ap_type == ApertureType.CIRCULAR_PINHOLE)

        self.row_num_slits.setVisible(is_multi)
        self.ctrl_slit_sep.setVisible(not is_single and not is_pinhole)
        self.lbl_slit_sep.setVisible(not is_single and not is_pinhole)
        self.ctrl_slit_height.setVisible(not is_pinhole)
        self.lbl_slit_height.setVisible(not is_pinhole)

        self._on_parameter_changed()

    def _on_mode_toggled(self, is_wave: bool) -> None:
        """Shows/hides quantum photon controls based on active mode."""
        self.quantum_box.setVisible(not is_wave)
        self._on_parameter_changed()

    def _on_preset_selected(self, index: int) -> None:
        """Loads selected scenario preset into all controls."""
        preset_key = self.combo_presets.currentData()
        if preset_key is None or preset_key not in PRESETS:
            self.lbl_preset_desc.setText("Custom configuration active.")
            return

        preset = PRESETS[preset_key]
        self._updating_preset = True

        self.ctrl_wavelength.set_display_value(preset.wavelength_nm)
        self.ctrl_slit_width.set_display_value(preset.slit_width_um)
        if preset.slit_separation_um > 0:
            self.ctrl_slit_sep.set_display_value(preset.slit_separation_um)
        self.ctrl_slit_height.set_display_value(preset.slit_height_mm)
        self.ctrl_distance_l.set_display_value(preset.distance_l_m)

        # Set aperture combo
        for i in range(self.combo_aperture.count()):
            if self.combo_aperture.itemData(i) == preset.aperture_type:
                self.combo_aperture.setCurrentIndex(i)
                break

        self.spin_num_slits.setValue(preset.num_slits)
        self.chk_which_way.setChecked(preset.which_way_detected)
        self.lbl_preset_desc.setText(preset.description)

        self._updating_preset = False
        self._on_parameter_changed()

    # ==========================================================================
    # Configuration Packaging
    # ==========================================================================

    def get_configuration(self) -> SimulationConfig:
        """Packages all current GUI inputs into a validated SimulationConfig."""
        mode = SimulationMode.CLASSICAL_WAVE if self.radio_wave.isChecked() else SimulationMode.QUANTUM_MONTE_CARLO
        ap_type = self.combo_aperture.currentData() or ApertureType.DOUBLE_SLIT

        return SimulationConfig(
            wavelength=self.ctrl_wavelength.get_si_value(),
            beam_radius=self.ctrl_beam_radius.get_si_value(),
            aperture_type=ap_type,
            slit_width=self.ctrl_slit_width.get_si_value(),
            slit_separation=self.ctrl_slit_sep.get_si_value(),
            slit_height=self.ctrl_slit_height.get_si_value(),
            num_slits=self.spin_num_slits.value(),
            distance_l=self.ctrl_distance_l.get_si_value(),
            screen_span=self.ctrl_screen_span.get_si_value(),
            mode=mode,
            which_way_detected=self.chk_which_way.isChecked(),
            photon_rate=int(self.ctrl_photon_rate.get_display_value()),
        )


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    from src.config import QT_STYLE_SHEET

    app = QApplication(sys.argv)
    app.setStyleSheet(QT_STYLE_SHEET)
    panel = ControlPanel()
    panel.setWindowTitle("Control Panel (Standalone)")
    panel.resize(360, 750)
    panel.show()
    sys.exit(app.exec())
