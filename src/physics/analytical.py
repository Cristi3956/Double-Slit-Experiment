"""Theoretical and analytical optics module for diffraction and interference.

This module provides closed-form Fraunhofer solutions for single slits,
double slits, multi-slit gratings, circular apertures (Airy disks), and
quantum decoherence states (which-way path detection).

These analytical formulas serve as the ground truth benchmark against which
the GPU-based numerical wave propagation (Angular Spectrum Method) is validated.

Mathematical Formulation:
-------------------------
1. Single-Slit Diffraction Envelope:
   beta = (pi * a / lambda) * sin(theta)
   I_diff(theta) = sinc^2(beta / pi) = (sin(beta) / beta)^2

2. Double-Slit Interference Factor (coherent superposition):
   alpha = (pi * d / lambda) * sin(theta)
   I_int(theta) = cos^2(alpha)

3. Combined Coherent Double-Slit Profile:
   I_double(theta) = I_0 * [sin(beta) / beta]^2 * cos^2(alpha)

4. Multi-Slit Grating (N slits):
   I_multi(theta) = I_0 * [sin(beta) / beta]^2 * [sin(N * alpha) / (N * sin(alpha))]^2

5. Quantum Decoherence (Which-Way path detection active):
   When which-way detection is performed at the slits, phase coherence between
   the two paths is completely destroyed. The cross-interference term vanishes:
   |psi_1 + psi_2|^2  -->  |psi_1|^2 + |psi_2|^2
   Resulting in the classical addition of two shifted single-slit envelopes:
   I_decoherent(x) = 0.5 * [I_single(x - d/2) + I_single(x + d/2)]

6. Circular Aperture (Airy Disk):
   gamma = (pi * D / lambda) * sin(theta)
   I_airy(theta) = I_0 * [2 * J_1(gamma) / gamma]^2
   where J_1 is the Bessel function of the first kind of order 1.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path when executed directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import scipy.special as sp

from src.config import ApertureType, SimulationConfig


# ==============================================================================
# 1. Analytical Metrics Container
# ==============================================================================

@dataclass(frozen=True)
class AnalyticalMetrics:
    """Calculated theoretical parameters and physical figures of merit.

    Attributes:
        fringe_spacing_m: Distance between adjacent interference maxima (m).
        envelope_width_m: Full width of the central diffraction envelope between
            its first two zeros (m).
        fringes_in_central_maximum: Number of interference fringes that fit
            within the central single-slit envelope: 2 * (d / a).
        fringe_visibility: Michelson visibility contrast V = (I_max - I_min) / (I_max + I_min).
            Equals 1.0 for pure coherent double slit; drops to 0.0 under full decoherence.
        fresnel_number: Dimensionless parameter N_F = a^2 / (lambda * L).
            If N_F << 1: Far-field Fraunhofer regime holds strictly.
            If N_F >= 1: Near-field Fresnel regime applies.
        regime_label: Text classification of the propagation regime.
        angular_spread_rad: Half-angle to first diffraction minimum: lambda / a (rad).
    """

    fringe_spacing_m: float
    envelope_width_m: float
    fringes_in_central_maximum: float
    fringe_visibility: float
    fresnel_number: float
    regime_label: str
    angular_spread_rad: float


# ==============================================================================
# 2. Metrics Calculation Function
# ==============================================================================

def calculate_analytical_metrics(config: SimulationConfig) -> AnalyticalMetrics:
    """Computes theoretical properties and validation metrics from configuration.

    Args:
        config: SimulationConfig containing wavelength, geometry, and slit dimensions.

    Returns:
        AnalyticalMetrics instance with theoretical values in SI units.
    """
    wavelength: float = config.wavelength
    slit_width: float = config.slit_width
    slit_sep: float = config.slit_separation
    distance_l: float = config.distance_l

    # 1. Fringe Spacing: Delta_x = lambda * L / d
    if slit_sep > 0.0 and config.aperture_type in (ApertureType.DOUBLE_SLIT, ApertureType.MULTI_SLIT):
        fringe_spacing = (wavelength * distance_l) / slit_sep
    else:
        fringe_spacing = 0.0

    # 2. Central Envelope Width: 2 * lambda * L / a
    if slit_width > 0.0:
        envelope_width = 2.0 * (wavelength * distance_l) / slit_width
        angular_spread = wavelength / slit_width
    else:
        envelope_width = 0.0
        angular_spread = 0.0

    # 3. Number of fringes in central envelope: 2 * (d / a)
    if slit_width > 0.0 and slit_sep > 0.0:
        fringes_in_center = 2.0 * (slit_sep / slit_width)
    else:
        fringes_in_center = 1.0

    # 4. Michelson Visibility Contrast: V = (I_max - I_min) / (I_max + I_min)
    if config.which_way_detected:
        # Decoherence eliminates interference oscillations, visibility -> 0
        fringe_visibility = 0.0
    elif config.aperture_type == ApertureType.DOUBLE_SLIT:
        fringe_visibility = 1.0
    else:
        fringe_visibility = 1.0

    # 5. Fresnel Number: N_F = a^2 / (lambda * L)
    characteristic_size = slit_sep if slit_sep > 0.0 else slit_width
    fresnel_number = (characteristic_size ** 2) / (wavelength * distance_l)

    if fresnel_number < 0.1:
        regime_label = "Fraunhofer (Far-Field, NF << 1)"
    elif fresnel_number < 1.0:
        regime_label = "Intermediate Fresnel/Fraunhofer"
    else:
        regime_label = "Fresnel (Near-Field, NF >= 1)"

    return AnalyticalMetrics(
        fringe_spacing_m=fringe_spacing,
        envelope_width_m=envelope_width,
        fringes_in_central_maximum=fringes_in_center,
        fringe_visibility=fringe_visibility,
        fresnel_number=fresnel_number,
        regime_label=regime_label,
        angular_spread_rad=angular_spread,
    )


# ==============================================================================
# 3. 1D Cross-Section Theoretical Intensity Profile
# ==============================================================================

def compute_theoretical_1d_slice(
    config: SimulationConfig,
    x_coordinates: np.ndarray,
) -> np.ndarray:
    """Calculates the theoretical 1D intensity cross-section I(x) on the screen.

    Uses paraxial/exact trigonometric projections to avoid small-angle drift:
        sin(theta) = x / sqrt(x^2 + L^2)

    Args:
        config: SimulationConfig specifying wavelength, aperture, and distances.
        x_coordinates: 1D NumPy array of horizontal screen positions in meters.

    Returns:
        1D NumPy array of normalized theoretical intensity values in range [0, 1].
    """
    wavelength: float = config.wavelength
    slit_width: float = config.slit_width
    slit_sep: float = config.slit_separation
    distance_l: float = config.distance_l
    aperture_type: ApertureType = config.aperture_type
    which_way: bool = config.which_way_detected

    # Exact sine of observation angle: sin(theta) = x / sqrt(x^2 + L^2)
    sin_theta = x_coordinates / np.sqrt(x_coordinates ** 2 + distance_l ** 2)

    # -------------------------------------------------------------------------
    # Case 1: Quantum Decoherence (Which-Way Detector Active)
    # -------------------------------------------------------------------------
    if which_way and aperture_type == ApertureType.DOUBLE_SLIT:
        # Incoherent sum of two individual slits displaced by +- d/2:
        # I(x) = 0.5 * [ sinc^2(a*(x - d/2)/(lambda*L)) + sinc^2(a*(x + d/2)/(lambda*L)) ]
        shift_right = x_coordinates - (0.5 * slit_sep)
        shift_left = x_coordinates + (0.5 * slit_sep)

        sin_theta_right = shift_right / np.sqrt(shift_right ** 2 + distance_l ** 2)
        sin_theta_left = shift_left / np.sqrt(shift_left ** 2 + distance_l ** 2)

        # Note: np.sinc(u) computes sin(pi * u) / (pi * u) in NumPy convention
        beta_right = (slit_width / wavelength) * sin_theta_right
        beta_left = (slit_width / wavelength) * sin_theta_left

        i_slit1 = (np.sinc(beta_right)) ** 2
        i_slit2 = (np.sinc(beta_left)) ** 2

        intensity = 0.5 * (i_slit1 + i_slit2)
        return intensity / (np.max(intensity) + 1e-15)

    # -------------------------------------------------------------------------
    # Case 2: Circular Aperture (Airy Disk)
    # -------------------------------------------------------------------------
    if aperture_type == ApertureType.CIRCULAR_PINHOLE:
        # gamma = (pi * a / lambda) * sin(theta)
        # I(gamma) = [ 2 * J_1(gamma) / gamma ]^2
        gamma = (np.pi * slit_width / wavelength) * sin_theta

        # Avoid zero division at central peak gamma = 0
        intensity = np.ones_like(gamma, dtype=np.float64)
        non_zero = np.abs(gamma) > 1e-12

        g = gamma[non_zero]
        intensity[non_zero] = (2.0 * sp.j1(g) / g) ** 2
        return intensity

    # -------------------------------------------------------------------------
    # Case 3: Rectangular Slits (Single, Double, or Multi-Slit)
    # -------------------------------------------------------------------------
    # Diffraction envelope factor: beta = (pi * a / lambda) * sin(theta)
    # np.sinc expects argument u such that sin(pi * u) / (pi * u)
    u_diffraction = (slit_width / wavelength) * sin_theta
    diffraction_factor = (np.sinc(u_diffraction)) ** 2

    if aperture_type == ApertureType.SINGLE_SLIT or config.num_slits <= 1:
        # Pure single slit: only diffraction envelope
        intensity = diffraction_factor

    elif aperture_type == ApertureType.DOUBLE_SLIT or config.num_slits == 2:
        # Standard Young's Double Slit: cos^2(alpha)
        alpha = (np.pi * slit_sep / wavelength) * sin_theta
        interference_factor = (np.cos(alpha)) ** 2
        intensity = diffraction_factor * interference_factor

    elif aperture_type == ApertureType.MULTI_SLIT:
        # N-slit diffraction grating: [ sin(N * alpha) / (N * sin(alpha)) ]^2
        n_slits = config.num_slits
        alpha = (np.pi * slit_sep / wavelength) * sin_theta

        interference_factor = np.ones_like(alpha, dtype=np.float64)
        non_zero_denom = np.abs(np.sin(alpha)) > 1e-10

        a_nz = alpha[non_zero_denom]
        interference_factor[non_zero_denom] = (
            np.sin(n_slits * a_nz) / (n_slits * np.sin(a_nz))
        ) ** 2

        intensity = diffraction_factor * interference_factor
    else:
        intensity = diffraction_factor

    # Normalize peak intensity to unity for uniform plotting
    max_val = np.max(intensity)
    if max_val > 0.0:
        intensity = intensity / max_val

    return intensity


# ==============================================================================
# 4. 2D Theoretical Intensity Field Generator
# ==============================================================================

def compute_theoretical_2d_field(
    config: SimulationConfig,
    grid_size: int,
    physical_span: float,
) -> np.ndarray:
    """Generates a full 2D theoretical irradiance map I(x, y) on the screen.

    For rectangular slits of width 'a' (along x) and height 'h' (along y):
        I(x, y) = I_x(x) * sinc^2((h / lambda) * sin(theta_y))

    Args:
        config: SimulationConfig instance.
        grid_size: Resolution of the square output matrix (e.g. 512, 1024).
        physical_span: Total physical width/height of the observation plane (m).

    Returns:
        2D NumPy array of shape (grid_size, grid_size) normalized to [0, 1].
    """
    coords = np.linspace(-physical_span / 2.0, physical_span / 2.0, grid_size, dtype=np.float64)

    # 1. Horizontal profile I_x(x)
    i_x = compute_theoretical_1d_slice(config, coords)

    if config.aperture_type == ApertureType.CIRCULAR_PINHOLE:
        # Radial symmetry for circular aperture: r = sqrt(x^2 + y^2)
        xx, yy = np.meshgrid(coords, coords)
        r = np.sqrt(xx ** 2 + yy ** 2)
        sin_theta = r / np.sqrt(r ** 2 + config.distance_l ** 2)

        gamma = (np.pi * config.slit_width / config.wavelength) * sin_theta
        field_2d = np.ones_like(gamma, dtype=np.float64)
        non_zero = np.abs(gamma) > 1e-12

        g = gamma[non_zero]
        field_2d[non_zero] = (2.0 * sp.j1(g) / g) ** 2
        return field_2d

    # 2. Vertical profile I_y(y) for slit height h
    sin_theta_y = coords / np.sqrt(coords ** 2 + config.distance_l ** 2)
    u_y = (config.slit_height / config.wavelength) * sin_theta_y
    i_y = (np.sinc(u_y)) ** 2

    # Outer product for separable rectangular aperture diffraction
    field_2d = np.outer(i_y, i_x)

    max_val = np.max(field_2d)
    if max_val > 0.0:
        field_2d /= max_val

    return field_2d


# ==============================================================================
# 5. Numerical vs. Analytical Comparison and Error Analysis
# ==============================================================================

def compare_simulation_to_theory(
    simulated_slice: np.ndarray,
    theoretical_slice: np.ndarray,
) -> Dict[str, float]:
    """Quantifies agreement between numerical GPU simulation and theoretical curve.

    Args:
        simulated_slice: 1D normalized intensity array from numerical wave solver.
        theoretical_slice: 1D normalized intensity array from analytical formula.

    Returns:
        Dictionary containing statistical error metrics:
            - "mse": Mean Squared Error
            - "rmse": Root Mean Squared Error
            - "max_error": Maximum Absolute Error
            - "correlation": Pearson correlation coefficient (1.0 = perfect match)
    """
    # Ensure arrays match in shape
    if simulated_slice.shape != theoretical_slice.shape:
        raise ValueError(
            f"Shape mismatch: simulated {simulated_slice.shape} vs "
            f"theoretical {theoretical_slice.shape}"
        )

    # Normalize both to peak unity for fair residual comparison
    sim_norm = simulated_slice / (np.max(simulated_slice) + 1e-15)
    theo_norm = theoretical_slice / (np.max(theoretical_slice) + 1e-15)

    error = sim_norm - theo_norm
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(np.abs(error)))

    # Pearson correlation coefficient
    std_sim = np.std(sim_norm)
    std_theo = np.std(theo_norm)

    if std_sim > 1e-12 and std_theo > 1e-12:
        correlation = float(
            np.mean((sim_norm - np.mean(sim_norm)) * (theo_norm - np.mean(theo_norm)))
            / (std_sim * std_theo)
        )
    else:
        correlation = 1.0

    return {
        "mse": mse,
        "rmse": rmse,
        "max_error": max_err,
        "correlation": correlation,
    }
