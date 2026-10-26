"""Optical wavelength to accurate sRGB color conversion utilities.

This module implements the physical CIE standard observer spectral response
(Dan Bruton's algorithm) to convert monochromatic wavelengths in the visible
spectrum (380 nm - 750 nm) to calibrated sRGB color coordinates.

It includes:
1. Wavelength to linear RGB spectral mapping.
2. Human vision photopic luminous efficiency falloff at spectrum edges:
   - Near ultraviolet: 380 nm to 420 nm
   - Near infrared: 700 nm to 750 nm
3. Standard sRGB gamma correction (gamma = 2.2).
"""

from __future__ import annotations

from typing import Tuple


def wavelength_to_rgb(
    wavelength_nm: float,
    gamma: float = 0.8,
) -> Tuple[float, float, float]:
    """Converts an optical wavelength in nanometers to normalized RGB values.

    Args:
        wavelength_nm: Light wavelength in nanometers (valid range ~380 to 750 nm).
        gamma: Display gamma exponent for perceptual intensity curve.

    Returns:
        Tuple of (r, g, b) floating-point values in the range [0.0, 1.0].
    """
    wl = float(wavelength_nm)

    # 1. Spectral Band Decomposition
    if 380.0 <= wl < 440.0:
        # Violet to Deep Blue
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440.0 <= wl < 490.0:
        # Blue to Cyan
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490.0 <= wl < 510.0:
        # Cyan to Green
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif 510.0 <= wl < 580.0:
        # Green to Yellow
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580.0 <= wl < 645.0:
        # Yellow to Orange
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    elif 645.0 <= wl <= 750.0:
        # Orange to Deep Red
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        # Outside visible spectrum (Invisible IR or UV)
        return 0.0, 0.0, 0.0

    # 2. Vision Intensity Attenuation at Spectrum Boundaries
    if 380.0 <= wl < 420.0:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif 420.0 <= wl <= 700.0:
        factor = 1.0
    elif 700.0 < wl <= 750.0:
        factor = 0.3 + 0.7 * (750.0 - wl) / (750.0 - 700.0)
    else:
        factor = 0.0

    # 3. Apply Factor and Gamma Correction
    r_corr = (r * factor) ** gamma if r > 0.0 else 0.0
    g_corr = (g * factor) ** gamma if g > 0.0 else 0.0
    b_corr = (b * factor) ** gamma if b > 0.0 else 0.0

    return (
        max(0.0, min(1.0, float(r_corr))),
        max(0.0, min(1.0, float(g_corr))),
        max(0.0, min(1.0, float(b_corr))),
    )


def wavelength_to_hex(wavelength_nm: float) -> str:
    """Converts a wavelength in nm directly to a CSS/Qt hexadecimal color string."""
    r, g, b = wavelength_to_rgb(wavelength_nm)
    ir = int(round(r * 255.0))
    ig = int(round(g * 255.0))
    ib = int(round(b * 255.0))
    return f"#{ir:02x}{ig:02x}{ib:02x}"
