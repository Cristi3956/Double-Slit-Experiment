# Double-Slit Experiment & Quantum Wave Optics Laboratory

An interactive, GPU-accelerated simulation of the classic double-slit experiment demonstrating wave-particle duality, 3D Helmholtz optical diffraction, and quantum measurement decoherence.

Built for the **Metode Avansate** (Advanced Methods) course using **Python**, **PyQt6**, **OpenGL**, **pyqtgraph**, and **CuPy (NVIDIA CUDA)**.

---

## Scientific & Physical Capabilities

### 1. 3D Optical Wave Propagation (Angular Spectrum & Fresnel-Fraunhofer)
- Solves the monochromatic 3D Helmholtz equation:
  $$\nabla^2 U(x, y, z) + k^2 U(x, y, z) = 0, \quad k = \frac{2\pi}{\lambda}$$
- Employs a unified Fourier optics framework that scales dynamically from microscopic aperture features ($a \sim 20\,\mu\text{m}$) to macroscopic observation screens ($L \sim 0.05 - 2.5\,\text{m}$) via 2D Fast Fourier Transforms.
- Solves both near-field (Fresnel) and far-field (Fraunhofer) regimes on your **NVIDIA GeForce RTX 3060 Laptop GPU** in **under 3 milliseconds**.

### 2. Quantum Monte Carlo Photon Buildup (Born's Rule)
- Demonstrates how continuous interference emerges from discrete, stochastic quantum detection events:
  $$P(x, y) = \frac{|\psi(x, y, L)|^2}{\iint |\psi(x', y', L)|^2 \, dx' dy'}$$
- Uses GPU-accelerated 2D inverse CDF search with sub-pixel spatial jitter to sample thousands of discrete photon clicks per frame.
- High-performance phosphor accumulation screen retaining up to 150,000 recent hits.

### 3. Quantum Decoherence & "Which-Way" Measurement
- Toggling the path detector simulates quantum state collapse:
  $$|\psi_1 + \psi_2|^2 \longrightarrow |\psi_1|^2 + |\psi_2|^2$$
- Eliminates the interference cross-term, showing the smooth classical sum of two single-slit patterns.
- Color-codes individual photons by slit origin (Cyan for Slit 1, Amber for Slit 2).

### 4. Analytical Verification & Physical Metrics
- Overlays the exact closed-form Fraunhofer theoretical curve:
  $$I(\theta) = I_0 \left[ \frac{\sin(\beta)}{\beta} \right]^2 \cos^2(\alpha)$$
- Computes live figures of merit:
  - **Fringe Spacing**: $\Delta x = \frac{\lambda L}{d}$
  - **Envelope Width**: $2x_0 = \frac{2\lambda L}{a}$
  - **Michelson Visibility**: $V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$
  - **Pearson Correlation**: $R > 99\%$ between GPU numerical solution and theory
  - **Fresnel Number**: $N_F = \frac{a^2}{\lambda L}$

---

## Architecture & Modules

```text
Double Slit Experiment/
├── run.py                     # Main application entry point
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
└── src/
    ├── config.py              # Constants, bounds, presets, and UI theme
    ├── physics/
    │   ├── analytical.py      # Closed-form Fraunhofer formulas and validation metrics
    │   ├── wave_engine.py     # GPU-accelerated wave propagation (CuPy / NumPy fallback)
    │   └── quantum_engine.py  # Quantum Monte Carlo sampling and phosphor accumulator
    ├── rendering/
    │   ├── color_utils.py     # CIE standard observer wavelength (nm) to sRGB conversion
    │   ├── gl_apparatus.py    # 3D interactive OpenGL optical bench (Laser, Slits, Screen)
    │   └── gl_detector.py     # 2D OpenGL screen texture and additive phosphor hits
    └── gui/
        ├── control_panel.py   # Dual slider-spinbox parameter controls and presets
        ├── plot_widget.py     # Real-time pyqtgraph 1D cross-section graph & HUD
        └── main_window.py     # PyQt6 main workstation window
```

---

## How to Run

Ensure your virtual environment or Python environment is active, then run:

```bash
python run.py
```

### Controls & Navigation:
- **3D Optical Bench**:
  - `Left-Click Drag`: Orbit camera around the apparatus.
  - `Right-Click Drag`: Pan camera target.
  - `Scroll Wheel`: Zoom in / out.
  - `Double-Click` or `R`: Reset camera to default overview.
- **2D Detector Screen**:
  - `Left-Click Drag`: Pan screen view.
  - `Scroll Wheel`: Zoom up to $30\times$ into individual microscopic fringes.
  - `Double-Click`: Reset view.
- **1D Intensity Graph**:
  - Move mouse across graph for live coordinate crosshair readout ($x$ in $\text{mm}$ and $I(x)$).
- **Presets**:
  - Use the dropdown in the control panel to load textbook scenarios (Green Laser, He-Ne Red, Single Slit, Grating $N=5$, Fresnel Near-Field, Quantum Decoherence).
- **Data Export**:
  - `Ctrl + S`: Export the 1D cross-section data points to a CSV spreadsheet.
