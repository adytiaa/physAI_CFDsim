# Physics-Informed Neural Network (PINN) CFD Simulation and Validation — NACA0012 Airfoil

This project demonstrates how to use a **Physics-Informed Neural Network (PINN)** to simulate **compressible, viscous Navier–Stokes flow** over a **NACA0012 airfoil** at **Mach ≈ 1.2**.  

It integrates **NVIDIA PhysicsNeMo** and cüstomaised **PyTorch-based fallback** and an **Validation** that automatically checks mass conservation, boundary conditions, and lift/drag accuracy.

---

## 🚀 Overview

The repo contains workflow for:

1. Generating the **NACA0012 geometry** and simulation grid.
2. Defining and training a **Physics-Based model** to solve the **compressible Navier–Stokes equations**.
3. Using **PhysicsNeMo** (if installed) for PDE residual computation.
4. Running an integrated **Validator Agent** every 100 epochs:
   - Checks **mass conservation**.
   - Enforces **no-penetration boundary condition**.
   - Computes **Lift**, **Drag**, **Cl**, and **Cd**.
   - Produces a **PDF Proof of Simulation Report**.

Each validation step also saves model checkpoints and plots streamlines and Cp contours. This automates the verification steps that are costly or subjective by computing the PDE residuals and check physics consistency, compare AI outputs to trusted solvers or learned surrogates, spot anomalies, outliers, or boundary-condition violations. In the end produce human-readable verdicts and evidence for on-chain contracts.

---

## 🧠 PhysicsNeMo Integration

- If `physicsnemo` is installed, the notebook uses:
  ```python
  from physicsnemo.sym.eq.pdes import navier_stokes as pn_ns
  ns_helper = pn_ns.NavierStokes(viscosity=1e-5, gamma=1.4)
  res = ns_helper.compute_residuals(xy, rho, u, v, p)
  ```
- If not available, it automatically falls back to a PyTorch autodiff-based Navier–Stokes residual evaluator.

---

## 🧩 Runtime Environment

**Recommended:** Google Colab (GPU runtime)

1. Open the notebook in [Google Colab](https://colab.research.google.com).
2. Go to **Runtime → Change runtime type → GPU**.
3. Run all cells top-to-bottom.

Dependencies are installed automatically inside Colab:
```bash
!pip install numpy matplotlib scipy torch
# Optional (if you have NVIDIA PhysicsNeMo access)
!pip install physicsnemo
```

---

## ⚙️ Training Configuration

| Parameter | Description | Default |
|------------|-------------|----------|
| Epochs | Total training iterations | 1000 |
| Validator interval | How often the validator runs | every 100 epochs |
| Grid size | (nx, ny) = (300, 150) | adjustable |
| Network | 4-layer MLP, 256 neurons, Tanh activations | fixed |
| Flow regime | Compressible (Mach ~ 1.2) | steady 2D |
| PDE | Navier–Stokes (energy equation included) | |

---

## 📊 Validation Metrics

The Validator computes and logs the following:

| Metric | Description |
|---------|--------------|
| **mass_div_mean** | Mean absolute mass continuity residual |
| **bc_no_pen_mean** | RMS normal velocity at surface (no-penetration) |
| **bc_no_pen_max** | Maximum violation of BC |
| **Lift**, **Drag** | Integrated surface pressure forces |
| **Cl**, **Cd** | Normalized aerodynamic coefficients |

---

## 📑 Output Files

| File | Description |
|------|--------------|
| `validation_report_epoch_100.pdf` | PDF “Proof of Simulation” report |
| `ckpts/ckpt_epoch_100.pth` | Model checkpoint |
| `validation_report_epoch_200.pdf`, etc. | Reports at subsequent validation steps |

The PDF includes:
- Simulation metrics and metadata (timestamp, environment, torch version)
- Streamline plots over the airfoil
- Pressure coefficient contour (Cp)
- Numerical summary of validation results

---

## 📘 Proof of Simulation Report Example

Contains
- Metadata (date/time, library versions)
- Metrics table (mass residuals, Cl/Cd)
- Plots:
  - Streamlines
  - Cp contour map

---

## 🧪 Extending This Project

You can extend or modify this notebook to:
- Use other NACA airfoils (e.g., NACA 2412 or 4415)
- Add viscous wall models or turbulence closures
- Export ONNX models for real-time surrogate inference
- Add uncertainty quantification (UQ) or autoencoder-based anomaly detection
- Connect to **NVIDIA PhysicsNemo** or **Omniverse**

---

## 🔐 Validation Provenance (Optional AI Proof)

Each PDF is timestamped and includes environment hashes.
Extensions can be:
- Add SHA256 model checkpoints
- Include blockchain-style hash signatures for auditability
- Store validation reports in Google Drive automatically

---

## 🏁 Citation

please cite:
```
NVIDIA PhysicsNeMo: Physics-Informed Neural Models for Simulation and Inference (2024)
https://developer.nvidia.com/physicsnemo
```

---

## 🧭 License
MIT 
This repository is open for research and educational use.  
© 2025
