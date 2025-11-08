"""
pinn_airfoil.py
A simple illustrative PINN using NVIDIA PhysicsNeMo (physicsnemo-sym) + PyTorch
Goal: solve steady incompressible Navier-Stokes around a 2D airfoil
"""

import torch
import torch.nn as nn
import numpy as np

# PhysicsNeMo imports (sym module contains PINN utilities & PDE helpers)
from physicsnemo.sym import geometry as geom
from physicsnemo.sym import pinn as pn  # high-level PINN helpers (hypothetical API names)
from physicsnemo.sym.pdes import NavierStokes2D  # PDE helper (example)
# NOTE: actual module/class names may differ slightly — see docs/examples for exact names. :contentReference[oaicite:1]{index=1}

# -------------------------
# 1) Problem parameters
# -------------------------
nu = 1.5e-5            # kinematic viscosity (m^2/s) — set for your Re
U_inf = 30.0           # freestream speed (m/s)
rho = 1.225            # density (kg/m^3)
domain_bounds = [-1.5, 3.0, -1.5, 1.5]  # xmin, xmax, ymin, ymax — covers airfoil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 2) Geometry and BCs
# -------------------------
# Create geometry object and airfoil obstacle (simple cambered shape or NACA profile)
# Here we create a placeholder airfoil via parametric function or load coordinates
airfoil_coords = np.loadtxt("naca0012_coords.txt")  # (x,y) points around the airfoil surface

domain = geom.Domain2D(xmin=domain_bounds[0], xmax=domain_bounds[1],
                       ymin=domain_bounds[2], ymax=domain_bounds[3])

# add airfoil as internal boundary
domain.add_internal_boundary("airfoil", points=airfoil_coords)

# Boundary conditions:
# - Inlet (left): u = U_inf, v = 0
# - Outlet (right): zero pressure gradient (or p=0 reference)
# - Top/Bottom: far-field (u=U_inf, v=0) or slip
# - Airfoil surface: no-slip u=v=0
bc = [
    ("inlet", {"u": U_inf, "v": 0.0}),
    ("top_bottom", {"u": U_inf, "v": 0.0}),
    ("airfoil", {"u": 0.0, "v": 0.0}),
    ("outlet", {"p": 0.0})  # reference pressure
]

# -------------------------
# 3) PINN neural network
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, width=128, depth=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, out_dim))  # outputs: u, v, p
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)

# -------------------------
# 4) PDE & loss terms
# -------------------------
# construct Navier-Stokes residuals using automatic differentiation
# PhysicsNeMo-sym provides helpers to compute PDE residuals; below is illustrative
ns = NavierStokes2D(viscosity=nu, density=rho)

def pinn_loss(batch_xy):
    """
    batch_xy: tensor (N,2) of x,y collocation points inside domain (not on boundary)
    compute PDE residuals + BC losses
    """
    batch_xy = batch_xy.detach().requires_grad_(True).to(device)
    uvp = model(batch_xy)            # (N,3): u, v, p
    u = uvp[:,0:1]
    v = uvp[:,1:2]
    p = uvp[:,2:3]

    # PDE residuals (continuity + momentum)
    continuity_res, mom_x_res, mom_y_res = ns.compute_residuals(batch_xy, u, v, p)

    # MSE of residuals
    res_loss = (continuity_res.pow(2).mean()
                + mom_x_res.pow(2).mean()
                + mom_y_res.pow(2).mean())

    # boundary loss terms
    # sample points on BCs and enforce BCs
    bc_loss = 0.0
    for bc_name, bc_dict in bc:
        bc_pts = domain.sample_boundary(bc_name, n=512)  # (Nbc,2)
        bc_pts = torch.tensor(bc_pts, dtype=torch.float32, device=device)
        pred = model(bc_pts)
        # compute BC mismatch
        if "u" in bc_dict and "v" in bc_dict:
            u_t = pred[:,0:1]; v_t = pred[:,1:2]
            bc_loss += ((u_t - bc_dict["u"])**2).mean() + ((v_t - bc_dict["v"])**2).mean()
        if "p" in bc_dict:
            p_t = pred[:,2:3]
            bc_loss += ((p_t - bc_dict["p"])**2).mean()

    total_loss = res_loss + 100.0 * bc_loss  # weighting BC heavier (tune)
    return total_loss, res_loss.item(), bc_loss.item()

# -------------------------
# 5) Training loop
# -------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

n_iters = 15000
for it in range(1, n_iters+1):
    # sample collocation points inside domain (Latin Hypercube / uniform)
    xy_interior = domain.sample_interior(n=4096)
    xy_interior = torch.tensor(xy_interior, dtype=torch.float32)

    optimizer.zero_grad()
    loss, res_l, bc_l = pinn_loss(xy_interior)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if it % 250 == 0:
        print(f"[{it:06d}] total_loss={loss.item():.3e} res={res_l:.3e} bc={bc_l:.3e}")

# -------------------------
# 6) Save / postprocess
# -------------------------
torch.save(model.state_dict(), "pinn_airfoil.pth")
# sample grid for visualization
xs = np.linspace(domain_bounds[0], domain_bounds[1], 400)
ys = np.linspace(domain_bounds[2], domain_bounds[3], 200)
grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
grid_t = torch.tensor(grid, dtype=torch.float32, device=device)
with torch.no_grad():
    pred = model(grid_t).cpu().numpy()
u = pred[:,0].reshape(len(ys), len(xs))
v = pred[:,1].reshape(len(ys), len(xs))
p = pred[:,2].reshape(len(ys), len(xs))

# Save simple CSV or VTK for Paraview; PhysicsNeMo examples show how to export .vtp/.vtk. :contentReference[oaicite:2]{index=2}
np.savetxt("airfoil_u.csv", u, delimiter=",")
np.savetxt("airfoil_v.csv", v, delimiter=",")
np.savetxt("airfoil_p.csv", p, delimiter=",")
print("Done. Saved model and field CSVs.")

