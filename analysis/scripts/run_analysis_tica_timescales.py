import mdshare
import sys
from pathlib import Path
import numpy as np
import pyemma
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import get_cmap
from analysis.dynamics import (
    plot_fes,
    compute_its,
    kl_2d,
    plot_its_custom
)
from analysis.topology import get_topology_from_mol
from analysis.dihedrals import get_dihedrals


# ─── 1. Download and load reference dihedral dataset ──────────────────────────
print("Fetching simulated data...")
ref_npz = mdshare.fetch('alanine-dipeptide-3x250ns-backbone-dihedrals.npz')
with np.load(ref_npz) as fh:
    # typically contains keys like 'arr_0','arr_1','arr_2' for three trajectories
    # ref_trajs = [fh[key][-1000:] for key in sorted(fh.keys())]  # shape: (n_frames, 2) each TODO: 1000 steps for debugging
    ref_trajs = [fh[key] for key in sorted(fh.keys())]  # shape: (n_frames, 2) each

# Process each trajectory separately
ref_features = [np.hstack([np.sin(a), np.cos(a)]) for a in ref_trajs]

# ─── 2. Load your model trajectory and compute features ───────────────────────
print("Loading samples...")
sample_path = Path(sys.argv[1])
mol_path = sample_path / "molecule_000.mol"
top = get_topology_from_mol(mol_path)
phi_atoms = [1, 3, 4, 6]
psi_atoms = [3, 4, 6, 8]

pickle_path = sample_path / "conformer.pkl"
if pickle_path.exists():
    phi, psi = get_dihedrals(pickle_path, top, phi_atoms, psi_atoms, save_pdb=False, file_type=None)
else:
    phi, psi = get_dihedrals(sample_path, top, phi_atoms, psi_atoms, save_pdb=False, file_type="mol")
model_angles = np.hstack([phi.reshape(-1, 1), psi.reshape(-1, 1)])
model_features = np.hstack([np.sin(model_angles), np.cos(model_angles)])

# ─── 3. Fit TICA on reference, project both datasets ───────────────────────────
print("Fitting TICA...")
tica = pyemma.coordinates.tica(data=[ref_features[0]], lag=10, dim=2)
ref_proj1 = tica.get_output()[0]
ref_proj2 = tica.transform([ref_features[1]])[0]
ref_proj3 = tica.transform([ref_features[2]])[0]
model_proj = tica.transform([model_features])[0]

# ─── 4. Plot Free Energy Landscapes ──────────────────────────────────────────
print("Plotting...")
plot_fes(ref_proj1, 'Reference MD (traj1, used for TICA fit)')
plot_fes(ref_proj2, 'Reference MD (traj2)')
plot_fes(ref_proj3, 'Reference MD (traj3)')
plot_fes(model_proj, 'ML-Generated Trajectory')

# ─── 5. Compute Implied Timescales for both ───────────────────────────────────
print("Computing timescales...")
its_ref1 = compute_its(ref_proj1, 'Reference1')
its_ref2 = compute_its(ref_proj2, 'Reference2')
its_ref3 = compute_its(ref_proj3, 'Reference3')
its_model = compute_its(model_proj, 'Model')

tab20c = get_cmap('tab20c')
blue_cmap = ListedColormap([tab20c(i) for i in range(0, 3)], name='tab20c_blue')
orange_cmap = ListedColormap([tab20c(i) for i in range(4, 7)], name='tab20c_orange')
green_cmap = ListedColormap([tab20c(i) for i in range(8, 11)], name='tab20c_green')
purple_cmap = ListedColormap([tab20c(i) for i in range(12, 15)], name='tab20c_purple')

plot_its_custom(its_ref1, label="Reference1", cmap=blue_cmap)
plot_its_custom(its_ref2, label="Reference2", cmap=orange_cmap)
plot_its_custom(its_ref3, label="Reference3", cmap=green_cmap)
plot_its_custom(its_model, label="Model", cmap=purple_cmap)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/marl/Code/SpaceTime/images/implied_timescales.png', dpi=300)
plt.close()

# ─── 6. Quantitative comparison via KL divergence ────────────────────────────
print("Calculating KL-divergence...")
kl_score = kl_2d(model_proj, ref_proj1)
print(f"KL divergence (ML vs Reference FES): {kl_score:.4f}")

print("Done!")
