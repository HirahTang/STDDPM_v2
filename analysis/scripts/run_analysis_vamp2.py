import mdshare
import sys
from pathlib import Path
import numpy as np

from analysis.dynamics import (
    plot_fes,
    kl_2d,
)
from analysis.topology import get_topology_from_mol
from analysis.dihedrals import get_dihedrals


# =========================
# VAMP utilities
# =========================
def lagged_pairs(X, lag):
    T = X.shape[0]
    if lag <= 0 or lag >= T:
        raise ValueError("Invalid lag")
    return X[:T - lag], X[lag:]


def cov(A, B=None):
    if B is None:
        B = A
    return (A.T @ B) / A.shape[0]


def inv_sqrtm_psd(C, reg=1e-8):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * (1.0 / np.sqrt(w + reg))) @ V.T


class VAMP:
    def __init__(self, lag=10, dim=2, reg=1e-8):
        self.lag = lag
        self.dim = dim
        self.reg = reg

    def fit(self, X):
        X0, Xt = lagged_pairs(X, self.lag)

        self.mean_ = X0.mean(axis=0)
        X0 -= self.mean_
        Xt -= self.mean_

        C00 = cov(X0)
        Ctt = cov(Xt)
        C0t = cov(X0, Xt)

        self.C00_isqrt_ = inv_sqrtm_psd(C00, self.reg)
        self.Ctt_isqrt_ = inv_sqrtm_psd(Ctt, self.reg)

        K = self.C00_isqrt_ @ C0t @ self.Ctt_isqrt_
        self.U_, self.S_, self.Vt_ = np.linalg.svd(K, full_matrices=False)

        return self

    def transform(self, X):
        Xm = X - self.mean_
        return Xm @ self.C00_isqrt_ @ self.U_[:, :self.dim]

    def vamp2_score(self, X, topk=None):
        X0, Xt = lagged_pairs(X, self.lag)

        mean = X0.mean(axis=0)
        X0 -= mean
        Xt -= mean

        C00 = cov(X0)
        Ctt = cov(Xt)
        C0t = cov(X0, Xt)

        C00_isqrt = inv_sqrtm_psd(C00, self.reg)
        Ctt_isqrt = inv_sqrtm_psd(Ctt, self.reg)

        K = C00_isqrt @ C0t @ Ctt_isqrt
        _, S, _ = np.linalg.svd(K, full_matrices=False)

        if topk is None:
            return float(np.sum(S**2)), S
        else:
            return float(np.sum(S[:topk]**2)), S


# =========================
# Main
# =========================
LAG = 10
DIM = 2
REG = 1e-8

# ─── 1. Reference data ─────────────────────────────────────────────
print("Fetching reference data...", flush=True)
ref_npz = mdshare.fetch("alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
with np.load(ref_npz) as fh:
    ref_trajs = [fh[key] for key in sorted(fh.keys())]

ref_features = [np.hstack([np.sin(a), np.cos(a)]) for a in ref_trajs]

# ─── 2. Model trajectory ───────────────────────────────────────────
print("Loading model samples...", flush=True)
sample_path = Path(sys.argv[1])
mol_path = sample_path / "molecule_000.mol"
top = get_topology_from_mol(mol_path)

phi_atoms = [1, 3, 4, 6]
psi_atoms = [3, 4, 6, 8]

pickle_path = sample_path / "conformer.pkl"
if pickle_path.exists():
    phi, psi = get_dihedrals(pickle_path, top, phi_atoms, psi_atoms,
                             save_pdb=False, file_type=None)
else:
    phi, psi = get_dihedrals(sample_path, top, phi_atoms, psi_atoms,
                             save_pdb=False, file_type="mol")

angles = np.hstack([phi.reshape(-1, 1), psi.reshape(-1, 1)])
model_features = np.hstack([np.sin(angles), np.cos(angles)])

# ─── 3. Fit VAMP on reference 1 ────────────────────────────────────
print("Fitting VAMP...", flush=True)
vamp = VAMP(lag=LAG, dim=DIM, reg=REG).fit(ref_features[0])

ref_proj1 = vamp.transform(ref_features[0])
ref_proj2 = vamp.transform(ref_features[1])
ref_proj3 = vamp.transform(ref_features[2])
model_proj = vamp.transform(model_features)

# ─── 4. VAMP-2 scores ──────────────────────────────────────────────
print("Computing VAMP-2 scores...", flush=True)
score_ref1, s_ref1 = vamp.vamp2_score(ref_features[0])
score_ref2, s_ref2 = vamp.vamp2_score(ref_features[1])
score_ref3, s_ref3 = vamp.vamp2_score(ref_features[2])
score_model, s_model = vamp.vamp2_score(model_features)

print(f"VAMP-2 Reference1: {score_ref1:.6f}")
print(f"VAMP-2 Reference2: {score_ref2:.6f}")
print(f"VAMP-2 Reference3: {score_ref3:.6f}")
print(f"VAMP-2 Model:      {score_model:.6f}")

# ─── 5. FES plots ─────────────────────────────────────────────────
print("Plotting FES...", flush=True)
plot_fes(ref_proj1, "Reference MD (traj1, VAMP fit)", method="vamp2")
plot_fes(ref_proj2, "Reference MD (traj2)", method="vamp2")
plot_fes(ref_proj3, "Reference MD (traj3)", method="vamp2")
plot_fes(model_proj, "ML-Generated Trajectory", method="vamp2")

# ─── 6. KL divergence ──────────────────────────────────────────────
print("Computing KL divergence...", flush=True)
kl_score = kl_2d(model_proj, ref_proj1)
print(f"KL divergence (ML vs Reference FES): {kl_score:.4f}")

print("Done!")
