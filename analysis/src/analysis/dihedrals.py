from glob import glob
from pathlib import Path
import numpy as np
import pickle
import mdtraj as md
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def get_dihedrals(coord_file_path, top, phi_atoms, psi_atoms, save_pdb=False, file_type="txt"):
    coord_file_path = Path(coord_file_path)
    traj = get_trajectory(coord_file_path, top, file_type, superpose=False)

    if save_pdb:
        traj.superpose(traj, 0)
        if coord_file_path.is_dir():
            traj.save_pdb(str(coord_file_path / "ensemble.pdb"))
        elif coord_file_path.suffix == ".pkl":
            traj.save_pdb(str(coord_file_path.parent / "ensemble.pdb"))
        else:
            raise NotImplementedError("Unknow data input type")

    # Compute phi and psi angles
    phi_angles = md.compute_dihedrals(traj, [phi_atoms])
    psi_angles = md.compute_dihedrals(traj, [psi_atoms])

    # Convert radians to degrees for easier interpretation
    return phi_angles.flatten(), psi_angles.flatten()


def get_trajectory(coord_file_path, top, file_type="txt", superpose=True):
    coord_file_path = Path(coord_file_path)
    n_atoms = top.n_atoms

    if coord_file_path.is_dir():
        # Read all coordinate .txt files
        coord_files = sorted(glob(f"{Path(coord_file_path).as_posix()}/*.{file_type}"))
        if len(coord_files) == 0:
            raise FileNotFoundError(f"No files of type {file_type} found")
    
        n_frames = len(coord_files)

        # Preallocate coordinate array
        xyz = np.zeros((n_frames, n_atoms, 3))

        if file_type == "txt":
            for i, fname in enumerate(coord_files):
                with open(fname) as f:
                    lines = f.readlines()
                    lines = [line.strip() for line in lines if line.strip() and not line[0].isdigit()]
                    for j, line in enumerate(lines):
                        parts = line.split()
                        xyz[i, j] = list(map(float, parts[1:]))
        elif file_type == "mol":
            for i, fname in enumerate(coord_files):
                rdmol = Chem.MolFromMolFile(fname, removeHs=True, sanitize=False)

                # Get the first (or only) conformer
                conf = rdmol.GetConformer()

                # Extract coordinates
                coords = []
                for atom in rdmol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())  # returns an RDGeom.Point3D
                    coords.append([pos.x, pos.y, pos.z])
                xyz[i] = np.array(coords)
        else:
            raise NotImplementedError(f"Unknown file type '{file_type}'")
        
    elif coord_file_path.suffix == ".pkl":
        with open(coord_file_path, "rb") as f:
            coord_dict = pickle.load(f)
        if not isinstance(coord_dict, dict):
            raise TypeError("Expected a pickled dict")
        coord_dict = {k: coord_dict[k] for k in sorted(coord_dict)}
        keys = coord_dict.keys()
        n_frames = len(keys)
        if not sorted(keys) == list(range(min(keys), max(keys) + 1)):
            raise ValueError("Keys are not consecutive!")
        xyz = np.stack(list(coord_dict.values()))
    else:
        raise NotImplementedError("Unknow data input type")

    # Convert to nanometers (MDTraj uses nm)
    xyz /= 10.0

    # Create trajectory
    traj = md.Trajectory(xyz=xyz, topology=top)
    if superpose:
        traj.superpose(traj, 0)

    return traj
