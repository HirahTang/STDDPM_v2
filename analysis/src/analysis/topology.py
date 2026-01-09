from pathlib import Path
from rdkit import Chem
import mdtraj as md
import os

def get_topology_from_mol(mol_path):
    rdmol = Chem.MolFromMolFile(mol_path, removeHs=True, sanitize=False)
    pdb_path = f"{Path(mol_path).with_suffix('')}.pdb"
    Chem.MolToPDBFile(rdmol, pdb_path)
    traj_template = md.load_pdb(pdb_path)
    return traj_template.topology


######################## Functions from Han ########################

def read_xyz_coordinates(xyz_path):
    with open(xyz_path, 'r') as f:
        lines = f.readlines()[2:]  # skip first 2 lines
        coords = [list(map(float, line.strip().split()[1:4])) for line in lines]
    return coords

def replace_coordinates(mol, new_coords):
    conf = mol.GetConformer()
    for i, coord in enumerate(new_coords):
        conf.SetAtomPosition(i, coord)
    return mol

def convert_all(mol_dir, xyz_dir, out_dir, output_pdb=False):
    os.makedirs(out_dir, exist_ok=True)
    mol_files = sorted([f for f in os.listdir(mol_dir) if f.endswith('.mol')])
    xyz_files = sorted([f for f in os.listdir(xyz_dir) if f.endswith('.xyz')])
    assert len(mol_files) == len(xyz_files), "Mismatch in number of .mol and .xyz files."
    for mol_file, xyz_file in zip(mol_files, xyz_files):
        mol_path = os.path.join(mol_dir, mol_file)
        xyz_path = os.path.join(xyz_dir, xyz_file)
        mol = Chem.MolFromMolFile(mol_path, removeHs=False)
        if mol is None:
            print(f"Failed to load: {mol_file}")
            continue
        coords = read_xyz_coordinates(xyz_path)
        if len(coords) != mol.GetNumAtoms():
            print(f"Atom count mismatch in {mol_file} and {xyz_file}")
            continue
        mol = replace_coordinates(mol, coords)
        if output_pdb:
            out_path = os.path.join(out_dir, mol_file.replace('.mol', '.pdb'))
            Chem.MolToPDBFile(mol, out_path)
        else:
            out_path = os.path.join(out_dir, mol_file)
            Chem.MolToMolFile(mol, out_path)
        print(f"Written: {out_path}")