from rdkit import Chem
import argparse
import os
from tqdm import tqdm
import pickle
def main():
    # Your main code here
    parser = argparse.ArgumentParser()
    parser.add_argument('--working_directory', type=str,
                        default="/home/qcx679/hantang/STDDPM_v2/outputs/STDDPM_nonequi_indexed_1_step_focus_on_smaller_steps_upweight_dynamic_loss/eval_markovian/dynamic_100",
                        help='Molecular path to work with')
    parser.add_argument('--output_directory', type=str,
                        default="/home/qcx679/hantang/STDDPM_v2/outputs/STDDPM_nonequi_indexed_1_step_focus_on_smaller_steps_upweight_dynamic_loss/eval_markovian/dynamic_100_compress",
                        help='Directory to save output files')
    args = parser.parse_args()
    
    # check if output directory exists
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    conformer = {}
    
    for molecule_path in tqdm(os.listdir(args.working_directory)):
        if molecule_path.endswith(".mol"):
            mol = Chem.MolFromMolFile(os.path.join(args.working_directory, molecule_path), sanitize=False)
        else:
            continue
        mol_index = int(molecule_path.split(".")[0].split("_")[-1])
        # get conformation (coordinates only of mol)
        # 
        conformer[mol_index] = mol.GetConformer().GetPositions()
        if mol_index == 0:
            # move the file
            os.rename(os.path.join(args.working_directory, molecule_path),
                      os.path.join(args.output_directory, molecule_path))
    
        # from IPython import embed; embed()
        # save the dictionary
    with open(os.path.join(args.output_directory, "conformer.pkl"), "wb") as f:
        pickle.dump(conformer, f)
    # Do something with the conformer dictionary

if __name__ == "__main__":
    main()