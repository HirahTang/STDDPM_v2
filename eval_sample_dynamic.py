try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from configs.datasets_config import qm9_with_h, qm9_without_h
from qm9 import dataset
from qm9.models import get_model

from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample
from configs.datasets_config import get_dataset_info


def save_and_sample_dynamic(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, n_samples=10):
    nodesxsample = nodes_dist.sample(n_samples)
    one_hot, charges, x, node_mask = sample(
        args, device, generative_model, dataset_info, eval_args.dynamic_t,
        nodesxsample=nodesxsample)
    # from IPython import embed; embed()
    # vis.visualize_mol(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
    #                   one_hot, x, dataset_info, id_from, name='chain')
    vis.visualize_mol(
        join(eval_args.model_path, f'eval_markovian/dynamic_{eval_args.dynamic_t}/'),
        one_hot, x, dataset_info, id_from=0, name='molecule')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument(
        '--probabilistic_model', type=str, default='dynamic',
        choices=['diffusion', 'dynamic'],
        help='Probabilistic model to use for sampling')
    parser.add_argument(
        '--dynamic_t', type=int, default=500,
        help='Dynamic time for sampling')
    parser.add_argument(
        '--markovian_sampling', action='store_true',
        help='Use markovian sampling')
    

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    args.probabilistic_model = eval_args.probabilistic_model
    args.markovian_sampling = eval_args.markovian_sampling
    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    flow, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'], dtype)
    flow.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                 map_location=device)

    flow.load_state_dict(flow_state_dict)

    print('Sampling handful of molecules.')
    save_and_sample_dynamic(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info, n_samples=1)

if __name__ == "__main__":
    main()