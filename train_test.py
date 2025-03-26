import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
from IPython import embed


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    nll_diff = []
    loss_dynamic = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        # embed()
        mol1 = data['mol1']
        mol2 = data['mol2']
        delta_t = data['delta_t'].to(device, dtype)
        # process mol1
        x1 = mol1['positions'].to(device, dtype)
        node_mask = mol1['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = mol1['edge_mask'].to(device, dtype)
        one_hot1 = mol1['one_hot'].to(device, dtype)
        charges1 = (mol1['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        x1 = remove_mean_with_mask(x1, node_mask)
        
        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps1 = sample_center_gravity_zero_gaussian_with_mask(x1.size(), x1.device, node_mask)
            x1 = x1 + eps1 * args.augment_noise
            
        x1 = remove_mean_with_mask(x1, node_mask)
        if args.data_augmentation:
            x1, rotational_matrix = utils.random_rotation(x1)
            x1 = x1.detach()
            
        check_mask_correct([x1, one_hot1, charges1], node_mask)
        assert_mean_zero_with_mask(x1, node_mask)
        
        # process mol2
        x2 = mol2['positions'].to(device, dtype)
        node_mask2 = mol2['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask2 = mol2['edge_mask'].to(device, dtype)
        one_hot2 = mol2['one_hot'].to(device, dtype)
        charges2 = (mol2['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        x2 = remove_mean_with_mask(x2, node_mask2)
        
        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps2 = sample_center_gravity_zero_gaussian_with_mask(x2.size(), x2.device, node_mask2)
            x2 = x2 + eps2 * args.augment_noise
            
        x2 = remove_mean_with_mask(x2, node_mask2)
        if args.data_augmentation:
            x2 = x2.transpose(1, 2)
            Rx, Ry, Rz = rotational_matrix
            x2 = torch.matmul(Rx, x2)
            x2 = torch.matmul(Ry, x2)
            x2 = torch.matmul(Rz, x2)
            x2 = x2.transpose(1, 2)
            x2 = x2.detach()
            
        check_mask_correct([x2, one_hot2, charges2], node_mask2)
        assert_mean_zero_with_mask(x2, node_mask2)
        
        
        h = {'categorical': one_hot1, 'integer': charges1}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        
        # transform batch through flow
        # embed()
        nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x1, x2, h, node_mask, edge_mask, context, delta_t)
        
        loss_t_step = loss_dict['loss_t'].mean(0)
        loss_s_step = loss_dict['loss_s'].mean(0)
        
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        
        # from IPython import embed; embed()
        nll_epoch.append(nll.item())
        nll_diff.append(loss_t_step.item())
        loss_dynamic.append(loss_s_step.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            if len(args.conditioning) > 0:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=wandb, mode='conditional')
        wandb.log({"Batch NLL": nll.item(),
                   "Batch_NLL_diff": loss_t_step.item(),
                   "Batch_loss_dynamic": loss_s_step.item()}, commit=True)
        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    wandb.log({"Train Epoch NLL_diff": np.mean(nll_diff)}, commit=False)
    wandb.log({"Train Epoch loss_dynamic": np.mean(loss_dynamic)}, commit=True)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)
    vis.visualize_mol(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, x, dataset_info, id_from, name='chain')
    
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=10, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        # from IPython import embed; embed()
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
