import torch


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[0][prop] for mol in batch]) for prop in batch[0][0].keys()}

        to_keep = (batch['charges'].sum(0) > 0)

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch

class PreprocessDialanine:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges
        
    def collate_fn(self, batch):
        batch = {
            "mol1": {prop: batch_stack([mol[0][prop] for mol in batch]) for prop in batch[0][0].keys()},
            "mol2": {prop: batch_stack([mol[1][prop] for mol in batch]) for prop in batch[0][1].keys()},
            "delta_t": batch_stack([mol[2] for mol in batch])
                 }
        
        to_keep_mol1 = (batch['mol1']['charges'].sum(0) > 0)
        to_keep_mol2 = (batch['mol2']['charges'].sum(0) > 0)
        
        batch['mol1'] = {key: drop_zeros(prop, to_keep_mol1) for key, prop in batch['mol1'].items()}
        batch['mol2'] = {key: drop_zeros(prop, to_keep_mol2) for key, prop in batch['mol2'].items()}
        
        atom_mask = batch['mol1']['charges'] > 0
        batch['mol1']['atom_mask'] = atom_mask
        batch['mol2']['atom_mask'] = atom_mask
        
        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        
        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        
        batch['mol1']['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        batch['mol2']['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        
        if self.load_charges:
            batch['mol1']['charges'] = batch['mol1']['charges'].unsqueeze(2)
            batch['mol2']['charges'] = batch['mol2']['charges'].unsqueeze(2)
        else:
            batch['mol1']['charges'] = torch.zeros(0)
            batch['mol2']['charges'] = torch.zeros(0)
        return batch
        