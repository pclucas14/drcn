import torch 

def label_balanced_split(labels, split=0.8):
    """
    Randomly split dataset in a reproducible fashion, ensuring label 
    proportions are maintained
    """
    train_idx, val_idx = [], []
    gen = torch.Generator()
    gen.manual_seed(0)

    if not isinstance(labels, torch.Tensor):
        labels = torch.Tensor(labels).long()
    for label in labels.unique():
        label_idx = torch.where(label == labels)[0]
        n_tot = label_idx.size(0)
        n_train = int(n_tot * split)
        shuffled_idx = label_idx[torch.randperm(n_tot, generator=gen)]
        train_idx += [shuffled_idx[:n_train]]
        val_idx   += [shuffled_idx[n_train:]]

    tr_split = torch.cat(train_idx)
    val_split = torch.cat(val_idx)

    return tr_split, val_split