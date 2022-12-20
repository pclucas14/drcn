import torch
from torchvision.datasets import SVHN as PyTorchSVHN
from torchvision import transforms
from torchvision.utils import save_image
from datasets.utils import label_balanced_split


class SVHN(torch.utils.data.Subset):
    """
    Wrapper over the Pytorch SVHN dataset to ensure consistent API across 
    datasets. We are extending `data.Subset` to easily make train / val subsets
    from the full training set.
    """
    def __init__(self, location, split, img_size=32, download=True):
        self.img_size = img_size
        assert split in ["train", "val", "test"]
        transform = self.get_transform(split)
        dataset = PyTorchSVHN(
            root=location,
            split={'train': 'train', 'val': 'train', 'test': 'test'}[split],
            transform=transform,
            download=download,
        )

        if split in ['train', 'val']:
            train_idx, val_idx = label_balanced_split(dataset.labels)
            idx = {'train': train_idx, 'val': val_idx}[split]
        else:
            idx = torch.arange(len(dataset))

        super().__init__(dataset, idx)


    def get_transform(self, split):
        base = [
            transforms.Resize(self.img_size),
        ]

        if split == "train":
            base += [
                transforms.RandomAffine(
                    degrees=(-15, 15), translate=(0.05, 0.1), scale=(0.8, 1.2)
                ),
            ]

        base += [transforms.ToTensor()]

        return transforms.Compose(base)


if __name__ == "__main__":
    svhn_tr = SVHN("./data", "train", download=True)
    svhn_te = SVHN("./data", "test", download=True)

    # sample
    train_samples = torch.stack(
        [svhn_tr[i][0] for i in torch.randperm(len(svhn_tr))[:16]]
    )
    test_samples = torch.stack(
        [svhn_te[i][0] for i in torch.randperm(len(svhn_te))[:16]]
    )

    assert train_samples.max() <= 1 and train_samples.min() >= 0

    save_image(torch.cat((train_samples, test_samples)), "./svhn_train_test.png")
