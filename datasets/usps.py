import torch
from torchvision.datasets import USPS as PyTorchUSPS
from torchvision import transforms
from torchvision.utils import save_image
from datasets.utils import label_balanced_split


class USPS(torch.utils.data.Subset):
    """
    Wrapper over the Pytorch USPS dataset to ensure consistent API across 
    datasets. We are extending `data.Subset` to easily make train / val subsets
    from the full training set.
    """
    def __init__(self, location, split, img_size=28, download=True):
        self.img_size = img_size
        assert split in ["train", "val", "test"]
        transform = self.get_transform(split)
        dataset = PyTorchUSPS(
            root=location,
            train=(split != "test"),
            transform=transform,
            download=download,
        )

        if split in ['train', 'val']:
            train_idx, val_idx = label_balanced_split(dataset.targets)
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
                    degrees=(-20, 20), translate=(0., 0.1), scale=(0.8, 1.2),
                ),
            ]

        base += [transforms.ToTensor()]

        return transforms.Compose(base)


if __name__ == "__main__":
    usps_tr = USPS("./data", "train", download=True)
    usps_te = USPS("./data", "test", download=True)

    # sample
    train_samples = torch.stack(
        [usps_tr[i][0] for i in torch.randperm(len(usps_tr))[:16]]
    )
    test_samples = torch.stack(
        [usps_te[i][0] for i in torch.randperm(len(usps_te))[:16]]
    )

    assert train_samples.max() <= 1 and train_samples.min() >= 0

    save_image(torch.cat((train_samples, test_samples)), "./usps_train_test.png")
