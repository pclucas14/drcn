import torch
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision import transforms
from torchvision.utils import save_image
from datasets.utils import label_balanced_split
import numpy as np

CLASSES = ['airplane', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']

class CIFAR10(torch.utils.data.Subset):
    """
    Wrapper over the Pytorch SVHN dataset to ensure consistent API across 
    datasets. We are extending `data.Subset` to easily make train / val subsets
    from the full training set.
    """
    def __init__(self, location, split, img_size=32, download=True):
        self.img_size = img_size
        assert split in ["train", "val", "test"]
        transform = self.get_transform(split)
        dataset = PyTorchCIFAR10(
            root=location,
            train=(split != 'test'),
            transform=transform,
            download=download,
        )

        # --- Ensure that the same label gets mapped to the same idx across datasets
        self.target_map = {}
        class_names, targets = np.array(dataset.classes), torch.Tensor(dataset.targets)
        for i, name in enumerate(CLASSES):
            current_idx = np.where(class_names == name)[0][0]
            self.target_map[current_idx] = i

        assert len(self.target_map) == len(CLASSES)

        # --- Restrict the current dataset to only the valid classes
        valid_idx = []
        for valid_label in self.target_map.keys():
            valid_idx += [torch.where(valid_label == targets)[0]]

        valid_idx = torch.cat(valid_idx)

        if split in ['train', 'val']:
            train_idx, val_idx = label_balanced_split(targets)
            idx = {'train': train_idx, 'val': val_idx}[split]
        else:
            idx = torch.arange(len(dataset))

        # lastly, we take the intersection between `idx` and `valid_idx`
        buffer = torch.zeros((len(dataset)))
        buffer.scatter_add_(0, valid_idx, torch.ones(valid_idx.size(0)))
        buffer.scatter_add_(0, idx, torch.ones(idx.size(0)))
        out_idx = torch.where(buffer == 2)[0]
        
        super().__init__(dataset, out_idx)


    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, self.target_map[y]


    # inspired by https://github.com/facebookresearch/suncet/
    def get_transform(self, split):
        if split == "train":
            color_jitter = transforms.ColorJitter(.8, .8, .8, .2)
            base = [
                transforms.RandomResizedCrop(size=self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        else:
            base = [transforms.Resize(size=self.img_size)]

        base += [
            transforms.ToTensor(), 
            transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261])
        ]

        return transforms.Compose(base)


if __name__ == "__main__":
    cifar_tr = CIFAR10("./data", "train", download=True)
    cifar_te = CIFAR10("./data", "test", download=True)

    # sample
    train_samples, test_samples, tr_y, te_y = [], [], [], []
    for i in torch.randperm(len(cifar_tr))[:16]:
        x, y = cifar_tr[i]
        train_samples += [x]; tr_y += [y]
    for i in torch.randperm(len(cifar_te))[:64]:
        x, y = cifar_te[i]
        test_samples += [x]; te_y += [y]

    train_samples = torch.stack(train_samples)
    test_samples = torch.stack(test_samples) 
    assert test_samples.mean().abs() < 0.2 and 0.8 < test_samples.std() < 1.2
    print(tr_y, te_y)
    save_image(torch.cat((train_samples, test_samples)), "./cifar10_train_test.png")
