# Deep Reconstruction-Classification Networks for Unsupervised Domain Adaptation

This is the code submission (and report) for the Huawei take-home coding examination. 

## Code Structure

    ├── pl_train.py             # Main file
    ├── args.py                 # List of arguments used for this experiment
    ├── pl_model.py             # Model Construction and Training / Test loops   
    ├── data_module.py          # Create and handle dataset, dataloaders and samplers 
    ├── Datasets                # Folder containing datasets wrappers 
    
### Usage
```
python pl_train --source_dataset <ds_name> --target_dataset <ds_Name> --method <base,drcn,drcn-s,drcn-st>
```

## Results 

#### Main Results 
| Model  \ Dataset  | MNIST -> USPS  | USPS -> MNIST | SVHN -> MNIST | MNIST -> SVHN | STL -> CIFAR | CIFAR -> STL | 
|-------------------|----------------|---------------|---------------|---------------|--------------|-------------|
| ConvNet_src (paper) | 85.6         | 65.8          | 62.3          | 26.0          | 54.2         | 63.6        |
| ConvNet_src (reimp) | 93.7         | 96.3          | 75.4          | 21.7          | 55.9         | 66.3        |
| | | | | | | |
| ConvNet_tgt (paper) | 96.1         | 98.7          | 98.7          | 91.5          | 78.8         | 66.5        |
| ConvNet_tgt (reimp) | 96.7         | 99.5          | 99.5          | 93.3          | 55.4         | 80.8        |
| | | | | | | |
| DRCN (paper)        | 91.8         | 73.7          | 82.0          | 40.0          | 58.9         | 66.4        | 
| DRCN (reimp)        | 91.8         | 94.5          | 73.4          | 24.0          | 53.8         | 66.5        |

I used 25, 50, and 100 epochs for the 3 dataset pairs.

1. The <u>reproduced baseline numbers are much higher than the ones in the paper</u>. Given that I performed no hyperparameter tuning whatsoever, a likely explanation is that the authors of the paper did not spend much time into tuning the baselines, but did spend time tuning the proposed method.
2. The SVHN -> MNIST experiments (both in baselines and DRCN) are underperforming. I suspect that is issue are the augmentations and noise used during training. I say this because trying different augmentations (different levels of rotation / translation) and different noise levels seems to have a strong impact on the MNIST SVHN pair. I used the same augmentations across dataset pairs but using custom ones for each pair will boost these numbers. 
3. The other results are fully reproduced



## DRCN flavors
#### Main Results 
| Model  \ Dataset  | MNIST -> USPS  | USPS -> MNIST | SVHN -> MNIST | MNIST -> SVHN | 
|-------------------|----------------|---------------|---------------|---------------|
| DRCN-s            | 91.4           | 94.8          | 65.9 |  24.2           |
| DRCN-st           | 91.8           | 94.3          | 66.6 |  24.5           |
| DRCN              | 91.8           | 94.5          | 73.4 |  24.0           | 

We see a similar trend as in the paper, where DRCN is doing a bit better. Note that a proper analysis would contain more seeds and uncertainty estimates.



## Overall Comments
1. The paper is surprisingly uninformative as to which augmentations to use, and which noise levels for the denoising objective. From my limited experimentation (and prior work in self-supervised visual learning), augmentations have a big impact on downstream peformance, and the aug. parameters (e.g. rotation angle, translation scale, how to infill pixels) is crucial for reproducibility. 
2.  A proper search over hyperparemeters and augmentations to use is required to properly evaluate whether DRCN outperforms baselines. 
3.  On cross-validation. Since we do not have any labelled data, we can only cross validate on classification performance on the source data. I found that this leads to a lot of variance across runs. For example, after 49 and 50 epochs or training, the validation accuracy on source data can be 89.21 and 89.22, but the validation accuracy on the target data is 79.xx and 71.xx. In other words, the correlation between source and target accuracy is not perfect, which leads to significant variance during the cross-validation process. This is why I find the standard deviation reported in the Tables 1 and 2 hard to believe and suspiciously small. 


## Modifications from the original paper 
1. In typical multitask learning setups, performing updates over many batches at once (mix-task batches) almost always performs better. If we treat the source and target as 2 tasks, it makes sense to train on both jointly (rather than one after the other). The current code is implemented as such, but you can revert this behavior with (`--no_mix_src_and_tgt`) this always led to faster convergence. 
2. For the SVHN / MNIST pair, since augmentations and normalization had a big impact on performance, I experimented with a learnable normalization for the target data, which helped performance. Can activate with `--learned_tgt_norm 1`
3. I used Adam rather than RMSprop, from limited tinkering it seems more reliable. 

