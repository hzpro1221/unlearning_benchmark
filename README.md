## Quick Start

First, navigate to the project root: 
```bash
cd unlearning_benchmark
```

### 1. Data Preparation
Download the required datasets by running the respective scripts. These handle automated downloading and local directory organization.
CIFAR-100	-> python dataset/downloader/cifar100.py
OfficeHome -> python dataset/downloader/officehome.py
PACS	-> python dataset/downloader/pacs.py
Tiny ImageNet ->	python dataset/downloader/tiny_imagenet.py

### 2. Training the Base Model
Before unlearning, you must train a "gold" pretrained model. To train a base model (e.g., using the MoE architecture on PACS with a 10% random split reserved for unlearning):
```
python learn.py --config config/learn/pacs_random_10_module_s_16.yaml
```
Key Parameters in Config:
- model_name: module_small_patch16_224
- unlearn_setting: random
- forget_ratio: 0.10

### 3. Unlearning Phase
Once you have a checkpoint, you can apply approximate unlearning algorithms to remove the influence of the "forget set."
To execute unlearning on PACS using the Module Unlearning algorithm:
```
python unlearn.py --config config/random_unlearn/ga_pacs_module_s_16.yaml
```

