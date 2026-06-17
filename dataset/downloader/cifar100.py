import os
import argparse

def download_cifar100_hf(data_dir):
    """downloads and organizes the cifar-100 dataset using hugging face."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "this script requires the hugging face `datasets` package.\n"
            "install it with: pip install datasets pillow"
        ) from e

    print(f"Preparing to download CIFAR-100 to: {data_dir}")
    
    # fetch the official cifar100 dataset from hugging face
    print("Fetching dataset from hugging face hub (cifar100)...")
    dataset = load_dataset("uoft-cs/cifar100")
    
    # cifar-100 has two label levels: 'coarse_label' (20 superclasses) and 'fine_label' (100 classes).
    # we will organize the folders by the 100 fine classes.
    fine_label_names = dataset["train"].features["fine_label"].names

    # cifar-100 has official 'train' (50k) and 'test' (10k) splits
    splits = ["train", "test"]

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\nExtracting and saving {split} images to {split_dir}...")
        split_data = dataset[split]
        
        for i, sample in enumerate(split_data):
            class_name = fine_label_names[sample["fine_label"]]
            image = sample["img"]

            # build standard folder structure: split/class_name/image.png
            out_dir = os.path.join(split_dir, class_name)
            os.makedirs(out_dir, exist_ok=True)
            
            # save locally. png is strongly preferred for 32x32 images to avoid artifacting.
            out_file = os.path.join(out_dir, f"{i:05d}.png")
            image.save(out_file, format="PNG")

            # print progress every 5000 images
            if (i + 1) % 5000 == 0:
                print(f"Saved {i + 1} / {len(split_data)} {split} images...")

    print(f"\nSuccess! dataset is beautifully reconstructed at: {data_dir}")

if __name__ == "__main__":
        # target directory is fixed by default
    target_dir = "./dataset/data_folder/cifar100"
    
    # ensure the parent directory exists
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    download_cifar100_hf(target_dir)