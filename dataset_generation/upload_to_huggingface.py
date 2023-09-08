from datasets import load_dataset
from huggingface_hub import create_repo
from datasets import DatasetDict

# Loading the datasets
train_dataset = load_dataset('csv', data_files='../data/extracted/train.csv', split='train')
test_dataset = load_dataset('csv', data_files='../data/extracted/test.csv', split='train')
val_dataset = load_dataset('csv', data_files='../data/extracted/val.csv', split='train')

# Creating a dictionary of splits for the dataset
dataset_dict = {
    'train': train_dataset,
    'test': test_dataset,
    'validation': val_dataset,
}

repo_name = "legal-reasoning-lfqa-synthetic"

# Make sure you have run huggingface-cli login
# Creating a repo on the Hub
repo_url = create_repo(repo_name, private=True)

# Pushing the dataset to the Hub
dataset = DatasetDict(dataset_dict)
dataset.push_to_hub(repo_name, private=True)




