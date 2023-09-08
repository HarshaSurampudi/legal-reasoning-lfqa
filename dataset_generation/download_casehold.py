import random
from datasets import load_dataset
import pandas as pd

# Loading the casehold/casehold dataset
dataset = load_dataset('casehold/casehold')

# Getting the total number of records in the train split
num_records = len(dataset['train'])

# Randomly sampling 15000 indices
random_indices = random.sample(range(num_records), 15000)

# Selecting the data at those indices
selected_data = [dataset['train'][i] for i in random_indices]

# Extracting the 'citing_prompt' column and convert to pandas DataFrame
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})

# Saving the DataFrame to a CSV file
df.to_csv('../data/downloaded/train.csv', index=False)

# Repeating the process for the validation split
num_records = len(dataset['validation'])
random_indices = random.sample(range(num_records), 1500)

selected_data = [dataset['validation'][i] for i in random_indices]
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})
df.to_csv('../data/downloaded/val.csv', index=False)

# Repeating the process for the test split
num_records = len(dataset['test'])
random_indices = random.sample(range(num_records), 1500)

selected_data = [dataset['test'][i] for i in random_indices]
df = pd.DataFrame({'Context': [item['citing_prompt'] for item in selected_data]})
df.to_csv('../data/downloaded/test.csv', index=False)


