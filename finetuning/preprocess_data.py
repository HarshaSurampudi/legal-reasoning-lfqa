import pandas as pd

df = pd.read_csv('../data/extracted/train.csv')

# Creating CSV file with reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nReasoning:\n' + df['Legal Reasoning'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('../data/preprocessed/train_with_reasoning.csv', index=False)

# Creating CSV file without reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('../data/preprocessed/train_without_reasoning.csv', index=False)

df = pd.read_csv('../data/extracted/val.csv')

# Creating CSV file with reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nReasoning:\n' + df['Legal Reasoning'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('../data/preprocessed/val_with_reasoning.csv', index=False)

# Creating CSV file without reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('../data/preprocessed/val_without_reasoning.csv', index=False)