import pandas as pd


df = pd.read_csv('train_extracted.csv')

# Create CSV file with reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nReasoning:\n' + df['Legal Reasoning'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('train_with_reasoning.csv', index=False)

# Create CSV file without reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('train_without_reasoning.csv', index=False)

df = pd.read_csv('val_extracted.csv')

# Create CSV file with reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nReasoning:\n' + df['Legal Reasoning'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('val_with_reasoning.csv', index=False)

# Create CSV file without reasoning
df['text'] = 'Context:\n' + df['Context'] + '\nQuestion:\n' + df['Question'] + '\nAnswer:\n' + df['Answer'] + '\n<|endoftext|>'
df[['text']].to_csv('val_without_reasoning.csv', index=False)

