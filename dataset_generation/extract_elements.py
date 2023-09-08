
import pandas as pd
import re

def keep_only_clean_rows(df):
    """
    Keeping only rows where the context and generated text are both strings and not empty.
    """
    df = df[df['Context'].apply(lambda x: isinstance(x, str))]
    df = df[df['Context'].apply(lambda x: x != '')]
    df = df[df['GeneratedText'].apply(lambda x: isinstance(x, str))]
    df = df[df['GeneratedText'].apply(lambda x: x != '')]
    return df

def process_text_columns(df):
    df = keep_only_clean_rows(df)
    # Defining regular expressions to match the sections
    question_pattern = r"Question:\s*(.*?)\s*Legal Reasoning:"
    reasoning_pattern = r"Legal Reasoning:\s*(.*?)\s*Answer:"
    answer_pattern = r"Answer:\s*(.*)"

    # Initializing empty lists to store the extracted content
    questions = []
    reasonings = []
    answers = []

    # Iterating over each row of the "Generated text" column
    for text in df["GeneratedText"]:
        # Extracting the "Question" content
        question_match = re.search(question_pattern, text, re.DOTALL)
        questions.append(question_match.group(1).strip().replace('\n', ' ') if question_match else None)

        # Extracting the "Legal Reasoning" content
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        reasonings.append(reasoning_match.group(1).strip().replace('\n', ' ') if reasoning_match else None)

        # Extracting the "Answer" content
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        answers.append(answer_match.group(1).strip().replace('\n', ' ') if answer_match else None)

    # Creating new columns with the extracted content
    df["Question"] = questions
    df["Legal Reasoning"] = reasonings
    df["Answer"] = answers

    # Filtering out rows where any of the new columns is None
    df = df.dropna(subset=["Question", "Legal Reasoning", "Answer"])

    df = df.drop(columns=['GeneratedText'])
    return df

val_df = pd.read_csv('../data/generated/val.csv')
val_df = process_text_columns(val_df)
val_df.to_csv('../data/extracted/val.csv', index=False)

test_df = pd.read_csv('../data/generated/test.csv')
test_df = process_text_columns(test_df)
test_df.to_csv('../data/extracted/test.csv', index=False)

train_df = pd.read_csv('../data/generated/train.csv')
train_df = process_text_columns(train_df)
train_df.to_csv('../data/extracted/train_extracted.csv', index=False)




