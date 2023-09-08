import pandas as pd
import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm
import multiprocessing
import shutil

load_dotenv()

# Copying the data to the ../data/generated folder using python
shutil.copy('../data/downloaded/train.csv', '../data/generated/train.csv')
shutil.copy('../data/downloaded/val.csv', '../data/generated/val.csv')
shutil.copy('../data/downloaded/test.csv', '../data/generated/test.csv')

# Initializing the OpenAI API with your key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Creating a global lock for file writing
lock = multiprocessing.Lock()

def get_response(system_prompt, user_prompt):
    """Get the response from the OpenAI API."""
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ])
    
    message = response["choices"][0]["message"]["content"]
    return message

def generate(context):
    system_prompt = "Your task is to generate one synthetic dataset record following given instructions"
    user_prompt = """Given the following legal context, generate Question, Legal Reasoning and Answer. The question should be in such a way that it is asked by a non legal professional to a legal professional. The legal reasoning should help with arriving at the answer and should follow the IRAC method in a single paragraph. The answer should be standalone. 
    Example:
    ```
    Context:
    and Statistical Manual of Mental Disorders 446 (4th ed.1994). 6 Malingering is the intentional production of false or grossly exaggerated physical or psychological symptoms motivated by external incentives, such as obtaining compensation or drugs, avoiding work or military duty, or evading criminal prosecution. American Psychiatric Association, supra, at 683. 7 The record does not reflect whether Dr. Nadel agreed to testify for defendants. If Dr. Nadel had been unwilling to do so, defendants could not have compelled his testimony. See Graham v. Gielchinsky, 126 N.J. 361, 369, 599 A.2d 149 (noting New Jersey in minority of jurisdictions not permitting compulsion of expert testimony); see also Genovese v. N.J. Transit Rail Operations, 234 N.J.Super. 375, 380, 560 A.2d 1272 (App.Div.) (<HOLDING>), certif. denied, 118 N.J. 195, 570 A.2d 960
    
    Question:
    Can the defendants compel Dr. Nadel to testify as an expert witness if he does not agree to do so?
    
    Legal Reasoning:
    The question relates to the compulsion of expert testimony. The court in Graham v. Gielchinsky established that in New Jersey, expert witnesses cannot be compelled to testify against their will. New Jersey is a minority jurisdiction that does not allow compulsion of expert testimony. The court's decision was based on the idea that an expert's knowledge, expertise, and opinions are their own personal property, and compelling them to share this against their will could be seen as a violation of their rights. Therefore, even if Dr. Nadel's testimony could have been crucial in this case involving malingering, if Dr. Nadel was unwilling to testify for the defendants, they could not have forced him to do so. The Genovese v. N.J. Transit Rail Operations case also supports this reasoning.
    
    Answer:
    No, the defendants cannot compel Dr. Nadel to testify as an expert witness if he does not agree to do so in New Jersey, as established in the Graham v. Gielchinsky and Genovese v. N.J. Transit Rail Operations cases.
    ```
    Context:
    {}
    """.format(context)
    return get_response(system_prompt, user_prompt)

def generate_and_save(args):
    index, context, filepath = args
    try:
        new_text = generate(context)
        # Save the result immediately
        with lock:
            df = pd.read_csv(filepath)
            df.at[index, 'GeneratedText'] = new_text
            df.to_csv(filepath, index=False)
        return (index, new_text)
    except Exception as e:
        return (index, None)

def generate_and_save_parallelly(filepath):
    
    try:
        # Loading the CSV into a DataFrame
        df = pd.read_csv(filepath)

        if 'GeneratedText' not in df.columns:
            df['GeneratedText'] = None

        to_generate = [(index, row['Context'], filepath) for index, row in df.iterrows() if pd.isnull(row['GeneratedText']) or row['GeneratedText'] == ""]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            list(tqdm(pool.imap(generate_and_save, to_generate), total=len(to_generate), desc="Generating text"))
                    
        # Checking if all rows have been generated
        df = pd.read_csv(filepath)
        if df['GeneratedText'].isnull().sum() == 0:
            print("All rows have been generated.")
                
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")


if __name__ == "__main__":
    generate_and_save_parallelly("../data/generated/train.csv")
    generate_and_save_parallelly("../data/generated/val.csv")
    generate_and_save_parallelly("../data/generated/test.csv")

