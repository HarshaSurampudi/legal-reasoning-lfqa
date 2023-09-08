# Legal LFQA with reasoning

## 0. Installing dependencies

```bash
pip install -r requirements.txt
```

## 1. Dataset Generation

### 1.1 Downloading the CaseHold dataset

```bash
python dataset_generation/download_casehold.py
```

### 1.2. Getting generations using GPT-3.5-turbo
    
```bash
python dataset_generation/generate.py
```
### 1.3. Extracting elements from the generations

```bash
python dataset_generation/extract_elements.py
```

The final dataset is in `data/extracted` folder.

### 1.4. Uploading the dataset to HuggingFace

```bash
huggingface-cli login
```

```bash
python dataset_generation/upload_to_huggingface.py
```

## 2. Fine-tuning

### 2.1. Preprocessing the dataset

```bash
python finetuning/preprocess_data.py
```

### 2.2. Fine-tuning the models

#### 2.2.1. GPT-J without reasoning

`finetuning/gptj_without_reasoning.ipynb`

#### 2.2.2. GPT-J with reasoning

`finetuning/gptj_with_reasoning.ipynb`

#### 2.2.3. LexGPT without reasoning

`finetuning/lexgpt_without_reasoning.ipynb`

#### 2.2.4. LexGPT with reasoning

`finetuning/lexgpt_with_reasoning.ipynb`

## 3. Evaluation

### 3.1 Generation

#### 3.1.1. GPT-J without reasoning

`evaluation/gptj_without_reasoning.ipynb`

#### 3.1.2 GPT-J with reasoning

`evaluation/gptj_with_reasoning.ipynb`

#### 3.1.3. LexGPT without reasoning

`evaluation/lexgpt_without_reasoning.ipynb`

#### 3.1.4. LexGPT with reasoning

`evaluation/lexgpt_with_reasoning.ipynb`

#### 3.1.5. GPT-J Zero-shot

`evaluation/gptj_zero_shot.ipynb`

#### 3.1.6. GPT-J One-shot

`evaluation/gptj_one_shot.ipynb`

#### 3.1.7. GPT-J Few-shot

`evaluation/gptj_few_shot.ipynb`

The generated outputs are in `evaluation/generated` folder.

### 3.2. Results

```bash
python evaluation/calculate_metrics.py
```