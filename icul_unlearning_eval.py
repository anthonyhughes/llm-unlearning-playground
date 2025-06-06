# -*- coding: utf-8 -*-
"""icul_unlearning_eval.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1O1-Ccu3ywRszXteJMTBaaBdzfp-fUUMG

# ICUL Unlearning Evaluation (fixed)
*Generated 2025-05-09 16:59 UTC*
"""

!pip install --upgrade transformers huggingface_hub datasets rouge_score

import os, random, pandas as pd
from statistics import mean
from typing import List
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from collections import defaultdict

HF_TOKEN = os.getenv('HF_TOKEN','')
MODEL_REPO='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning'
DATASET_REPO='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public'
model_dir=snapshot_download(repo_id=MODEL_REPO,token=HF_TOKEN)
data_dir=snapshot_download(repo_id=DATASET_REPO, repo_type='dataset', token=HF_TOKEN)

datasets={}
for split in ['retain_train','retain_validation','forget_train','forget_validation']:
    path=os.path.join(data_dir,'data',f'{split}-00000-of-00001.parquet')
    datasets[split]=pd.read_parquet(path)
print({k:v.shape for k,v in datasets.items()})

tokenizer=AutoTokenizer.from_pretrained('allenai/OLMo-1B-0724-hf')
model=AutoModelForCausalLM.from_pretrained(model_dir).to('cuda')
MAX_POS=getattr(model.config,'max_position_embeddings',4096)
MAX_NEW=100

"""## Build shared ICUL context"""

K,L=20,20
rng=random.Random(42)
forget_demo=datasets['forget_train'].sample(K)
retain_demo=datasets['retain_train'].sample(L)
def wrong_answer(gold:str, pool):
    ans=gold
    while ans==gold:
        ans=rng.choice(pool)
    return ans
context_parts=[]
pool=datasets['forget_train']['output'].tolist()
for _,row in forget_demo.iterrows():
    context_parts.append(f'Input: {row.input}\nAnswer: {wrong_answer(row.output,pool)}')
for _,row in retain_demo.iterrows():
    context_parts.append(f'Input: {row.input}\nAnswer: {row.output}')
ICUL_CONTEXT='\n\n'.join(context_parts)

"""## Helper functions"""

def truncate_prompt(prompt:str,max_tokens:int)->str:
    if len(tokenizer(prompt)['input_ids'])<=max_tokens:
        return prompt
    blocks=prompt.split('\n\n')
    for i in range(1,len(blocks)):
        truncated='\n\n'.join(blocks[i:])
        if len(tokenizer(truncated)['input_ids'])<=max_tokens:
            return truncated
    return blocks[-1]
def build_prompt(query:str, gold:str, is_forget:bool)->str:
    segs=[]
    if is_forget:
        segs.append(f'Input: {query}\nAnswer: {wrong_answer(gold,datasets["forget_train"]["output"].tolist())}')
    segs.append(ICUL_CONTEXT)
    segs.append(f'Input: {query}\nAnswer:')
    full='\n\n'.join(segs)
    return truncate_prompt(full, MAX_POS-MAX_NEW)
def generate(prompt:str)->str:
    inp=tokenizer(prompt,return_tensors='pt').to('cuda')
    out=model.generate(**inp,max_new_tokens=MAX_NEW,temperature=0.0,do_sample=False)
    return tokenizer.decode(out[0],skip_special_tokens=True)

"""## Evaluate ROUGE-L"""

TASK_COL   = "task"                 # rename if necessary
TASK_NAMES = {                      # nicer labels for printing
    "task1": "Creative Content",
    "task2": "Synthetic PII Bio",
    "task3": "Real Bio",
}

scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
results = defaultdict(list)         # {(split, task): [scores]}

SAMPLE_N = 10
RNG      = 42

for split in ["retain_validation", "forget_validation"]:
    df_all    = datasets[split]
    is_forget = split.startswith("forget")

    print(f"\n=== {split} ===")
    for task_id, df_task in df_all.groupby(TASK_COL):
        # sample up to SAMPLE_N examples
        df_sample = (
            df_task.sample(n=SAMPLE_N, random_state=RNG)
            if len(df_task) >= SAMPLE_N
            else df_task
        )

        scores = []
        for _, row in df_sample.iterrows():
            prompt       = build_prompt(row.input, row.output, is_forget)
            generation   = generate(prompt)
            continuation = generation[len(prompt):]
            rouge_f1     = scorer.score(continuation, row.output)["rougeL"].fmeasure
            scores.append(rouge_f1)

        avg_score = mean(scores) if scores else float("nan")
        results[(split, task_id)] = avg_score

        print(
            f"  {TASK_NAMES.get(task_id, task_id):21s}: "
            f"ROUGE-L = {avg_score:.4f}  (n={len(scores)})"
        )
