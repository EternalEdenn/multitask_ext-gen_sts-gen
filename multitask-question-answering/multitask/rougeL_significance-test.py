import evaluate
import math
import numpy as np
import pandas as pd
import scipy.stats as st
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp1', '--dirprediction1', required=True)
    parser.add_argument('-dp2', '--dirprediction2', required=True)
    parser.add_argument('-dr', '--dirref', required=True)
    
    args = parser.parse_args()
    return args

args = parse_args()


rouge = evaluate.load("rouge")
predictions1 = []
predictions2 = []
with open(args.dirprediction1) as f:
    for line in f:
        predictions1.append(line)

with open(dirprediction2) as f:
    for line in f:
        predictions2.append(line)

references = []
with open(args.dirref) as f:
    for line in f:
        references.append(line)

results1 = rouge.compute(predictions=predictions1,references=references,tokenizer=lambda x: x.split(),use_aggregator=False)
results2 = rouge.compute(predictions=predictions2,references=references,tokenizer=lambda x: x.split(),use_aggregator=False)

RougeL1=results1["rougeL"]
RougeL2=results2["rougeL"]
RougeL1=pd.Series(RougeL1)
RougeL2=pd.Series(RougeL2)

t, p = st.ttest_ind(RougeL1, RougeL2, equal_var=True)
MU = abs(RougeL1.mean()-RougeL2.mean())
SE =  MU/t
DF = len(RougeL1)+len(RougeL2)-2
CI = st.t.interval( alpha=0.95, loc=MU, scale=SE, df=DF )

print('Welch t-test')
print(f'p_value = {p:.3f}')
print(f't_value = {t:.2f}')
print(f'mean difference = {MU:.2f}')
print(f'standard error of difference = {SE:.2f}')
print(f'95% confidence interval CI for mean difference = [{CI[0]:.2f} , {CI[1]:.2f}]')


