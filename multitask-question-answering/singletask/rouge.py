import evaluate
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dp', '--dirprediction', required=True)
    parser.add_argument('-dr', '--dirref', required=True)
    
    args = parser.parse_args()
    return args

args = parse_args()

rouge = evaluate.load("rouge")
predictions = []
with open(args.dirprediction) as f:
    for line in f:
        predictions.append(line)

references = []
with open(args.dirref) as f:
    for line in f:
        references.append(line)

results = rouge.compute(predictions=predictions,references=references,tokenizer=lambda x: x.split())
print(results)
