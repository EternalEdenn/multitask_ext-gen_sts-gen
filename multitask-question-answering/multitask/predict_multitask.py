import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dm', '--dirmodel', required=True)
    parser.add_argument('-dt', '--dirtest', required=True)
    parser.add_argument('-do', '--dirout', required=True)
    
    args = parser.parse_args()
    return args

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained("./mt5_qa_tokenizer/") 

model = AutoModelForSeq2SeqLM.from_pretrained(args.dirmodel)

from predict import pipeline 
#pipelines.py script in the cloned repo
multimodel = pipeline("multitask",tokenizer=tokenizer,model=model)

with open(args.dirtest,'r') as f:
    data = json.load(f)

datalist = data['data']

print(f"Writing predictions...")
with open(args.dirout,'w',encoding='utf-8') as f:
    for idx, data in enumerate(datalist):
        question = data['question']
        context = data['context']
        answer = data['answers']['text']
        prediction = multimodel({"context":f"{context}","question":question})
        f.write(prediction+'\n')
print(f"Finishing prediction")

'''
with open('ground_truth.txt','w',encoding='utf-8') as f:
    for idx, data in enumerate(datalist):
        answer = data['answers']['text']
        f.write(answer)
'''


    