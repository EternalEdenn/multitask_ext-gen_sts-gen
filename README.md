# MultiTask for Generative&amp;Extractive and Generative&amp;Semantic-Similarity

## There are three tasks included:
- singletask: train with only Generative dataset
- multitask: train with Generative dataset and Extractive dataset
- multitask: train with Generative dataset with inter-sentence Semantic Similarity

## Requirment:
The packages used in this research are placed in the "pack_list.txt".

We do this research by the setting as below:
```
Python: 3.7.8
PyTorch: 1.12.1+cu113 
GPU: A6000
```
You can download PyTorch corresponding to your CUDA version refer to [CUDA-PyTorch](https://pytorch.org/get-started/previous-versions/).
## Run the code
First of all, huggingface-transformers should be downloaded.
```
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```
Secondly, we should download protobuf.
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf/python/
python setup.py install --cpp_implementation
protoc --version    # verify the version
========================================================
# or you can directly use pip install:
pip install protobuf==3.20.0rc1
```
### SingleTask
```
cd multitask-question-answering/singletask
nohup sh exe_seq2seq_qa_train_tamura.sh > singletask_T5.txt &   # train the model
sh exe_seq2seq_qa_predict_tamura.sh                             # do the inference
sh rouge.sh
```
### MultiTask
```
cd multitask-question-answering/singletask
nohup sh exe_seq2seq_qa_train_tamura.sh > singletask_T5.txt &   # train the model
```
