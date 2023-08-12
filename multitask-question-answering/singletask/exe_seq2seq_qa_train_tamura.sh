python run_seq2seq_qa_tamura.py \
  --model_name_or_path google/mt5-base \
  --train_file ../dataset/singletask/train_filtered_by_length.json \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train True\
  --do_eval False\
  --do_predict False\
  --per_device_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 40 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir ./mt5_wikihow_singletask/ \
  --version_2_with_negative False \
  --predict_with_generate 

