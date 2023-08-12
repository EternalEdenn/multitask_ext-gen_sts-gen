LAMBDA_MT5=0.25
LAMBDA_MNRL=0.75

python run_multi_sts_gen.py \
--train_file_path ./data/train_data_gen_mt5.pt \
--valid_file_path ./data/valid_data_gen_mt5.pt \
--output_dir ./output_mt5_trained_sts_gen-$LAMBDA_MT5-$LAMBDA_MNRL \
--do_train True \
--do_eval True \
--load_best_model_at_end True \
--remove_unused_columns False \
--evaluation_strategy steps \
--num_train_epochs 40 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--lambda_mt5 $LAMBDA_MT5 \
--lambda_mnrl $LAMBDA_MNRL