PREDICTION_PATH=./predict_multitask/ext_gen/ext_gen_mt5_best.txt
REFERENCE_PATH=../../data-process/ground_truth_test_filtered_by_length.txt

python rouge.py \
 --dirprediction $PREDICTION_PATH \
 --dirref $REFERENCE_PATH