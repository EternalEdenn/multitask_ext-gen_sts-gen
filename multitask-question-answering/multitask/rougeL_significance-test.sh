PREDICTION1_PATH=../singletask/predict_mt5_wikihow_singletask/generated_predictions.txt
PREDICTION2_PATH=./predict_multitask/ext_gen/ext_gen_mt5_best.txt
REFERENCE_PATH=../../data-process/ground_truth_test_filtered_by_length.txt

python rouge.py \
 --dirprediction1 $PREDICTION1_PATH \
 --dirprediction2 $PREDICTION2_PATH \
 --dirref $REFERENCE_PATH