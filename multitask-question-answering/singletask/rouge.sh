PREDICTION_PATH=./predict_mt5_wikihow_singletask/generated_predictions.txt
REFERENCE_PATH=../../data-process/ground_truth_test_filtered_by_length.txt

python rouge.py \
 --dirprediction $PREDICTION_PATH \
 --dirref $REFERENCE_PATH