TRAIN_MODEL_PATH=./output_mt5_trained_ext_gen/
TESTDATA_PATH=../dataset/multitask/test_filtered_by_length.json
OUTPUT_PATH=./predict_multitask/ext_gen/ext_gen_mt5_best.txt


python predict_multitask.py \
 --dirmodel $TRAIN_MODEL_PATH \
 --dirtest $TESTDATA_PATH \
 --dirout $OUTPUT_PATH

 sh eval_test.sh $OUTPUT_PATH


