#https://github.com/mjpost/sacrebleu

file=$1
sacrebleu ../../data-process/ground_truth_test_filtered_by_length.txt \
-i $file -l en-ja
