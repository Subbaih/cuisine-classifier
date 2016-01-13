# STEP: Data Extraction and Cleaning
#python3 helper.py extract_data_1000 "['data_1000/indian','data_1000/italian','data_1000/thai','data_1000/chinese','data_1000/greek']"

rm train*;
rm test*;

# STEP: Feature Engineering - using k-fold cross validation
n=$1;
k=$2; 
features=$3;

# Run K-fold cross validation
for i in `seq 1 $k` 
do
	python3 helper.py get_features "$features" "train"$i".in" "data/cuisine"$n".p.train"$k 0 0;
	python3 helper.py get_features "$features" "test"$i".in" "data/cuisine"$n".p.test"$k 1 0;
	echo 'Megam Learn K-fold:'$i
	./megam_i686.opt multiclass train$i.in > train$i.model
	echo 'Megam Classify K-fold:'$i
	./megam_i686.opt -predict train$i.model multiclass test$i.in  > test$i.out
	python3 helper.py parse_svm_out test$i.out test$i.out
	python3 helper.py run_eval test$i.in.label test$i.out
done
