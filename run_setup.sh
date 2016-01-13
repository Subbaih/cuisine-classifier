n=$1;
k=$2;
python3 helper.py get_corpus data/ data/cuisine.p
python3 helper.py get_n_corpus data/cuisine$n.p $n data/cuisine.p
python3 helper.py get_kfold_corpus data/cuisine$n.p $k data/cuisine$n.p
