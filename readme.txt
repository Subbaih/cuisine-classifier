Folders needed:
	data
		- contains csv files
		- stores pickle files
			- for complete cuisine set
			- for running kfold validation
	helper.py
		- helper script for manipulating cuisine objects
	cuisine.py
		- Framework for creating cuisine objects

	megam executable file
	main.sh
		- parent script to start with
		- runs run_setup.sh and run_megam_features.sh
	run_setup.sh
		- Converts csv files to python objects and store them as pickle files
		- Samples 'n' recipes and store them as pickle files
		- Creates K files of training and testing pickle files for K-fold cross validation
	run_megam_features.sh 
		- generates the features given as arguments
		- runs MegaM training and prediction
		- fetches MegaM predicted labels
		- creates confusion matrix  
