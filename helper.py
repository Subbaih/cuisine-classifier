#!/usr/bin/env python3

"""Multi-Cuisine detector Code
Helper functions for Cuisine corpus:
"""
import nltk;
import sys;
import os;
import re;
import json;
import csv;
import math;
from cuisine import *;
import pickle;

__author__ = "Subbiah Ganesan"
__email__ = "subbiahg@usc.edu"
dpath = 'data/';
pickled_corpus = dpath + 'cuisine';
tagger = None;

def general_decoder(s):
	try:
		es = [i for i in nltk.sent_tokenize(s)];
		#print(es);
	except Exception as E:
		raise Exception('general-decoder: not able to decode %s.Exception-%s'%(s,E));
	return es;

def ingredients_decoder(s):
	try:
		token_plist = set(nltk.corpus.stopwords.words('english'));
		pos_oklist = ['NN','NNS','NNP','NNPS']; # Only nouns
		words = nltk.word_tokenize(s);
		ptags = tagger.tag(words);
		cwords = [p[0] for p in ptags if p[1] in pos_oklist and p[0] not in token_plist];
		es = cwords;
	except Exception as E:
		#print(E);
		raise Exception('ingredients-decoder: not able to decode %s.Exception-%s'%(s,E));
	return es;

def cuisine_json_decoder(data):
	try:
		ndata = {'ingredients':[],'method':[]};
		ndata['title'] = general_decoder(data['title']);
		for i in data['ingredients']:
			ndata['ingredients'].extend(ingredients_decoder(i));
		for i in data['method']:
			ndata['method'].extend(general_decoder(i));
	except Exception as E:
		#print(E);
		print('');
	else:
		with open(extract_data_path+'.csv','a') as fd:
			w = csv.writer(fd);
			w.writerow([ndata['title'],ndata['ingredients'],ndata['method']]);

def extract_data(files):
	files = eval(files);
	for f in files:
		global extract_data_path;
		extract_data_path = f;	
		with open(f+'.json','r') as fd:
			js = json.load(fd,object_hook=cuisine_json_decoder);

def get_cuisine(fname):
	base = os.path.basename(fname);
	stext = os.path.splitext(base);
	label = stext[0];
	ext = stext[1];
	if ext!='.csv':
		return None;
	print(fname);
	with open(fname) as f:
		#lines = f.readlines();
		lines = csv.reader(f);
		c = cuisine(label=label);
		i = 0;
		for l in lines:
			try:
				i = i+1;
				c.add_recipe(recipe(l));	
			except Exception as E:
				print('lines:%s'%(i));
				print(E);
		return c;

def get_corpus(path,cfile=pickled_corpus):
	corpus = cuisine_corpus();
	files = [];
	for root,dirs,f in os.walk(path):
		files.extend([os.path.join(root,sf) for sf in f]);

	for f in files:
		c = get_cuisine(f)
		if c!=None:
			corpus.add_cuisine(c);
	pickle.dump(corpus,open(cfile,'wb'));
	return corpus;

# Randomly select n recipes from each cuisine 
def divide_corpus(corpus,n):
	cuisines = []; 
	for label in corpus.cuisines.keys():
		c = corpus.cuisines[label]; 
		recipes = c.select_recipes(n);
		cuisines.append(cuisine(label=label,recipes=recipes));
	dcorpus = cuisine_corpus(cuisines=cuisines);
	return dcorpus;

# Randomly select n recipes from each cuisine 
def divide_corpus_kfold(corpus,k):
	n = sys.maxsize;
	for label in corpus.cuisines.keys():	
		c = corpus.cuisines[label]; 
		n = min(n,len(c.recipes));

	ret = [];
	prev_idx = -1;
	window_sz = math.floor(n/k);
	for i in range(0,k):
		idx1 = prev_idx+1; idx2 = window_sz + idx1-1; prev_idx = idx2;
		#print('idx1=%d idx2=%d'%(idx1,idx2));
		ctrain = []; ctest = [];
		for label in corpus.cuisines.keys():
			c = corpus.cuisines[label]; 
			rtest = c.recipes[idx1:idx2+1];
			rtrain = c.recipes[0:idx1] + c.recipes[idx2+1:];
			ctrain.append(cuisine(label=label,recipes=rtrain));
			ctest.append(cuisine(label=label,recipes=rtest));
		cctrain = cuisine_corpus(cuisines=ctrain);
		cctest = cuisine_corpus(cuisines=ctest);
		ret.append([cctrain,cctest]);
	return ret;

# Assumption folder name is part of corpus_name
def get_n_corpus(corpus_name,n,ref_corpus_loc=dpath,corpus=None):
	if corpus == None:
		corpus = pickle.load(open(ref_corpus_loc,'rb'));
	n = int(n);
	dcorpus = divide_corpus(corpus,n);
	pickle.dump(dcorpus,open(corpus_name,'wb'));

# Assumption folder name is part of corpus_name
def get_kfold_corpus(corpus_name,k,ref_corpus_loc=pickled_corpus,corpus=None):
	if corpus == None:
		corpus = pickle.load(open(ref_corpus_loc,'rb'));
	k = int(k);
	dcorpus = divide_corpus_kfold(corpus,k);
	for i in range(0,len(dcorpus)):
		trainc = dcorpus[i][0]; testc = dcorpus[i][1];	
		pickle.dump(trainc,open(corpus_name+'.train'+str(i+1),'wb'));
		pickle.dump(testc,open(corpus_name+'.test'+str(i+1),'wb'));

def get_lex_diversity(txt):
	print(float(len(set(txt)))/len(txt));
	
def get_freq_dist(tokens,n=25):
	txt = nltk.Text(tokens); 
	pdf = nltk.FreqDist(txt);
	print(pdf.most_common(n));

def get_collocations(txt):
	t = txt.collocations();
	print(t);

def get_corpus_stats(features,topn=25,cfile=pickled_corpus,corpus=None):
	topn = int(topn);
	features = eval(features);
	if corpus == None:
		corpus = pickle.load(open(cfile,'rb'));
	for label in corpus.cuisines.keys():
		print('CUISINE = %s'%(label));
		for f in features:
			print(f);
			ftokens = [];
			for recipe in corpus.cuisines[label].recipes:
				f_func = getattr(recipe,f);
				tokens = f_func();
				ftokens.extend(tokens);
			print('FEATURE = %s'%(f));
			get_freq_dist(ftokens,topn);
		print('\n');
	
# corpus, features_list, file_name to write to
# returns the labels
# Assumption: vocab storage - train,test runs alternately
def get_features(flist,fname,cfile,is_test=0,svm=1,corpus=None): 
	flist = eval(flist);
	svm = bool(int(svm));

	is_test = bool(int(is_test));
	# Load the vocabulary from training
	if is_test == True:
		cuisine_load_vocab(); 

	if corpus == None:
		corpus = pickle.load(open(cfile,'rb'));

	output = []; labels = [];
	for label in corpus.cuisines.keys():
		cuisine = corpus.cuisines[label]; cuisine_output = []; 
		number_label = str(cuisine.number_label);
		for recipe in cuisine.recipes:
			if svm:
				cuisine_output.append(recipe.get_svm_features(flist,number_label,is_test));
			else:
				cuisine_output.append(recipe.get_megam_features(flist,number_label,is_test));
		labels.extend([number_label]*len(cuisine_output));
		output.extend(cuisine_output);

	# Write the features generated
	with open(fname,'w') as fd:
		fd.writelines('\n'.join(output));

	# Write the labels generated
	with open(fname+'.label','w') as fd:
		fd.writelines('\n'.join(labels));

	# Save the vocabulary for testing
	if is_test == False:
		cuisine_save_vocab(); 

def write_vocab(ifile):
	cuisine_load_vocab(); 
	cuisine_print_vocab(ifile); 

#file1 - original labels
def run_eval(file1,file2,labels=None):
	original_dic = cuisines_num_label;
	dic = dict(zip(original_dic.values(),original_dic.keys()));
	with open(file1,'r') as fd:
		f1lines = fd.readlines();
	with open(file2,'r') as fd:
		f2lines = fd.readlines();

	labels = list(dic.keys()); # all cuisine labels	
	cl_arr = {}; # Confusion matrix 
	for i in labels:
		cl_arr[i] = {};
		for j in labels:
			cl_arr[i][j] = 0;

	plabels = {} # Original labels present
	# Calculate the matrix
	i = 0;
	while i<len(f1lines) and i<len(f2lines):
		c1 = int(f1lines[i]); c2 = int(f2lines[i]);
		plabels[c1] = 1; plabels[c2] = 1; 
		cl_arr[c1][c2] = cl_arr[c1][c2] + 1;	
		i = i+1;
	if i!=len(f1lines) or i!=len(f2lines):
		print('Files length dont match\n');

	labels = plabels.keys();
	# Print the matrix
	tot_samples = i;
	for i in labels:
		for j in labels:
			if cl_arr[i][j]!=0:
				print('%s as %s = %d'%(dic[i],dic[j],cl_arr[i][j]));
	prec = {};
	for i in labels:
		s = 0;
		# Pickup all columns for label-i
		for j in labels:
			s = s + cl_arr[j][i];
		if s!=0:
			prec[i] = cl_arr[i][i]/float(s);
		else:
			prec[i] = 0;
		print('Precision %s %g'%(dic[i],prec[i]));	
	recall = {};
	for i in labels:
		s = 0;
		# Pickup all rows for label-i
		for j in labels:
			s = s + cl_arr[i][j];
		if s!=0:
			recall[i] = cl_arr[i][i]/float(s);
		else:
			recall[i] = 0;
		recall[i] = cl_arr[i][i]/float(s);
		print('Recall %s %g'%(dic[i],recall[i]));	
	f1 = {};
	for i in labels:
		if prec[i]!=0 or recall[i]!=0:
			f1[i] = 2*((prec[i]*recall[i])/(prec[i]+recall[i]));
		else:
			f1[i] = 0;
		print('F1 score %s %g'%(dic[i],f1[i]));	
	s = 0.0;
	for i in labels:
		s = s + cl_arr[i][i];
	print('Accuracy: %g\n'%(s/tot_samples));
	return;

def parse_svm_out(infile,outfile):
	with open(infile,'r') as fd:
		lines = fd.readlines();	
	olines = [line.split()[0] for line in lines];
	with open(outfile,'w') as fd:
		lines = fd.writelines('\n'.join(olines));	

def main(func,argv):
	global tagger;
	tagger = cuisine_init();
	return func(*argv);

if __name__=='__main__':
	func = locals()[sys.argv[1]];
	main(func,sys.argv[2:]);
