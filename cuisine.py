#!/usr/bin/env python3

"""Multi-Cuisine detector Code
Provides object framework for Cuisine corpus:
"""
import nltk;
import sys;
import os;
import re;
import random;
import pickle;

__author__ = "Subbiah Ganesan"
__email__ = "subbiahg@usc.edu"
tagger = None;
vocab_dict = None;
lmtzr = None;

cuisines_num_label  = {
		      'CANADIAN':1,
		      'CARIBBEAN':2,
		      'CHINESE':3,
		      'EUROPEAN-UK':4,
		      'FRENCH':5,
		      'GERMAN':6,
		      'GREEK':7,
		      'INDIAN':8,
		      'ITALIAN':9,
		      'JAPANESE':10,
		      'MEXICAN':11,
		      'SOUTH-AMERICAN':12,
		      'THAI':13,
		     };

def get_ngram(tokens,n,sep='_',prefix=''): 
	op = [];
	for i in list(range(0,len(tokens)-n+1)):
		ngram = '';
		for j in range(0,n-1):
			ngram = ngram+tokens[i+j]+sep;
		ngram = ngram+tokens[i+n-1];
		op.append(prefix+ngram);
	return op;

class cuisine_corpus(object):
	def __init__(self,cuisines=None):
		self.cuisines = {};
		if cuisines != None:
			for cuisine in cuisines: 
				self.add_cuisine(cuisine);

	def add_cuisine(self,cuisine):
		if cuisine == None:
			raise Exception('No cuisine to add');
		if cuisine.label in self.cuisines.keys():
			raise Exception('%s - cuisine already present'%cuisine.label);
		self.cuisines[cuisine.label] = cuisine;

	def del_cuisine(self,cuisine):
		if cuisine.label not in self.cuisines.keys():
			raise Exception('cuisine not present for deletion');
		if cuisine != None:
			self.cuisines.pop(cuisine.label,None);
 
class cuisine(object):
	def __init__(self,label=None,recipes=None):
		label = label.upper();
		if label not in cuisines_num_label.keys():
			raise Exception('%s - cuisine not identified'%label);
		self.label = label;
		self.number_label = cuisines_num_label[self.label];
		self.recipes = [];
		if recipes != None:
			for recipe in recipes: 
				self.add_recipe(recipe);

	def add_recipe(self,recipe,check=False):
		if recipe == None:
			raise Exception('No recipe to add');
		if check:
			for r in self.recipes:
				if r==recipe:
					raise Exception('recipe already present'); 
		self.recipes.append(recipe);

	def del_recipe(self,recipe):
		for r in self.recipes:
			if r==recipe:
				self.recipes.remove(r);
				return; 
		raise Exception('recipe not present for deletion');
	
	def select_recipes(self,n):
		if n > len(self.recipes):
			raise Exception('n=%d > length=%d'%(n,len(self.recipes)));
		# randomly select n recipes
		nrecipes = random.sample(self.recipes,n); # Copy + shuffle
		nrecipes = nrecipes[0:n+1];
		assert(len(nrecipes) == n);
		return nrecipes;

	def get_ingredients_txt(self):
		r = [];
		[r.extend(recipe.ingredients) for recipe in self.recipes]
		return nltk.Text(r); 

	def get_prep_txt(self):
		r = [];
		[r.extend(step) for recipe in self.recipes for step in recipe.prep]; 
		return nltk.Text(r); 


# Assumption: title,"ingredients","prep" is the expected format
def recipe_parser(txt):
	try:
		t = list(re.search('^"(.*)".*"[(.*)]","[(.*)]"',txt).groups());
		assert(len(t)==3);
		print(len(t));
	except Exception as E:
		print(txt);
	else:
		return t;

# Builds the vocabulary on the run
# Returns token label in vocabulary and 
# number of occurences in this context as key,val
def cuisine_build_features(tokens,is_test):
	ret = [];
	for token in tokens:
		#ltoken = lmtzr.lemmatize(token);
		#ltoken = lmtzr.stem(token);
		if token in vocab_dict.keys():
			token_label = vocab_dict[token];
			ret.append(token_label);
		elif is_test == False:
			token_label = len(vocab_dict.keys()) + 1;
			vocab_dict[token] = token_label; 
			ret.append(token_label);
		else:
			print('it is here:%s'%token);
			# ignore the token
	return ret;

def basic_op_words(tokens):
	ret = [t.lower() for t in tokens]; 
	return ret;	

def remove_stop_words(tokens):
	stop_words = set(nltk.corpus.stopwords.words('english'));
	ret = [t.lower() for t in tokens if t not in stop_words]; 
	return ret;	

def stem_words(tokens):
	ret = [lmtzr.stem(t) for t in tokens]; 
	return ret;	

class recipe(object):
	def __init__(self,ltxt):
		[title,ingredients,prep] = ltxt;
		self.title = title; 
		raw_ingredients = [i.strip() for i in ingredients];
		self._ingredients = eval(ingredients);
		self.ingredients = nltk.word_tokenize(' '.join(self._ingredients));
		raw_prep = eval(prep);
		prep = [];
		for rp in raw_prep:
			prep.extend(nltk.sent_tokenize(rp));	
		self._prep = prep;
		self._pos = []; # For chunking
		self.prep = []; # one sublist inside this list for every step	
		self.prep_pos = [];	
		self.extract_prep(self._prep); 
	
	def extract_prep(self,prep):
		for step in prep:
			tokens = nltk.word_tokenize(step);
			pos = tagger.tag(tokens); # returns a list of sub-list [token,pos_token]
			self._pos.append(pos);
			self.prep.append(tokens);
			self.prep_pos.append([plist[1] for plist in pos]); 

	def __eq__(self,o2):
		return (o2!=None and self._title == o2._title);

	def ingredients_unigram(self,sep='_'):
		t = self.ingredients;
		t = basic_op_words(self.ingredients);
		return get_ngram(t,1,sep);

	def ingredients_unigram_stem(self,sep='_'):
		t = self.ingredients;
		t = stem_words(t);
		#t = remove_stop_words(self.ingredients);
		return get_ngram(t,1,sep);

	def ingredients_bigram(self,sep='_'):
		t = self.ingredients;
		#t = remove_stop_words(self.ingredients);
		return get_ngram(t,2,sep);

	def _prep_flist(self,l,order):
		tprep = [];
		if order == True:
			[tprep.extend(step) for step in l]
		else:
			[tprep.extend(step) for step in l]
		return tprep; 
		
	def prep_unigram(self,sep='_',order=False):
		tprep = self._prep_flist(self.prep,order);
		tprep = basic_op_words(tprep);
		tprep = remove_stop_words(tprep);
		return get_ngram(tprep,1,sep);

	def prep_unigram_stem(self,sep='_',order=False):
		tprep = self._prep_flist(self.prep,order);
		tprep = remove_stop_words(tprep);
		tprep = stem_words(tprep);
		return get_ngram(tprep,1,sep);

	def prep_bigram(self,sep='_',order=False):
		tprep = self._prep_flist(self.prep,order);
		tprep = remove_stop_words(tprep);
		return get_ngram(tprep,2,sep);

	def prep_unigram_pos(self,sep='_',order=False):
		tprep = self._prep_flist(self.prep_pos,order);
		return get_ngram(tprep,1,sep);

	def prep_bigram_pos(self,sep='_',order=False):
		tprep = self._prep_flist(self.prep_pos,order);
		return get_ngram(tprep,2,sep);

	def prep_trim_by_pos(self,sep='_',order=False):
		tokens = [];
		for step in self._pos:
			for pos in step:
				if pos[1] in ['NN','NNP','NNS','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']:
					tokens.append(pos[0]);
		#print(tokens);
		tokens = remove_stop_words(tokens);
		return tokens;

	def prep_chunk(self,sep='_',order=False):
		for pos in self._pos:
			chunk = nltk.ne_chunk(pos,binary=True);  # returns a nltk tree;
			print(chunk);
		#return chunk;
		return [];
		 
	# TODO: check the format of SVM multiclass
	def get_svm_features(self,features,prefix,is_test):
		ret = [prefix]; ftokens = [];
		if is_test: 
			ret = [str('1')]; # default label SVM accepts
		for f in features:
			f_func = getattr(self,f);
			tokens = f_func();
			ftokens.extend(cuisine_build_features(tokens,is_test));
		counter = {};  
		for token in ftokens:
			counter[token] = counter.get(token,0) + 1;
		ret.extend([str(f)+':'+str(counter[f]) for f in sorted(counter.keys())]);
		return '\t'.join(ret);

	def get_megam_features(self,features,prefix,is_test):
		ret = [prefix]; ftokens = [];
		if is_test: 
			ret = [str('1')]; # default label SVM accepts
		for f in features:
			f_func = getattr(self,f);
			tokens = f_func();
			ftokens.extend(tokens);
		counter = {};  
		for token in ftokens:
			counter[token] = counter.get(token,0) + 1;
		ret.extend([str(f) for f in sorted(counter.keys())]);
		return '\t'.join(ret);

def cuisine_init():
	global tagger;
	tagger = nltk.tag.perceptron.PerceptronTagger();
	global vocab_dict, lmtzr;
	vocab_dict = {};
	lmtzr = nltk.stem.snowball.EnglishStemmer();
	return tagger;

def cuisine_save_vocab(): 
	global vocab_dict;
	pickle.dump(vocab_dict,open('vocab.p','wb'));

def cuisine_load_vocab(): 
	global vocab_dict;
	vocab_dict = pickle.load(open('vocab.p','rb'));

def cuisine_print_vocab(ifile): 
	global vocab_dict;
	rdict = dict(zip(vocab_dict.values(),vocab_dict.keys()));
	t = ['%s=%s'%(k,rdict[k]) for k in rdict.keys()];
	with open(ifile,'w') as fd:
		fd.writelines('\n'.join(t));
		
