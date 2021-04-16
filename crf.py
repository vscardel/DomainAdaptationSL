#para usar apenas o crf sem o lstm
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import utilidades
import pickle

import argparse

def parse_arguments(parser):
	parser.add_argument('--data_path', type=str, default="",
				help="caminho da pasta dados")
	parser.add_argument('--corpus', type=str, default="",
				help="corpus para treinar ou testar")
	parser.add_argument('--train', type=str, default="",
			help="eh para invocar o script para treino")
	parser.add_argument('--test', type=str, default="",
			help="eh para invocar o script para teste")
	parser.add_argument('--save_model_in', type=str, default="",
			help="diretorio para salvar o modelo")
	parser.add_argument('--saved_model', type=str, default="",
			help="diretorio para salvar o modelo")
	parser.add_argument('--path_results', type=str, default="",
			help="caminho do diretório dos resultados")


	
	args = parser.parse_args()
	for k in args.__dict__:
		print(k + ": " + str(args.__dict__[k]))
	return args


def load_dataset(file_path):
	f = open(file_path,'r')
	dataset = []
	current_batch = []
	for line in f.readlines():
		if line == '\n':
			dataset.append(current_batch)
			current_batch = []
		else:
			l = tuple(line.strip().split(' '))
			current_batch.append(l)
	#ultimo batch
	dataset.append(current_batch)
	return dataset

def word2features(batch,i,contextual_features):
	palavra = batch[i][0]
	features = { 
		'bias': 1.0,
		'word.lower()': palavra.lower() 
	}

	if palavra.lower() in contextual_features:

		embedding = contextual_features[palavra.lower()]

		for i,num in enumerate(embedding):
			new_key = 'f'+str(i)
			features[new_key] = num

	return features

def batch2features(batch,contextual_features):
	return [word2features(batch, i,contextual_features) for i in range(len(batch))]

def batch2labels(batch):
	return [label for token,label in batch]

def write_results(path_results,dataset,results):
	f = open(path_results,'r')
	for i,batch in enumerate(dataset):
		results_batch = results[i]
		print(len(results_batch))
		print(len(batch))

def main():

	parser = argparse.ArgumentParser(description="")
	opt = parse_arguments(parser)
	corpora = utilidades.Corpora()

	contextual_features = corpora.load_embeddings(opt.data_path +
									'/embeddings_contextuais/' +
									opt.corpus + '_embeddings_100d.txt')
	if opt.train == 'yes':

		dataset = load_dataset(opt.data_path+'/'+opt.corpus+'/train.txt')

		X_train = [batch2features(b,contextual_features) for b in dataset if b]
		y_train = [batch2labels(b) for b in dataset if b]
		

		crf = sklearn_crfsuite.CRF(
			algorithm='lbfgs',
			c1=0.1,
			c2=0.1,
			max_iterations=100,
			all_possible_transitions=True
		)
		print('treinando modelo')
		crf.fit(X_train, y_train)

		print('salvando modelo')
		pickle.dump(crf, open(opt.save_model_in, 'wb'))

	elif opt.test == 'yes':

		dataset_test = load_dataset(opt.data_path+'/'+opt.corpus+'/test.txt')

		X_test = [batch2features(b,contextual_features) for b in dataset_test if b]
		y_test = [batch2labels(b) for b in dataset_test if b]

		crf = pickle.load(open(opt.saved_model, 'rb'))
		labels = list(crf.classes_)
		y_pred = crf.predict(X_test)

		print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))

		print('escrevendo resultados')
		write_results(opt.path_results,dataset_test,y_pred)

if __name__ == "__main__":
    main()