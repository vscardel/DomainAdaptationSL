import os
from bs4 import BeautifulSoup
import string
import nltk
import math
from scipy import spatial
from scipy import stats
import numpy as np
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer
from config import Dados
from gensim.models import KeyedVectors

path_corpora = Dados.path_corpora
path_embeddings = Dados.path_embeddings


class Corpora:

	# retorna o corpus harem no formato BIO. o formato eh
	# uma palavra por linha, com sua tag separada por um
	# espaco. Sentencas distintas estao separadas por uma
	# quebra de linha

	def get_BIO_harem(self,com_classe = False):

		util = Util()
		
		corpus_path_harem_tagged = path_corpora['harem']
		corpus_path_harem_docs = path_corpora['harem_docs']

		f_harem_tagged = open(corpus_path_harem_tagged,encoding = 'ISO-8859-1')
		f_harem_docs = open(corpus_path_harem_docs,encoding = 'ISO-8859-1')

		xt_harem_tagged =  BeautifulSoup(f_harem_tagged,'lxml')
		xt_harem_docs =  BeautifulSoup(f_harem_docs,'lxml')

		#documentos anotados
		docs_tagged = xt_harem_tagged.find_all('doc')
		doc_ids = [doc.get('docid') for doc in docs_tagged]
		all_docs = xt_harem_docs.find_all('doc')

		docs_non_tagged = []

		#get the non anotated doc version
		for doc_id in doc_ids:
			for doc in all_docs:
				if doc.get('docid') == doc_id:
					docs_non_tagged.append(doc)
		
		#paragraphs in both versions, tagged and non_tagged
		p_tagged = []
		p_non_tagged = []

		for i in range(len(docs_tagged)):

			p_tagged_curr = docs_tagged[i].find_all('p')
			p_non_tagged_curr = docs_non_tagged[i].find_all('p')

			#essa condicao exclui um dos 129 documentos do harem
			#por algum motivo o numero de paragrafos nao eh o mesmo
			#entao prefiro nao incluir do q deixar passar alguma
			#inconsistencia
			if len(p_tagged_curr) == len(p_non_tagged_curr):
				p_tagged += p_tagged_curr
				p_non_tagged = p_non_tagged + p_non_tagged_curr

		BIO_corpus = []

		for i in range(len(p_tagged)):

			list_em_text = [str(em.text) for em in p_tagged[i].find_all('em')]
			list_em_classes = [str(em.get('categ')) for em in p_tagged[i].find_all('em')]

			clean_text = str(p_non_tagged[i].text)
			#substring matching
			for i,em_text in enumerate(list_em_text):
				clean_text = util.substring_marking(em_text,clean_text,list_em_classes[i],com_classe)

			words_text = nltk.word_tokenize(clean_text,language='portuguese')
			for cont,word in enumerate(words_text):
				if len(word.split('-')) == 1:
					words_text[cont] = words_text[cont] + '-O'

			marked_text = TreebankWordDetokenizer().detokenize(words_text)
			BIO_corpus.append(marked_text)

		return BIO_corpus

	#recebe a saida de get_bio_harem
	#e um arquivo para escrever os resultados
	def write_harem(self,file_path,harem):
		f = open(file_path,'w')
		for p in harem:
			palavras = p.split()
			for palavra in palavras:
				p_tags = palavra.split('-')
				p = p_tags[0]
				#caso O
				if len(p_tags) == 2:
					f.write(p + ' O' + '\n')
				#caso classe
				elif len(p_tags) == 3:
					f.write(p + ' ' + p_tags[1] + '-' + p_tags[2] + '\n')
			f.write('\n')

	#auxiliar para escrever os arquivos da funcao partition corpra
	def aux_partition_corpora(self,f_obj,list_of_p):
		for p in list_of_p:
			for line in p:
				f_obj.write(line)
			f_obj.write('\n')

	#recebe um arquivo full_data.txt e o tamanho do conj de treino e cria
	#os arquivos train, dev e text. dev e test possuem o mesmo tamanho
	def partition_corpora(self,path_full_data,path_corpora,size_of_train):
		f = open(path_full_data,'r')
		p = []
		list_p = []
		for line in f.readlines():
			if line == '\n':
				list_p.append(p)
				p = []
			else:
				p.append(line)
		#embaralha os indices para que a distribuicao dos dados
		#seja igualitaria
		arr = np.arange(len(list_p))
		np.random.shuffle(arr)
		shuffled_list_p = []
		for idx in arr:
			shuffled_list_p.append(list_p[idx])

		size_of_train = int(len(shuffled_list_p)*0.8)
		train = shuffled_list_p[:size_of_train+1]
		rest = shuffled_list_p[size_of_train+1:]
		dev = rest[:len(rest)//2]
		test = rest[len(rest)//2:]


		f_train = open(path_corpora + '/train.txt','w')
		f_dev = open(path_corpora + '/dev.txt','w')
		f_test = open(path_corpora + '/test.txt','w')

		self.aux_partition_corpora(f_train,train)
		self.aux_partition_corpora(f_dev,dev)
		self.aux_partition_corpora(f_test,test)
	

	#recebe corpus no formato BIO e retorna
	#dict com formato palavra -> cont_palavra
	def conta_palavras(self,corpus_path):
		try:
			f = open(corpus_path,'r')
		except:
			print('nao foi possivel abrir o arquivo')		
			return
		cont_palavras = {}
		lines = f.readlines()
		for line in lines:
			palavra = line.split(' ')[0]
			if palavra not in cont_palavras:
				cont_palavras[palavra.lower()] = 1
			else:
				cont_palavras[palavra.lower()] += 1
		return cont_palavras

	def load_embeddings(self,embedding_path):
		f = open(embedding_path,'r')
		embedding_dict = {}
		for i,line in enumerate(f.readlines()):
			#ignore the first line
			if i != 0:
				split_line = line.split()
				word = split_line[0]
				embedding = split_line[1:]
				if word not in embedding_dict:
					try:
						embedding_dict[word.lower()] = np.array(embedding).astype(np.float)
					except:
						#one line on the file have a character invading the embedding
						embedding = embedding[1:]
						embedding_dict[word.lower()] = np.array(embedding).astype(np.float)

		return embedding_dict

	def load_dataset_embeddings(self,embeddings_dict,corpus_path,embedding_dim):
		corpus_embedding_dict = {}
		corpus_dict = self.conta_palavras(corpus_path)
		for palavra in corpus_dict:
			if palavra in embeddings_dict:
				corpus_embedding_dict[palavra] = embeddings_dict[palavra]
			else:
				corpus_embedding_dict[palavra] = np.random.rand(embedding_dim)
		return corpus_embedding_dict


	def write_on_gensim_format(self,path_corpora,path_embeddings,path_new_file,dim):
		f = open(path_new_file,'w')
		print('carregando modelo gensim')
		model_all_embeddings = KeyedVectors.load_word2vec_format(path_embeddings)
		palavras_corpora = self.conta_palavras(path_corpora)
		cont = 0
		print('escrevendo embeddings')
		f.write(str(len(palavras_corpora)) + ' ' + str(dim) + '\n')
		for palavra in palavras_corpora:
			if palavra in model_all_embeddings:
				embedding = model_all_embeddings[palavra]
			else:
				cont = cont + 1
				embedding = np.random.rand(dim)

			f.write(palavra + ' ')
			for i,num in enumerate(embedding):
				if i == len(embedding)-1:
					f.write(str(num) + '\n')
				else:
					f.write(str(num) + ' ')
		print('palavras inicializadas aleatoriamente: ' + str(cont))

class Util:

	#marca a substring da entidade nomeada no texto
	def substring_marking(self,subs,string,classe,com_classe):

		words_subs = nltk.word_tokenize(subs,language='portuguese')
		words_string = nltk.word_tokenize(string,language='portuguese')

		pos_subs = 0

		for cont,word_string in enumerate(words_string):

			if pos_subs == len(words_subs):

				pos_ini = cont-len(words_subs)
				off_set = len(words_subs)
				for j in range(pos_ini,pos_ini+off_set):
					if j == pos_ini:
						if com_classe:
							classe = classe.split('|')[0]
							words_string[j] = words_string[j] + '-B' + '-' + str(classe)
						else:
							words_string[j] = words_string[j] + '-B' 
					else:
						if com_classe:
							classe = classe.split('|')[0]
							words_string[j] = words_string[j] + '-I' + '-' + str(classe)
						else:
							words_string[j] = words_string[j] + '-I'

				pos_subs = 0

			if word_string == words_subs[pos_subs]:
				pos_subs = pos_subs + 1
			else:
				pos_subs = 0

		return TreebankWordDetokenizer().detokenize(words_string)


	#recebe dois arquivos no formato gensim e calcula a divergencia
	#kl entre eles.

	def divergencia_KL(self,glove_model_source,glove_model_target):

		p_distribution_target,p_distribution_source = np.array([]),np.array([])

		list_palavra_source = [palavra for palavra in glove_model_source.key_to_index]
		list_palavra_target = [palavra for palavra in glove_model_target.key_to_index]

		union = list(set(list_palavra_source+list_palavra_target))

		num_sample = 10
		cont_outside = 0

		for i,palavra in enumerate(union):

			cont_s,cont_t = 0,0

			if palavra in glove_model_source:
				emb_palavra = glove_model_source[palavra]
				most_similar_source = glove_model_source.most_similar(positive=[palavra],topn=num_sample)
				distancias_source = [ms[1] for ms in most_similar_source]
				distancia_media_source = sum(distancias_source)*(1/num_sample)
				# print(most_similar_source)
			else:
				most_similar_source = False

			if palavra in glove_model_target:
				emb_palavra = glove_model_target[palavra]
				most_similar_target = glove_model_target.most_similar(positive=[palavra],topn=num_sample)
				distancias_target = [ms[1] for ms in most_similar_target]
				distancia_media_target = sum(distancias_target)*(1/num_sample)
				# print(most_similar_target)
			else:
				most_similar_target = False

			for j in range(num_sample):

				if most_similar_source:
					curr_MS_word_source = most_similar_source[j][0] 
					curr_similarity_with_word_source = most_similar_source[j][1]
					if curr_similarity_with_word_source >= distancia_media_source:
						cont_s += 1
				else:
					cont_outside += 1
					break

				if most_similar_target:

					curr_MS_word_target = most_similar_target[j][0] 
					curr_similarity_with_word_target = most_similar_target[j][1]
					if curr_similarity_with_word_target >= distancia_media_target:
						cont_t += 1
				else:
					cont_outside += 1
					break

			p_source = cont_s/num_sample
			p_target = cont_t/num_sample

			p_distribution_source = np.append(p_distribution_source,p_source)
			p_distribution_target = np.append(p_distribution_target,p_target)

			# if i % 100 == 0:
			# 	print(str(round(i/len(union)*100)) + ' % ' + 'completo')
			# 	# print(spatial.distance.cosine(emb,dict_s[amostra_s[j]]))

		p_outside = cont_outside/len(union)
		p_distribution_target = np.append(p_distribution_target,p_outside)
		p_distribution_source = np.append(p_distribution_source,p_outside)

		#kl pura
		kl = stats.entropy(p_distribution_target,p_distribution_source)

		#kl simetrica  (js)
		m = 1./2*(p_distribution_source + p_distribution_target)
		js = stats.entropy(p_distribution_source,m, base=np.e)/2. +  stats.entropy(p_distribution_target, m, base=np.e)/2.

		return kl,js

	def calculate_mean(self,dict_emb,emb_dim):
		X = [dict_emb[j] for j in range(len(dict_emb))]
		return np.mean(X,axis=0)

	def centroid_diff(self,kv_source,kv_target):
		emb_dim = kv_source.vector_size
		mean_s = self.calculate_mean(kv_source,emb_dim)
		mean_t = self.calculate_mean(kv_target,emb_dim)
		return spatial.distance.euclidean(mean_s,mean_t)