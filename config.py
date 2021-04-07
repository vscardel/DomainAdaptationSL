#neste arquivo estão as variaveis globais e caminhos dos recursos

import os

class Dados:

	work_path = "/home/victor/TCC/experimentos"
	path_modelos = work_path + '/modelos/'

	path_corpora = {
		#colecao dourada segundo harem
		'harem': work_path + "/corpora/PacoteRecursosSegundoHAREM/lampada2.0/coleccoes/CDSegundoHAREMReRelEM.xml",
		'harem_docs': work_path + "/corpora/PacoteRecursosSegundoHAREM/lampada2.0/coleccoes/colSegundoHAREM.xml",
		'lener_dev': work_path + "/corpora/LeNER-Br/leNER-Br/dev/",
		'lener_train': work_path + "/corpora/LeNER-Br/leNER-Br/train/",
		'lener_test': work_path + "/corpora/LeNER-Br/leNER-Br/test/",
		'lener_raw': work_path + "/corpora/LeNER-Br/leNER-Br/raw_text",
		'geo_corpus': work_path + '/corpora/geo_corpus/geo_corpus.txt',
		'exp1': work_path + '/corpora/experimento_1',
		'harem_exp1': work_path + '/corpora/experimento_1/corpus_segmentado_para_segmentado_e_categorizado_harem.txt',
		'S-HAREM': work_path + '/corpora/S-HAREM/'
	}

	path_embeddings = {
		'glove_50': work_path + "/embeddings/glove/glove_s50.txt",
		'glove_100': work_path + '/pytorch_lstmcrf/data/glove_s100.txt'
	}

	modelos = {
		'experimento_1': work_path + path_modelos + 'modelo_exp1.crfsuite',
		'segmentador': work_path + path_modelos + 'segmentador.crfsuite'
	}

	pytorch = {
		'full_data_harem':work_path + '/pytorch_lstmcrf/data/harem_seg/full_data.txt', 
		'full_data_geocorpus': work_path + '/pytorch_lstmcrf/data/geocorpus/full_data.txt',
		'full_data_lener': work_path + '/pytorch_lstmcrf/data/lener_br/full_data.txt',
		'full_data_cojur': work_path + '/pytorch_lstmcrf/data/cojur/full_data.txt'
	}

