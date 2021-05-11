from config import Dados
import utilidades
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import argparse

def parse_arguments(parser):
	parser.add_argument('--data_path', type=str, default="",
				help="define o caminho da pasta data")
	parser.add_argument('--d', type=str, default="",
				help="dimensao do subespaço aproximado")
	parser.add_argument('--contextual', type=str, default="",
				help="para saber se eh aproximacao dos embeddings contextuais")
	parser.add_argument('--original', type=str, default="",
				help="para saber se eh para fornecer a div dos originais")


	args = parser.parse_args()
	for k in args.__dict__:
		print(k + ": " + str(args.__dict__[k]))
	return args

#para converter as matrizes obtidas pelo metodo de aproximacao
#em dicionarios palavra->vetor
#shape(M) = dim X num_embeddings_corpus
def convert_matrix_to_dict(dict_corpus,M):
	cont = 0
	new_dict = {}
	for palavra in dict_corpus: 
		new_dict[palavra] = M[:,cont]
		cont = cont + 1
	return new_dict

def convert_to_gensim(dict_corpus):
	for i,palavra in enumerate(dict_corpus):
		if i == 0:
			model = KeyedVectors(len(dict_corpus[palavra]))
		model.add_vector(palavra,dict_corpus[palavra])
	return model

#source e target no formato gensim
def calcula_devergencias(source,target,util):
	kl,js = util.divergencia_KL(source,target)
	centr = util.centroid_diff(source,target)
	print('KL: ' + str(kl))
	print('JS: ' + str(js))
	print('Centroide: ' + str(centr))


def build_matrix_observartions(dict_emb,emb_dim):
	X = np.zeros([emb_dim,len(dict_emb)])
	
	for i,word in enumerate(dict_emb):

		if len(dict_emb[word]) == emb_dim:
			X[:,i] = dict_emb[word]

	#normalize matrix of observations
	mean_vector = X.mean(1)
	mean_matrix = np.tile(np.array([mean_vector]).transpose(), (1, len(dict_emb)))
	normalized_observations = X - mean_matrix
	return normalized_observations

def plot_log_scale(values):
	plt.figure(1)
	plt.semilogy(np.diag(values))
	plt.title('singular values')
	plt.show()

def select_d_first_eigenvectors(U,d):
	dim = len(U[:,0])
	d_eig = np.zeros([dim,d])
	for i in range(d):
		d_eig[:,i] = U[:,i]
	return d_eig

def compute_principal_components(dict_emb,d_eig,d):
	dim = len(d_eig[:,0])
	pc = np.zeros([d,len(dict_emb)])
	for i,palavra in enumerate(dict_emb):
		emb_transposed = np.zeros([1,dim])
		emb_transposed[0] = dict_emb[palavra]
		pc[:,i] = np.matmul(emb_transposed,d_eig)
	return pc

def align_corporas(pc_s,d_eig_s,d_eig_t):
	M = np.matmul(np.transpose(d_eig_s),d_eig_t)
	aligned_emb = np.matmul(np.transpose(pc_s),M)
	return np.transpose(aligned_emb)

#supondo q tudo esta na ordem certa
def write_embeddings_to_file(path,corpus_dict,all_dict,emb_to_write,d_eig_corpus):
	try:
		f = open(path,'w')
	except:
		print('cant open file')
		return

	dict_to_write = {}

	cont = 0
	dim = len(emb_to_write[:,0])

	for palavra in corpus_dict:
		emb = emb_to_write[:,cont]
		cont = cont + 1
		dict_to_write[palavra] = emb

	for palavra in all_dict:
		if palavra not in corpus_dict:
			#reduce dimensionality of the vector
			emb = np.matmul(all_dict[palavra],d_eig_corpus)
			dict_to_write[palavra] = emb
		
	#numero de linhas e dimensao dos embeddings
	f.write(str(len(dict_to_write)-1) + ' ' + str(dim) + '\n')

	for i,palavra in enumerate(dict_to_write):

		if palavra != '\n' and palavra != ' ':
			f.write(palavra + ' ')
			emb = dict_to_write[palavra]
			for i,num in enumerate(emb):
				if i == len(emb)-1:
					f.write(str(num) + '\n')
				else:
					f.write(str(num) + ' ')


def main():

	parser = argparse.ArgumentParser(description="")
	opt = parse_arguments(parser)

	util = utilidades.Util()

	dim = 100
	
	print('###################')
	print('carregando os embeddings dos corporas')

	all_embeddings = utilidades.Corpora().load_embeddings(opt.data_path + '/embeddings_originais/glove_s100.txt')

	#receber as dimensoes como argumento dps
	if opt.contextual == 'no':

		harem_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,opt.data_path+'/harem/full_data.txt',dim)
		geocorpus_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,opt.data_path+'/geocorpus/full_data.txt',dim)
		lener_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,opt.data_path+'/lener/full_data.txt',dim)
		cojur_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,opt.data_path+'/cojur/full_data.txt',dim)

		print(len(harem_embeddings))
		print(len(geocorpus_embeddings))
		print(len(lener_embeddings))
		print(len(cojur_embeddings))


	elif opt.contextual == 'yes': 

		all_harem_embeddings = utilidades.Corpora().load_embeddings(opt.data_path + '/embeddings_contextuais/harem_embeddings_100d.txt')
		all_geo_embeddings = utilidades.Corpora().load_embeddings(opt.data_path + '/embeddings_contextuais/geocorpus_embeddings_100d.txt')
		all_lener_embeddings = utilidades.Corpora().load_embeddings(opt.data_path + '/embeddings_contextuais/lener_embeddings_100d.txt')
		all_cojur_embeddings = utilidades.Corpora().load_embeddings(opt.data_path + '/embeddings_contextuais/cojur_embeddings_100d.txt')

		harem_embeddings = utilidades.Corpora().load_dataset_embeddings(all_harem_embeddings,opt.data_path+'/harem/full_data.txt',dim)
		geocorpus_embeddings = utilidades.Corpora().load_dataset_embeddings(all_geo_embeddings,opt.data_path+'/geocorpus/full_data.txt',dim)
		lener_embeddings = utilidades.Corpora().load_dataset_embeddings(all_lener_embeddings,opt.data_path+'/lener/full_data.txt',dim)
		cojur_embeddings = utilidades.Corpora().load_dataset_embeddings(all_cojur_embeddings,opt.data_path+'/cojur/full_data.txt',dim)

		print(len(harem_embeddings))
		print(len(geocorpus_embeddings))
		print(len(lener_embeddings))
		print(len(cojur_embeddings))

	else:
		print('valor invalido para o argumento contextual')
		return

	if opt.original == 'yes':

		print('convertendo os embeddings origniais para o formato gensim')

		harem_embeddings_gensim = convert_to_gensim(harem_embeddings)
		geocorpus_embeddings_gensim = convert_to_gensim(geocorpus_embeddings)
		lener_embeddings_gensim = convert_to_gensim(lener_embeddings)
		cojur_embeddings_gensim = convert_to_gensim(cojur_embeddings)

		print(len(harem_embeddings_gensim))
		print(len(geocorpus_embeddings_gensim))
		print(len(lener_embeddings_gensim))
		print(len(cojur_embeddings_gensim))

		print('calculando divergencia dos embeddings originais')

		print('harem/geo')
		calcula_devergencias(harem_embeddings_gensim,geocorpus_embeddings_gensim,util)

		print('harem/lener')
		calcula_devergencias(harem_embeddings_gensim,lener_embeddings_gensim,util)

		print('harem/cojur')
		calcula_devergencias(harem_embeddings_gensim,cojur_embeddings_gensim,util)

	print('###################')
	print('construindo matrizes de observacao')
	harem_matrix = build_matrix_observartions(harem_embeddings,dim)
	geocorpus_matrix = build_matrix_observartions(geocorpus_embeddings,dim)
	lener_matrix = build_matrix_observartions(lener_embeddings,dim)
	cojur_matrix = build_matrix_observartions(cojur_embeddings,dim)

	print('###################')
	print('computando as matrizes de covariancia')
	cov_X = np.cov(harem_matrix)
	cov_Y = np.cov(geocorpus_matrix)
	cov_Z = np.cov(lener_matrix)
	cov_alfa = np.cov(cojur_matrix)

	print('###################')
	print('realizando eigendecomposition')
	U_harem,S_harem,W_harem = np.linalg.svd(cov_X)
	U_geo,S_geo,W_geo = np.linalg.svd(cov_Y)
	U_lener,S_lener,W_lener = np.linalg.svd(cov_Z)
	U_cojur,S_cojur,W_cojur = np.linalg.svd(cov_alfa)


	# print('plotando autovalores em escala logaritmica')
	# plot_log_scale(S_cojur)

	# unico hiperparametro do metodo

	d = int(opt.d)

	print('###################')
	print('selecionando os d=' + str(d) + ' primeiros autovetores')
	d_eig_harem = select_d_first_eigenvectors(U_harem,d)
	d_eig_geo = select_d_first_eigenvectors(U_geo,d)
	d_eig_lener = select_d_first_eigenvectors(U_lener,d)
	d_eig_cojur = select_d_first_eigenvectors(U_cojur,d)


	print('###################')
	print('computando componentes principais')
	#os componentes do harem serao alinhados com o corpus alvo
	pc_harem = compute_principal_components(harem_embeddings,d_eig_harem,d)
	dict_pc_harem = convert_matrix_to_dict(harem_embeddings,pc_harem)
	pc_geocorpus = compute_principal_components(geocorpus_embeddings,d_eig_geo,d)
	dict_pc_geocorpus = convert_matrix_to_dict(geocorpus_embeddings,pc_geocorpus)
	pc_lener = compute_principal_components(lener_embeddings,d_eig_lener,d)
	dict_pc_lener = convert_matrix_to_dict(lener_embeddings,pc_lener)
	pc_cojur = compute_principal_components(cojur_embeddings,d_eig_cojur,d)
	dict_pc_cojur = convert_matrix_to_dict(cojur_embeddings,pc_cojur)



	print('###################')
	print('alinhando corporas')
	print('harem -> geocorpus')
	emb_harem_geo = align_corporas(pc_harem,d_eig_harem,d_eig_geo)
	dict_harem_to_geo = convert_matrix_to_dict(harem_embeddings,emb_harem_geo)
	print('harem -> lener')
	emb_harem_lener = align_corporas(pc_harem,d_eig_harem,d_eig_lener)
	dict_harem_to_lener = convert_matrix_to_dict(harem_embeddings,emb_harem_lener)
	print('harem -> cojur')
	emb_harem_cojur = align_corporas(pc_harem,d_eig_harem,d_eig_cojur)
	dict_harem_to_cojur = convert_matrix_to_dict(harem_embeddings,emb_harem_cojur)

	print('Computando Divergências dos embeddings aproximados')

	emb_harem_geo_gensim = convert_to_gensim(dict_harem_to_geo)
	emb_harem_lener_gensim = convert_to_gensim(dict_harem_to_lener)
	emb_harem_cojur_gensim = convert_to_gensim(dict_harem_to_cojur)

	pc_geocorpus_gensim = convert_to_gensim(dict_pc_geocorpus)
	pc_lener_gensim = convert_to_gensim(dict_pc_lener)
	pc_cojur_gensim = convert_to_gensim(dict_pc_cojur)

	print(len(emb_harem_geo_gensim))
	print(len(emb_harem_lener_gensim))
	print(len(emb_harem_cojur_gensim))

	print(len(pc_geocorpus_gensim))
	print(len(pc_lener_gensim))
	print(len(pc_cojur_gensim))

	print('harem -> geo')
	calcula_devergencias(emb_harem_geo_gensim,pc_geocorpus_gensim,util)

	print('harem -> lener')
	calcula_devergencias(emb_harem_lener_gensim,pc_lener_gensim,util)

	print('harem -> cojur')
	calcula_devergencias(emb_harem_cojur_gensim,pc_cojur_gensim,util)

	print('###################')
	print('escrevendo embeddings para os arquivos')

	if opt.contextual == 'yes':

		print('file harem_to_geocorpus')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'+
			'harem_to_geocorpus_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_geo,d_eig_harem)

		print('file harem_to_lener')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'+
			'harem_to_lener_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_lener,d_eig_harem)

		print('file harem_to_cojur')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'+
			'harem_to_cojur_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_cojur,d_eig_harem)

		print('file pc_geocorpus')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'+
				'pc_geocorpus_'+str(d)+
				'.txt',geocorpus_embeddings,all_embeddings,pc_geocorpus,d_eig_geo)

		print('file pc_lener')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'
				'pc_lener_'+str(d)+
				'.txt',lener_embeddings,all_embeddings,pc_lener,d_eig_lener)

		print('file pc_cojur')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/contextuais/'+str(d)+'/'
				'pc_cojur_'+str(d)+
				'.txt',cojur_embeddings,all_embeddings,pc_cojur,d_eig_cojur)
	else:

		print('file harem_to_geocorpus')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
			'harem_to_geocorpus_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_geo,d_eig_harem)

		print('file harem_to_lener')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
			'harem_to_lener_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_lener,d_eig_harem)

		print('file harem_to_cojur')

		write_embeddings_to_file(opt.data_path+
			'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
			'harem_to_cojur_'+str(d)+
			'.txt',harem_embeddings,all_embeddings,emb_harem_cojur,d_eig_harem)

		print('file pc_geocorpus')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
				'pc_geocorpus_'+str(d)+
				'.txt',geocorpus_embeddings,all_embeddings,pc_geocorpus,d_eig_geo)

		print('file pc_lener')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
				'pc_lener_'+str(d)+
				'.txt',lener_embeddings,all_embeddings,pc_lener,d_eig_lener)

		print('file pc_cojur')

		write_embeddings_to_file(opt.data_path+
				'/embeddings_aproximados_normalizados/nao_contextuais/'+str(d)+'/'
				'pc_cojur_'+str(d)+
				'.txt',cojur_embeddings,all_embeddings,pc_cojur,d_eig_cojur)


if __name__ == "__main__":
    main()