from config import Dados
import utilidades
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

path_embeddings = Dados.path_embeddings

def build_matrix_observartions(dict_emb,emb_dim):
	X = np.zeros([emb_dim,len(dict_emb)])
	
	for i,word in enumerate(dict_emb):
		if len(dict_emb[word]) == emb_dim:
			X[:,i] = dict_emb[word]
	return X

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
def write_embeddings_to_file(path,corpus_dict,all_dict,emb_to_write,d_eig_corpus,d):
	try:
		f = open(path,'w')
	except:
		print('cant open file')
		return
	cont = 0
	for palavra in all_dict:
		if palavra in corpus_dict:
			emb = emb_to_write[:,cont]
			cont = cont + 1
		else:
			#reduce dimensionality of the vector
			if len(all_dict[palavra]) == d:
				emb = np.matmul(all_dict[palavra],d_eig_corpus)

		f.write(palavra + ' ')

		for i,num in enumerate(emb):
			if i == len(emb)-1:
				f.write(str(num) + '\n')
			else:
				f.write(str(num) + ' ')


dim = 100
file_embeddings = path_embeddings['glove_100']
print('carregando o dicionário de embeddings')
all_embeddings = utilidades.Corpora().load_embeddings(file_embeddings)

print('###################')
print('carregando os embeddings dos corporas')
#receber as dimensoes como argumento dps
harem_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,Dados.pytorch['full_data_harem'],dim)
geocorpus_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,Dados.pytorch['full_data_geocorpus'],dim)
lener_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,Dados.pytorch['full_data_lener'],dim)
cojur_embeddings = utilidades.Corpora().load_dataset_embeddings(all_embeddings,Dados.pytorch['full_data_cojur'],dim)

harem_contextual_embeddings = 


print('###################')
print('construindo matrizes de observacao')
X = build_matrix_observartions(harem_embeddings,dim)
Y = build_matrix_observartions(geocorpus_embeddings,dim)
Z = build_matrix_observartions(lener_embeddings,dim)
alfa = build_matrix_observartions(cojur_embeddings,dim)

print('###################')
print('computando as matrizes de covariancia')
cov_X = np.cov(X)
cov_Y = np.cov(Y)
cov_Z = np.cov(Z)
cov_alfa = np.cov(alfa)

print('###################')
print('realizando eigendecomposition')
U_harem,S_harem,W_harem = np.linalg.svd(cov_X)
U_geo,S_geo,W_geo = np.linalg.svd(cov_Y)
U_lener,S_lener,W_lener = np.linalg.svd(cov_Z)
U_cojur,S_cojur,W_cojur = np.linalg.svd(cov_alfa)


# print('plotando autovalores em escalar logaritmica')
# plot_log_scale(S_cojur)

# unico hiperparametro do metodo
d = 20

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
pc_geocorpus = compute_principal_components(geocorpus_embeddings,d_eig_geo,d)
pc_lener = compute_principal_components(lener_embeddings,d_eig_lener,d)
pc_cojur = compute_principal_components(cojur_embeddings,d_eig_cojur,d)


print('###################')
print('alinhando corporas')
print('harem -> geocorpus')
emb_harem_geo = align_corporas(pc_harem,d_eig_harem,d_eig_geo)
print('harem -> lener')
emb_harem_lener = align_corporas(pc_harem,d_eig_harem,d_eig_lener)
print('harem -> cojur')
emb_harem_cojur = align_corporas(pc_harem,d_eig_harem,d_eig_cojur)

print('###################')
print('escrevendo embeddings para os arquivos')
print('file harem_to_geocorpus')
write_embeddings_to_file('pytorch_lstmcrf/data/harem_to_geo_20.txt',harem_embeddings,all_embeddings,emb_harem_geo,d_eig_harem,d)
print('file harem_to_lener')
write_embeddings_to_file('pytorch_lstmcrf/data/harem_to_lener_20.txt',harem_embeddings,all_embeddings,emb_harem_lener,d_eig_harem,d)
print('file harem_to_cojur')
write_embeddings_to_file('pytorch_lstmcrf/data/harem_to_cojur_20.txt',harem_embeddings,all_embeddings,emb_harem_cojur,d_eig_harem,d)

print('file pc_geocorpus')
write_embeddings_to_file('pytorch_lstmcrf/data/pc_geocorpus_20.txt',geocorpus_embeddings,all_embeddings,pc_geocorpus,d_eig_geo,d)
print('file pc_lener')
write_embeddings_to_file('pytorch_lstmcrf/data/pc_lener_20.txt',lener_embeddings,all_embeddings,pc_lener,d_eig_lener,d)
print('file pc_cojur')
write_embeddings_to_file('pytorch_lstmcrf/data/pc_cojur_20.txt',cojur_embeddings,all_embeddings,pc_cojur,d_eig_cojur,d)