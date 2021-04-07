import utilidades
from gensim.models import KeyedVectors

corpora = utilidades.Corpora()
util = utilidades.Util()

path_harem_gensim = 'pytorch_lstmcrf/data/harem_seg/harem_gensim_format.txt'
path_geo_gensim = 'pytorch_lstmcrf/data/geocorpus/geocorpus_gensim_format.txt'
path_lener_gensim = 'pytorch_lstmcrf/data/lener_br/lener_gensim_format.txt'
path_cojur_gensim = 'pytorch_lstmcrf/data/cojur/cojur_gensim_format.txt'

path_harem_to_geo = 'pytorch_lstmcrf/data/geocorpus/harem_to_gensim_format.txt'
path_geo_gensim_aproximado = 'pytorch_lstmcrf/data/geocorpus/geocorpus_gensim_format_aproximado.txt'

# harem_embeddings = corpora.load_dataset_embeddings(all_embeddings,path_harem,100)
# target_embeddings = corpora.load_dataset_embeddings(all_embeddings,path_,100)

print('calculando as divergencias')

print('KL - carregando glove embeddings pelo gensim')

kv_harem = KeyedVectors.load_word2vec_format(path_harem_gensim)
kv_geo = KeyedVectors.load_word2vec_format(path_geo_gensim)

kv_harem_aproximado = KeyedVectors.load_word2vec_format(path_harem_to_geo)
kv_geo_aproximado = KeyedVectors.load_word2vec_format(path_geo_gensim_aproximado)

print('Divergência para os datasets originais')
kl,js = util.divergencia_KL(kv_harem,kv_geo)
centr = util.centroid_diff(kv_harem,kv_geo)
print('KL: ' + str(kl))
print('JS: ' + str(js))
print('Centroide: ' + str(centr))

print()

print('Divergências para os datasets aproximados')
kl,js = util.divergencia_KL(kv_harem_aproximado,kv_geo_aproximado)
centr = util.centroid_diff(kv_harem_aproximado,kv_geo_aproximado)
print('KL: ' + str(kl))
print('JS: ' + str(js))
print('Centroide: ' + str(centr))