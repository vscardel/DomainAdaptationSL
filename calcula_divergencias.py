import utilidades
from gensim.models import KeyedVectors
import argparse

def parse_arguments(parser):
	parser.add_argument('--data_path', type=str, default="",
				help="define o caminho da pasta data")
	parser.add_argument('--contextual', type=str, default="",
				help="para saber se eh aproximacao dos embeddings contextuais")
	parser.add_argument('--fonte', type=str, default="",
				help="dataset fonte")
	parser.add_argument('--alvo', type=str, default="",
				help="dataset alvo")
	parser.add_argument('--originais', type=str, default="",
				help="para saber se é a aproximação de embedings originais")
	parser.add_argument('--dim_aprox', type=str, default="",
				help="se aproximado, qual a dimensao")



	args = parser.parse_args()
	for k in args.__dict__:
		print(k + ": " + str(args.__dict__[k]))
	return args

#retorna keyedVectors para um corpora especifico
def create_keywords_for_corpus(all_key,path_full_data_corpus):
	palavras_corpus = corpora.conta_palavras(path_full_data_corpus)
	model = KeyedVectors(all_key.vector_size)
	for palavra in palavras_corpus:
		if palavra in all_key:
			model.add_vector(palavra,all_key[palavra])
	return model


corpora = utilidades.Corpora()
util = utilidades.Util()

def main():

	parser = argparse.ArgumentParser(description="")
	opt = parse_arguments(parser)

	print('Carregando os Datasets')

	full_data_source = opt.data_path+'/'+opt.fonte+'/full_data.txt'
	full_data_target = opt.data_path+'/'+opt.alvo+'/full_data.txt'

	if opt.contextual == 'no':

		if opt.originais == 'yes':

			all_keys = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_originais/glove_s100.txt')

			keys_source = create_keywords_for_corpus(all_keys,full_data_source)
			keys_target = create_keywords_for_corpus(all_keys,full_data_target)

		else:
			dim = opt.dim_aprox
			#no aproximado a fonte sempre eh o harem, entao basta testar o alvo
			if opt.fonte != 'harem':
				print('o conjunto fonte deve ser o harem')
				return
			else:
				all_keys_fonte = KeyedVectors.load_word2vec_format(
				opt.data_path+
				'/embeddings_aproximados/nao_contextuais/'
				 +str(dim)+'/harem_to_'+opt.alvo+'_'+str(dim)+'.txt')

				all_keys_alvo = KeyedVectors.load_word2vec_format(
				opt.data_path+
				'/embeddings_aproximados/nao_contextuais/'
				 +str(dim)+'/pc_'+opt.alvo+'_'+str(dim)+'.txt')

			keys_source = create_keywords_for_corpus(all_keys_fonte,full_data_source)
			keys_target = create_keywords_for_corpus(all_keys_alvo,full_data_target)
	else:

		if opt.originais == 'yes':

			if opt.fonte == 'harem':
				all_keys_fonte = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/harem_embeddings_100d.txt')
			elif opt.fonte == 'geocorpus':
				all_keys_fonte = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/geocorpus_embeddings_100d.txt')
			elif opt.fonte == 'lener':
				all_keys_fonte = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/lener_embeddings_100d.txt')
			elif opt.fonte == 'cojur':
				all_keys_fonte = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/cojur_embeddings_100d.txt')

			if opt.alvo == 'harem':
				all_keys_alvo = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/harem_embeddings_100d.txt')
			elif opt.alvo == 'geocorpus':
				all_keys_alvo = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/geocorpus_embeddings_100d.txt')
			elif opt.alvo == 'lener':
				all_keys_alvo = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/lener_embeddings_100d.txt')
			elif opt.alvo == 'cojur':
				all_keys_alvo = KeyedVectors.load_word2vec_format(
				opt.data_path+'/embeddings_contextuais/cojur_embeddings_100d.txt')

			keys_source = create_keywords_for_corpus(all_keys_fonte,full_data_source)
			keys_target = create_keywords_for_corpus(all_keys_alvo,full_data_target)
		else:
			pass

	print('Calculando as Divergências:')
	kl,js = util.divergencia_KL(keys_source,keys_target)
	centr = util.centroid_diff(keys_source,keys_target)
	print('KL: ' + str(kl))
	print('JS: ' + str(js))
	print('Centroide: ' + str(centr))


if __name__ == "__main__":
    main()