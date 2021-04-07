import argparse
import random
import numpy as np
from src.config import Config, ContextEmb, evaluate_batch_insts, write_results
import time
from src.model import NNCRF
import torch
from typing import List
from termcolor import colored
import os
from src.config.utils import load_elmo_vec
from src.config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import NERDataset
from src.data.data_utils import build_word_idx
from torch.utils.data import DataLoader
from src.config.utils import get_optimizer, lr_decay
from src.data import Instance

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="conll2003_sample")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=100, help="early stop when there is n epoch not increasing on dev")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--static_context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="static contextual word embedding, our old ways to incorporate ELMo and BERT.")
    parser.add_argument('--add_iobes_constraint', type=int, choices=[0, 1], default=0,
                        help="add IOBES constraint for transition parameters to enforce valid transitions")

    #meus hiperparametros
    parser.add_argument('--my_test_file', type=str, default='',
                        help="indicate file to apply the model")
   
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args




#cria uma instância de NERDataset do arquivo de treino.
#essa eh a instancia q tem label2idx
#por padrao, usarei o harem para construir label2idx

parser = argparse.ArgumentParser(description="LSTM CRF implementation")
opt = parse_arguments(parser)

#apenas para criar as labels
train_dataset = NERDataset('data/harem_seg/train.txt',True)
idx2labels = train_dataset.idx2labels

test_file = opt.my_test_file
test_dataset = NERDataset(test_file,False,train_dataset.label2idx)

#passa a lista vazia para dev, ja que nao sera usado
word2idx, _, char2idx, _ = build_word_idx(train_dataset.insts,[],test_dataset.insts)

num_workers = 8
label_size = len(train_dataset.label2idx)

#precisa chamar pra instanciar uns atributos, por mais que nao va usar elmo embeddings
test_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx, elmo_vecs=None)

#se nao me engano isso ja pode ser passado para o model
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers,
                                     collate_fn=test_dataset.collate_fn)

parser = argparse.ArgumentParser(description="LSTM CRF implementation")
opt = parse_arguments(parser)
conf = Config(opt)

conf.build_emb_table(word2idx=word2idx)

#carregando modelo
folder_name = "modelos_treinados_100/harem_model"
f = open(folder_name + "/config.conf", 'rb')
model = NNCRF(pickle.load(f))
model.load_state_dict(torch.load(folder_name + "/lstm_crf.m", map_location = "cpu"))
model.eval()

batch_size = test_dataloader.batch_size
#avaliando

p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
insts = test_dataloader.dataset.insts
batch_id = 0
dev = "cpu"

#arquivo passado como argumento para eval para escrever os resultados
f = open('/home/victor/TCC/experimentos/pytorch_lstmcrf/results/bla.txt','w')

with torch.no_grad():

	for iter, batch in tqdm(enumerate(test_dataloader, 1), desc="--evaluating batch", total=len(test_dataloader)):
		
		one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]

		batch_id += 1

		batch_max_scores, batch_max_ids = model.decode(words = batch.words.to(dev), word_seq_lens = batch.word_seq_len.to(dev),
			context_emb=batch.context_emb.to(dev) if batch.context_emb is not None else None,
			chars = batch.chars.to(dev), char_seq_lens = batch.char_seq_lens.to(dev))

		batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts,f,batch_max_ids, batch.labels, batch.word_seq_len, idx2labels)
		p_dict += batch_p
		total_predict_dict += batch_predict
		total_entity_dict += batch_total
	
total_p = sum(list(p_dict.values()))
total_predict = sum(list(total_predict_dict.values()))
total_entity = sum(list(total_entity_dict.values()))

precision, recall, fscore = get_metric(total_p, total_entity, total_predict)

print(precision,recall,fscore)