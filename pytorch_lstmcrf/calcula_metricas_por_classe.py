#esse script recebe como entrada dois arquivos: O arquivo dos resultados
#e o arquivo de teste com as classes (nao apenas segmentando as entidades)

import sys

def read_file(file):
	l = []
	for line in file.readlines():
		if line != '\n':
			l.append(line)
	return l

#matriz de confusao pra cada classe
def init_mapp(classe,mapp):
	mapp[classe] = {
		'B':[[0,0],[0,0]],
		'I':[[0,0],[0,0]],
		'O':[[0,0],[0,0]],
		'size':0
	}

def sum_matrix(m1,m2):
	n = len(m1[0])
	m3 = [[0,0],[0,0]]
	for i in range(n):
		for j in range(n):
			m3[i][j] = m1[i][j] + m2[i][j]
	return m3

name_f1 = sys.argv[1]
name_f2 = sys.argv[2]
peso_por_tamanho = True

if name_f1 and name_f2:
	try:
		f_resultados = open(name_f1,'r')
		f_classes = open(name_f2,'r')
	except:
		print('entrada inválida')
		sys.exit()

lines_resultados = read_file(f_resultados)
lines_classes = read_file(f_classes)

print()
print('Linhas de cada Dataset')
print(len(lines_resultados),len(lines_classes))
print()

mapp = {}

for i in range(len(lines_resultados)):

	line_resultado = lines_resultados[i]
	line_classes = lines_classes[i]

	tag_result = line_resultado.split()[1]
	tag_teste = line_classes.split()[1]

	classe_atual = ''
	if tag_result != 'O':
		#tirando o '-'
		tag_result = line_resultado.split()[1][:-1]

	if tag_teste != 'O':
		tag_teste = line_classes.split()[1].split('-')[0]
		classe_atual = line_classes.split()[1].split('-')[1]
	else:
		classe_atual = 'FORA'
########################################

	if classe_atual not in mapp:
		init_mapp(classe_atual,mapp)
	else:
		mapp[classe_atual]['size'] += 1

	mb = mapp[classe_atual]['B']
	mi = mapp[classe_atual]['I']
	mo = mapp[classe_atual]['O']

	if tag_teste == tag_result:
		if tag_result == 'B':
			mb[0][0] += 1
			mi[1][1] += 1
			mo[1][1] += 1
		elif tag_result == 'I':
			mi[0][0] += 1
			mb[1][1] += 1
			mo[1][1] += 1
		elif tag_result == 'O':
			mo[0][0] += 1
			mb[1][1] += 1
			mi[1][1] += 1
	else:
		if tag_result == 'B' and tag_teste == 'I':
			mb[0][1] += 1
			mi[1][0] += 1
		elif tag_result == 'B' and tag_teste == 'O':
			mb[0][1] += 1
			mo[1][0] += 1
		elif tag_result == 'I' and tag_teste == 'B':
			mi[0][1] += 1
			mb[1][0] += 1
		elif tag_result == 'I' and tag_teste == 'O':
			mi[0][1] += 1
			mo[1][0] += 1
		elif tag_result == 'O' and tag_teste == 'B':
			mo[0][1] += 1
			mb[1][0] += 1
		elif tag_result == 'O' and tag_teste == 'I':
			mo[0][1] += 1
			mi[1][0] += 1

#para o caso do peso por tamanho da classe
sizes_all_classes = [mapp[classe]['size'] for classe in mapp]
total_size = sum(sizes_all_classes)

#MICRO AVERAGE PARA CADA CLASSE
precisao_media,recall_medio,f_measure_media = 0,0,0

for cont,classe in enumerate(mapp):

	if classe != 'ABSTRACAO':

		print('classe: '+ classe)

		matrizes = mapp[classe]
		m_aux = sum_matrix(matrizes['B'],matrizes['I'])
		MC = sum_matrix(m_aux,matrizes['O'])

		precisao = MC[0][0]/(MC[0][0]+MC[0][1])
		recall = MC[1][1]/(MC[1][1]+MC[1][0])
		f_measure = (2*precisao*recall)/(precisao+recall)

		precisao_media += precisao
		recall_medio += recall
		f_measure_media += f_measure
		
		print('precisao: ' + str(round(precisao,2)))
		print('recall: ' + str(round(recall,2)))
		print('f-measure: ' + str(round(f_measure,2)))
		print()

pm = round(precisao_media/(cont+1),2)
rm = round(recall_medio/(cont+1),2)
fm = round(f_measure_media/(cont+1),2)

print('Precisao Total: '+str(pm)) 
print('Recall Total: '+str(rm))
print('f_measure Total: '+str(fm)) 