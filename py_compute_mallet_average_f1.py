def make_state(data):
	import math
	min_auc = min(data)
	max_auc = max(data)
	mean_auc = float(sum(data))/len(data)
	if len(data) == 1:
		stdev_auc = 0
	else:
		tmp = [(d - mean_auc)*(d - mean_auc) for d in data]
		stdev_auc = math.sqrt(sum(tmp)/(len(tmp)-1))
	return min_auc, max_auc, mean_auc, stdev_auc
	
in_dir = 'D:/mallet-2.0.7/py_code_topic_modeling'
datasets = ['20_news', 'ohsumed', 'rcv1']
#dataset = '20_news'
#dataset = 'ohsumed'
dataset = 'rcv1'
for dataset in datasets:
	num_folds = 1
	print ''
	for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
		label_macro = [[], [], []]
		label_micro = [[], [], []]
		ex_macro = [[], [], []]
		ex_micro = [[], [], []]
		for i in range(num_folds):	
			#read label based loss	
			fd = open(in_dir + '/' + dataset + '/fold_'+str(i)+'/loss/' + 'global_label_based_loss_fold_'+str(i)+'_'+str(th)+'.txt')
			line = fd.read()
			line = line.strip().split(' ')
			line = [float(l) for l in line]	
			fd.close()
			
			label_micro[0].append(line[0])
			label_micro[1].append(line[1])
			label_micro[2].append(line[2])
			
			label_macro[0].append(line[3])		
			label_macro[1].append(line[4])		
			label_macro[2].append(line[5])
			
			#read example based loss	
			fd = open(in_dir + '/' + dataset + '/fold_'+str(i)+'/loss/' + 'hier_f_fold_'+str(i)+'_'+str(th)+'.txt')
			line = fd.read()
			line = line.strip().split(' ')
			line = [float(l) for l in line]	
			fd.close()
			
			ex_micro[0].append(line[0])		
			ex_micro[1].append(line[1])
			ex_micro[2].append(line[2])
			
			ex_macro[0].append(line[3])
			ex_macro[1].append(line[4])		
			ex_macro[2].append(line[5])
		label_macro_state = []
		label_micro_state = []
		for i in range(3):
			label_macro_state.append(make_state(label_macro[i]))
			label_micro_state.append(make_state(label_micro[i]))
		ex_macro_state = []
		ex_micro_state = []
		for i in range(3):
			ex_macro_state.append(make_state(ex_macro[i]))
			ex_micro_state.append(make_state(ex_micro[i]))	
		o_str = str(th) + '\t'
		o_str += 'Labeled based\t'
		o_str += dataset + '\t\t' + 'Macro '
		for i in range(3):
			o_str += ' ' + ('%.3f' % label_macro_state[i][2]) + '+-' +  ('%.3f' % label_macro_state[i][3])
		o_str += ' \t\tMicro '	
		for i in range(3):
			o_str += ' ' + ('%.3f' % label_micro_state[i][2]) + '+-' +  ('%.3f' % label_micro_state[i][3])	
		print o_str
		o_str = str(th) + '\t'
		o_str += 'Example based\t'
		o_str += dataset + '\t\t' + 'Macro '
		for i in range(3):
			o_str += ' ' + ('%.3f' % ex_macro_state[i][2]) + '+-' +  ('%.3f' % ex_macro_state[i][3])
		o_str += ' \t\tMicro '	
		for i in range(3):
			o_str += ' ' + ('%.3f' % ex_micro_state[i][2]) + '+-' +  ('%.3f' % ex_micro_state[i][3])
		print o_str

