def pr(tp, fp):
	if tp + fp == 0:
		return 0.0
	else:
		return float(tp)/(tp+fp)

def re(tp, fn):
	if tp + fn == 0:
		return 0.0
	else:
		return float(tp)/(tp+fn)
		
def f1(p, r):
	if p+r==0.0:
		return 0.0
	else:
		return 2*p*r/(p+r)
	
		
def read_label(fname):
	labels = []
	fd = open(fname)
	for line in fd:
		line = line.strip().split(' ')
		if int(line[1]) == 0:
			labels.append(set([]))
		else:
			ls = [int(l) for l in line[2:]]
			labels.append(set(ls))
	fd.close()
	return labels

def compute_label_based_f1(all_labels, true_labels, test_labels):	
	max_l = max(all_labels)
	tp = [0 for i in range(max_l+1)]
	fp = [0 for i in range(max_l+1)]
	fn = [0 for i in range(max_l+1)]

	sm_tp = 0
	sm_fp = 0
	sm_fn = 0
	for i in range(len(true_labels)):
		true_s = true_labels[i]
		test_s = test_labels[i]
		for l in all_labels:		
			if l in true_s and l in test_s:
				tp[l] += 1
			elif l in true_s and l not in test_s:
				fn[l] += 1
			elif l not in true_s and l in test_s:
				fp[l] += 1
				
	#compute macro
	mac_f1 = 0
	mac_pr = 0
	mac_re = 0
	for l in all_labels:
		p = pr(tp[l], fp[l])
		r = re(tp[l], fn[l])
		mac_f1 += f1(p, r)
		mac_pr += p
		mac_re += r
		sm_tp += tp[l]
		sm_fp += fp[l]
		sm_fn += fn[l]
	mac_pr = mac_pr/len(all_labels)
	mac_re = mac_re/len(all_labels)
	mac_f1 = mac_f1/len(all_labels)
	mi_p = pr(sm_tp, sm_fp)
	mi_r = re(sm_tp, sm_fn)
	mi_f1 = f1(mi_p, mi_r)
	return mi_p, mi_r, mi_f1, mac_pr, mac_re, mac_f1,

def compute_example_based_hier_f1(true_labels, predict_labels):
	sum_t = 0
	sum_p = 0
	sum_p_t = 0
	macro_pre = 0
	macro_rec = 0
	macro_f1s = 0
	for i in range(len(predict_labels)):
		t_ls = true_labels[i]
		p_ls = predict_labels[i]
		t = len(t_ls)
		p = len(p_ls)
		p_t = len(set(t_ls) & set(p_ls))
		if p == 0:
			pre = 0
		else:
			pre = float(p_t) / p
		if t == 0:
			rec = 0
		else:
			rec = float(p_t) / t
		if pre + rec == 0:
			f1s = 0
		else:
			f1s = 2 * pre * rec  / (pre+ rec)
		macro_pre += pre
		macro_rec += rec
		macro_f1s += f1s
		sum_t += t
		sum_p += p
		sum_p_t += p_t
	macro_pre /= len(predict_labels)
	macro_rec /= len(predict_labels)
	macro_f1s /= len(predict_labels)
	if sum_p == 0:
		pre = 0
	else:
		pre = float(sum_p_t) / sum_p
	if sum_t == 0:
		rec = 0
	else:
		rec = float(sum_p_t) / sum_t
	if pre + rec == 0:
		f1s = 0
	else:
		f1s = 2 * pre * rec  / (pre+ rec)
	return pre, rec, f1s, macro_pre, macro_rec, macro_f1s

"""
used_nodes = set()
used_nodes_fname = '/home/mpi/topic_model_svm/rcv1/rcv1_dynamic_used_nodes.txt'
fd = open(used_nodes_fname)
for line in fd:
	used_nodes.add(int(line.strip()))
fd.close()
"""

for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
	in_test_label = '/media/DataVolume1/py_code_topic_modeling/20_news/fold_0/loss/predict_labels_fold_0_'+str(th)+'.txt'
	in_true_label = '/media/DataVolume1/datasets/20news/new_dataset/5_folds/20_news_0_fold_test_label'

	test_labels = read_label(in_test_label)
	true_labels = read_label(in_true_label)

	all_labels = set()
	for t in true_labels:
		for tt in t:
			all_labels.add(tt)
	for t in test_labels:
		for tt in t:
			all_labels.add(tt)
			
	l1 = compute_label_based_f1(all_labels, true_labels, test_labels)
	print 'Labeled based', th, l1[2], l1[5]	
	l2 = compute_example_based_hier_f1(true_labels, test_labels)
	print 'Example based', th, l2[2], l2[5]
