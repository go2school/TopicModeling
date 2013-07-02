def pr(tp, fp, fn):
	if tp + fp != 0:
		return float(tp)/(tp+fp)
	else:
		return 0.0
		
def re(tp, fp, fn):
	if tp + fn != 0:
		return float(tp)/(tp+fn)
	else:
		return 0.0

def compute_pr_re_f1(data1, data2, all_labels):
	#compute label based macro and micro loss
	tp = {}
	fp = {}
	tn = {}
	fn = {}
	for l in all_labels:
		tp[l] = 0
		fp[l] = 0
		tn[l] = 0
		fn[l] = 0
		
	for i in range(len(data1)):
		labels1 = set(data1[i])
		labels2 = set(data2[i])
		for l in all_labels:
			if l in labels1 and l in labels2:
				tp[l] += 1
			elif l in labels1 and l not in labels2:
				fn[l] += 1
			elif l not in labels1 and l in labels2:
				fp[l] += 1
			elif l not in labels1 and l not in labels2:
				tn[l] += 1
	
	#compute macro F1
	macro_pr = 0
	macro_re = 0
	macro_f1 = 0
	for l in all_labels:
		macro_pr += pr(tp[l], fp[l], fn[l])
		macro_re += re(tp[l], fp[l], fn[l])
	macro_pr /= len(all_labels)
	macro_re /= len(all_labels)
	macro_f1 = 2 * macro_pr * macro_re / (macro_pr + macro_re)	
	
	#compute micro F1
	all_tp = 0
	all_fp = 0
	all_tn = 0
	all_fn = 0
	for l in all_labels:
		all_tp += tp[l]
		all_fp += fp[l]
		all_tn += tn[l]
		all_fn += fn[l]
	micro_pr = pr(all_tp, all_fp, all_fn)
	micro_re = re(all_tp, all_fp, all_fn)
	micro_f1 = 2 * micro_pr * micro_re / (micro_pr + micro_re)	
	return macro_f1, micro_f1

def usage():
	print "pyApp --indir XX --outdir YY --true_label ZZ --prefix MM"
		
if __name__ == '__main__': 	
	import sys, getopt	
	opts, args = getopt.getopt(sys.argv[1:], 'x', ['infile=', 'outdir=', 'prefix=', 'true_label='])
	infile = ''		
	outdir = ''	
	prefix = ''
	true_label = ''
	for o, a in opts:
		if o == "--infile":
			infile = a					
		elif o == "--outdir":
			outdir = a		
		elif o == "--prefix":
			prefix = a		
		elif o == "--true_label":
			true_label = a		
	if infile == '' or outdir == '' or prefix == '' or true_label == '':
		print usage()
		sys.exit(1)			

	#read doc IDs
	ids = []	
	fd = open(true_label)
	for line in fd:
		a = line.index(' ')
		ids.append(int(line[:a]))		
	fd.close()
	
	for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
		#read test labels
		test_labels = []
		test_file = infile
		fd = open(test_file)					
		for line in fd:			
			line = line.strip().split(' ')
			labels = line[1:]
			n_len = len(labels) / 2
			n_labels = []
			for i in range(0, n_len, 2):
				if float(labels[i+1]) >= th:
					n_labels.append(int(labels[i]))
			test_labels.append(n_labels)				
		fd.close()
		
		#lens not equal, quit 
		if len(test_labels) != len(ids):
			print 'length not equal'
			break			
		
		#write test labels
		fw = open(outdir + '/'+prefix+'_'+str(th)+'.txt', 'w')
		for i in range(len(test_labels)):
			v = test_labels[i]
			v.sort()
			sl = [str(l) for l in v]
			o_str = str(ids[i]) + ' ' + str(len(test_labels[i])) + ' ' + ' '.join(sl)
			fw.write(o_str + '\n')
		fw.close()
