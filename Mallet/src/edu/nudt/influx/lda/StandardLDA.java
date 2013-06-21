package edu.nudt.influx.lda;
/* 
 * changed by Xiao 11182012 for saving inference model
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import cc.mallet.topics.tui.InferTopics;
import cc.mallet.topics.tui.Vectors2Topics;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;
import edu.nudt.influx.lda.InferLDATopicsGibbs.WordProb;

//import edu.nudt.lab613.SNA.tc.infoDiffusion.DocTopic;

public class StandardLDA implements Serializable {

	int numTopics; // Number of topics to be fit
	public double alpha; // Dirichlet(alpha,alpha,...) is the distribution over
							// topics
	public double beta; // Prior on per-topic multinomial distribution over
						// words
	double tAlpha;// sum of alpha
	double vBeta;//sum of beta
	InstanceList ilist; // the data field of the instances is expected to hold a
						// FeatureSequence
	int[][] topics; // indexed by <document index, sequence index>
	public int numTypes;// number of distinguish words over all documents
	int numTokens;// number of words over all documents including duplicate
	int numDocs;
	// words
	int[][] docTopicCounts; // indexed by <document index, topic index>
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>

	int numTopWords = 50;
	
	public StandardLDA(int numberOfTopics) {
		this(numberOfTopics, 50.0, 0.01);
	}

	public StandardLDA(int numberOfTopics, double alphaSum, double beta) {
		this.numTopics = numberOfTopics;
		this.alpha = alphaSum / numTopics;
		this.beta = beta;
	}

	public void estimate(InstanceList documents, int numIterations,
			int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r) {
		ilist = documents.shallowClone();
		numTypes = ilist.getDataAlphabet().size();// get the distinctive words
													// in all documents
		numDocs = ilist.size();// get number of documents
		topics = new int[numDocs][];
		docTopicCounts = new int[numDocs][numTopics];
		typeTopicCounts = new int[numTypes][numTopics];
		tokensPerTopic = new int[numTopics];
		tAlpha = alpha * numTopics;
		vBeta = beta * numTypes;

		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen;
		FeatureSequence fs;
		for (int di = 0; di < numDocs; di++) {
			try {
				fs = (FeatureSequence) ilist.get(di).getData();
			} catch (ClassCastException e) {
				System.err
						.println("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
								+ "With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
				throw e;
			}
			seqLen = fs.getLength();
			numTokens += seqLen;
			topics[di] = new int[seqLen];
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				topic = r.nextInt(numTopics);
				topics[di][si] = topic;// randomly assign topic to every words
										// in document di
				docTopicCounts[di][topic]++;
				typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				tokensPerTopic[topic]++;
			}
		}

		this.estimate(0, numDocs, numIterations, showTopicsInterval,
				outputModelInterval, outputModelFilename, r);

	}

	public void addDocuments(InstanceList additionalDocuments,
			int numIterations, int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r, double w) {
		if (ilist == null)
			throw new IllegalStateException(
					"Must already have some documents first.");

		ilist = additionalDocuments;
		numTypes = additionalDocuments.getDataAlphabet().size();
		vBeta = beta * numTypes;
		int numToken = numTokens;
		numTokens = 0;

		int numDocs = additionalDocuments.size();
		int[][] newTopics = new int[numDocs][];
		topics = newTopics;
		int[][] newDocTopicCounts = new int[numDocs][numTopics];
		docTopicCounts = newDocTopicCounts;
		int[][] newTypeTopicCounts = new int[numTypes][numTopics];
		FeatureSequence fs1;
		for (int di = 0; di < topics.length; di++) {
			fs1 = (FeatureSequence) additionalDocuments.get(di).getData();
			int seqLen = fs1.getLength();
			numTokens += seqLen;
		}
		/*
		 * if(numToken<(numTokens*w)) { w=(numTokens*w)/numToken; } else
		 * if(numToken>(numTokens/w)) { w=numTokens/(numToken*w); }
		 */
		w = (numTokens * w) / numToken;
		int sum = 0;
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < typeTopicCounts.length; j++) {
				newTypeTopicCounts[j][i] = (int) Math
						.round(typeTopicCounts[j][i] * w);
				sum = sum + newTypeTopicCounts[j][i];
			}
			tokensPerTopic[i] = sum;
			sum = 0;
		}
		typeTopicCounts = newTypeTopicCounts;
		numTokens = 0;
		FeatureSequence fs;
		for (int di = 0; di < topics.length; di++) {
			fs = (FeatureSequence) additionalDocuments.get(di).getData();
			int seqLen = fs.getLength();
			numTokens += seqLen;
			topics[di] = new int[seqLen];
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				int topic = r.nextInt(numTopics);
				topics[di][si] = topic;
				docTopicCounts[di][topic]++;
				typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				tokensPerTopic[topic]++;
			}
		}
		this.estimate(0, numDocs, numIterations, showTopicsInterval,
				outputModelInterval, outputModelFilename, r);
	}

	/*
	 * Perform several rounds of Gibbs sampling on the documents in the given
	 * range.
	 */
	public void estimate(int docIndexStart, int docIndexLength,
			int numIterations, int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r) {
		for (int iterations = 0; iterations < numIterations; iterations++) {
			if (iterations % 10 == 0)
				System.out.print(iterations);
			else
				System.out.print(".");
			System.out.flush();
			if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0
					&& iterations > 0) {
				System.out.println();
				
				//printTopWords(5, false);
				printToFile(true, new File(para_topwordsFile.value + "_" + iterations));
			}
			if (outputModelInterval != 0
					&& iterations % outputModelInterval == 0 && iterations > 0) {
				this.write(new File(outputModelFilename + '.' + iterations));
			}
			sampleTopicsForDocs(docIndexStart, docIndexLength, r);
		}

	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForDocs(int start, int length, Randoms r) {
		assert (start + length <= docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start + length; di++) {
			sampleTopicsForOneDoc((FeatureSequence) ilist.get(di).getData(),
					topics[di], docTopicCounts[di], topicWeights, r);
		}
	}

	private void sampleTopicsForOneDoc(FeatureSequence oneDocTokens,
			int[] oneDocTopics, // indexed by seq position
			int[] oneDocTopicCounts, // indexed by topic index
			double[] topicWeights, Randoms r) {
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			type = oneDocTokens.getIndexAtPosition(si);
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
			oneDocTopicCounts[oldTopic]--;
			typeTopicCounts[type][oldTopic]--;
			tokensPerTopic[oldTopic]--;
			// Build a distribution over topics for this token
			Arrays.fill(topicWeights, 0.0);
			topicWeightsSum = 0;
			currentTypeTopicCounts = typeTopicCounts[type];
			for (int ti = 0; ti < numTopics; ti++) {
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
						* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha);
																// is constant
																// across all
																// topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;			
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete(topicWeights, topicWeightsSum);			
			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
			oneDocTopicCounts[newTopic]++;
			typeTopicCounts[type][newTopic]++;
			tokensPerTopic[newTopic]++;
		}
	}

	public Alphabet megdict(InstanceList ilist1, InstanceList ilist2) {// Alphabet
																		// datadic=null;
																		// boolean
																		// flag=false;
		Alphabet datadic1 = ilist1.getAlphabet();
		System.out.print(datadic1.size() + "\n");
		Alphabet datadic2 = ilist2.getAlphabet();
		System.out.print(datadic2.size() + "\n");
		for (int i = 0; i < datadic2.size(); i++) {
			datadic1.lookupIndex(datadic2.lookupObject(i).toString());
			// System.out.print(datadic1.size());
		}

		System.out.print(datadic1.size() + "\n");
		return datadic1;
	}

	public int[][] getDocTopicCounts() {
		return docTopicCounts;
	}

	public int[][] getTypeTopicCounts() {
		return typeTopicCounts;
	}

	public int[] getTokensPerTopic() {
		return tokensPerTopic;
	}
	
	public int[][] getTopics(){
		return this.topics;
	}

	public void printTopWords(int numWords, boolean useNewLines) {
		class WordProb implements Comparable {
			int wi;
			double p;

			public WordProb(int wi, double p) {
				this.wi = wi;
				this.p = p;
			}

			public final int compareTo(Object o2) {
				if (p > ((WordProb) o2).p)
					return -1;
				else if (p == ((WordProb) o2).p)
					return 0;
				else
					return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb(wi,
						(double) (typeTopicCounts[wi][ti] + beta)
								/ (tokensPerTopic[ti] + vBeta));
			Arrays.sort(wp);
			if (useNewLines) {
				System.out.println("\nTopic " + ti);
				for (int i = 0; i < numWords; i++)
					System.out.println(ilist.getDataAlphabet()
							.lookupObject(wp[i].wi).toString()
							+ "\t" + wp[i].p);
				// System.out.println
				// (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString());
			} else {
				System.out.print("Topic " + ti + ": ");
				for (int i = 0; i < numWords; i++)
					System.out.print(ilist.getDataAlphabet()
							.lookupObject(wp[i].wi).toString()
							+ "\t");
				System.out.println();
			}
		}
	}

	public void printToFile(boolean useNewLines, File file) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(file));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		printToFile(numTopWords, useNewLines, pw);
		pw.close();
	}

	public void printToFile(int numWords, boolean useNewLines, PrintWriter out) {
		class WordProb implements Comparable {
			int wi;
			double p;

			public WordProb(int wi, double p) {
				this.wi = wi;
				this.p = p;
			}

			public final int compareTo(Object o2) {
				if (p > ((WordProb) o2).p)
					return -1;
				else if (p == ((WordProb) o2).p)
					return 0;
				else
					return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb(wi, ((double) typeTopicCounts[wi][ti])
						/ tokensPerTopic[ti]);
			Arrays.sort(wp);
			if (useNewLines) {
				out.println("\nTopic " + ti);
				for (int i = 0; i < numWords; i++)
					out.println(ilist.getDataAlphabet().lookupObject(wp[i].wi)
							.toString()
							+ " " + wp[i].p);
			} else {
				out.print("Topic " + ti + ": ");
				for (int i = 0; i < numWords; i++)
					out.print(ilist.getDataAlphabet().lookupObject(wp[i].wi)
							.toString()
							+ " ");
				out.println();
			}
		}
	}

	public void SprintToFile(int numWords, boolean useNewLines, File file) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(file));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		SprintToFile(numWords, useNewLines, pw);
		pw.close();
	}

	public void SprintToFile(int numWords, boolean useNewLines, PrintWriter out) {
		class WordProb implements Comparable {
			int wi;
			double p;

			public WordProb(int wi, double p) {
				this.wi = wi;
				this.p = p;
			}

			public final int compareTo(Object o2) {
				if (p > ((WordProb) o2).p)
					return -1;
				else if (p == ((WordProb) o2).p)
					return 0;
				else
					return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb(wi,
						(double) (typeTopicCounts[wi][ti] + beta)
								/ (tokensPerTopic[ti] + vBeta));
			Arrays.sort(wp);
			if (useNewLines) {
				out.println("\nTopic " + ti);
				for (int i = 0; i < numWords; i++)
					out.println(ilist.getDataAlphabet().lookupObject(wp[i].wi)
							.toString()
							+ "\t" + wp[i].p);
			} else {
				out.print("Topic " + ti + ": ");
				for (int i = 0; i < numWords; i++)
					out.print(ilist.getDataAlphabet().lookupObject(wp[i].wi)
							.toString()
							+ "\t");
				out.println();
			}
		}
	}

	public void printDocumentTopics(File f) throws IOException {
		printDocumentTopics(new PrintWriter(new FileWriter(f)));
	}

	public void printDocumentTopics(PrintWriter pw) {
		printDocumentTopics(pw, 0.0, -1);
		pw.close();
	}

	public void printDocumentTopics(PrintWriter pw, double threshold, int max) {
		pw.println("#doc source topic proportion ...");
		int docLen;
		double topicDist[] = new double[numTopics];
		for (int di = 0; di < topics.length; di++) {
			pw.print(di);
			pw.print(' ');
			if (ilist.get(di).getSource() != null) {
				pw.print(ilist.get(di).getSource().toString());
			} else {
				pw.print("null-source");
			}
			pw.print(' ');
			docLen = topics[di].length;
			for (int ti = 0; ti < numTopics; ti++)
				topicDist[ti] = (((float) docTopicCounts[di][ti]) / docLen);
			if (max < 0)
				max = numTopics;
			for (int tp = 0; tp < max; tp++) {
				double maxvalue = 0;
				int maxindex = -1;
				for (int ti = 0; ti < numTopics; ti++)
					if (topicDist[ti] > maxvalue) {
						maxvalue = topicDist[ti];
						maxindex = ti;
					}
				if (maxindex == -1 || topicDist[maxindex] < threshold)
					break;
				pw.print(maxindex + " " + topicDist[maxindex] + " ");
				topicDist[maxindex] = 0;
			}
			pw.println(' ');
		}
	}

	public void SprintDocumentTopics(File f) throws IOException {
		SprintDocumentTopics(new PrintWriter(new FileWriter(f)));
	}

	public void SprintDocumentTopics(PrintWriter pw) {
		SprintDocumentTopics(pw, 0.0, -1);
		pw.close();
	}

	public void SprintDocumentTopics(PrintWriter pw, double threshold, int max) {
		pw.println("#doc source topic proportion ...");
		int docLen;
		double topicDist[] = new double[numTopics];
		for (int di = 0; di < topics.length; di++) {
			pw.print(di);
			pw.print('\t');
			if (ilist.get(di).getSource() != null) {
				pw.print(ilist.get(di).getSource().toString());
			} else {
				pw.print("null-source");
			}
			pw.print('\t');
			docLen = topics[di].length;
			for (int ti = 0; ti < numTopics; ti++)
				topicDist[ti] = ((double) (docTopicCounts[di][ti] + alpha) / (docLen + tAlpha));
			// 排序
			if (max < 0)
				max = numTopics;
			for (int tp = 0; tp < max; tp++) {
				double maxvalue = 0;
				int maxindex = -1;
				for (int ti = 0; ti < numTopics; ti++)
					if (topicDist[ti] > maxvalue) {
						maxvalue = topicDist[ti];
						maxindex = ti;
					}
				if (maxindex == -1 || topicDist[maxindex] < threshold)
					break;
				pw.print(maxindex + "\t" + topicDist[maxindex] + "\t");
				topicDist[maxindex] = 0;
			}
			pw.println('\t');
		}
	}
/*
	// 计算各文件的话题分布
	public ArrayList<DocTopic> ChannalTopic() {

		ArrayList<DocTopic> docTopicList = new ArrayList<DocTopic>();
		int docLen;
		String[] fileName = new String[topics.length];
		double topicDist[] = new double[numTopics];
		for (int di = 0; di < topics.length; di++) {
			if (ilist.get(di).getSource() != null) {
				// 记录是哪个文件
				fileName[di] = ilist.get(di).getSource().toString();
			} else
				System.out.println("数据在ChannalTopic函数不存在！");

			docLen = topics[di].length;
			for (int ti = 0; ti < numTopics; ti++)
				topicDist[ti] = ((double) (docTopicCounts[di][ti] + alpha) / (docLen + tAlpha));

			DocTopic newDocTopic = new DocTopic();
			newDocTopic.fileName = fileName[di];
			for (int j = 0; j < topicDist.length; j++)
				newDocTopic.topicDist.add(topicDist[j]);
			docTopicList.add(newDocTopic);

		}

		return docTopicList;
	}
*/
	public void printTopicDocument(File pf) throws FileNotFoundException {
		FileOutputStream ou = new FileOutputStream(pf);
		PrintStream p = new PrintStream(ou);
		double[][] topicdoc = new double[numTopics][topics.length];
		double[] avg = new double[numTopics];
		for (int ti = 0; ti < numTopics; ti++) {
			avg[ti] = 0;
			for (int di = 0; di < topics.length; di++) {
				topicdoc[ti][di] = ((double) (docTopicCounts[di][ti] + alpha) / (topics[di].length + tAlpha));
				avg[ti] = avg[ti] + topicdoc[ti][di];
			}
			avg[ti] = avg[ti] / (topics.length);
		}

		for (int i = 0; i < numTopics; i++)
			p.println(avg[i]);
		p.close();

	}

	public void printTopicDocuments(File pf) throws FileNotFoundException {
		class WordProb implements Comparable {
			int wi;
			double p;

			public WordProb(int wi, double p) {
				this.wi = wi;
				this.p = p;
			}

			public final int compareTo(Object o2) {
				if (p > ((WordProb) o2).p)
					return -1;
				else if (p == ((WordProb) o2).p)
					return 0;
				else
					return 1;
			}
		}

		int doc = (topics.length > 10) ? 10 : topics.length;
		WordProb[][] wp = new WordProb[numTopics][topics.length];
		FileOutputStream ou = new FileOutputStream(pf);
		PrintStream p = new PrintStream(ou);
		for (int ti = 0; ti < numTopics; ti++) {
			for (int di = 0; di < topics.length; di++) {
				wp[ti][di] = new WordProb(
						di,
						((double) (docTopicCounts[di][ti] + alpha) / (topics[di].length + tAlpha)));
			}
			Arrays.sort(wp[ti]);
		}
		for (int ti = 0; ti < numTopics; ti++) {
			p.println("Topic " + ti);
			for (int di = 0; di < doc; di++) {
				p.println(ilist.get(wp[ti][di].wi).getSource().toString()
						+ "\t" + wp[ti][di].p);
			}
		}

		p.close();

	}

	public void printState(File f) throws IOException {
		PrintWriter writer = new PrintWriter(new FileWriter(f));
		printState(writer);
		writer.close();
	}

	public void printState(PrintWriter pw) {
		Alphabet a = ilist.getDataAlphabet();
		pw.println("#doc pos typeindex type topic");
		for (int di = 0; di < topics.length; di++) {
			FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
			for (int si = 0; si < topics[di].length; si++) {
				int type = fs.getIndexAtPosition(si);
				pw.print(di);
				pw.print(' ');
				pw.print(si);
				pw.print(' ');
				pw.print(type);
				pw.print(' ');
				pw.print(a.lookupObject(type));
				pw.print(' ');
				pw.print(topics[di][si]);
				pw.println();
			}
		}
	}

	public void write(File f) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(
					new FileOutputStream(f));
			oos.writeObject(this);
			oos.close();
		} catch (IOException e) {
			System.err.println("Exception writing file " + f + ": " + e);
		}
	}

	// Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;

	private void writeObject(ObjectOutputStream out) throws IOException {
		out.writeInt(CURRENT_SERIAL_VERSION);
		out.writeObject(ilist);
		out.writeInt(numTopics);
		out.writeDouble(alpha);
		out.writeDouble(beta);
		out.writeDouble(tAlpha);
		out.writeDouble(vBeta);
		for (int di = 0; di < topics.length; di++)
			for (int si = 0; si < topics[di].length; si++)
				out.writeInt(topics[di][si]);
		for (int di = 0; di < topics.length; di++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt(docTopicCounts[di][ti]);
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt(typeTopicCounts[fi][ti]);
		for (int ti = 0; ti < numTopics; ti++)
			out.writeInt(tokensPerTopic[ti]);
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		int featuresLength;
		int version = in.readInt();
		ilist = (InstanceList) in.readObject();
		numTopics = in.readInt();
		alpha = in.readDouble();
		beta = in.readDouble();
		tAlpha = in.readDouble();
		vBeta = in.readDouble();
		int numDocs = ilist.size();
		topics = new int[numDocs][];
		for (int di = 0; di < ilist.size(); di++) {
			int docLen = ((FeatureSequence) ilist.get(di).getData())
					.getLength();
			topics[di] = new int[docLen];
			for (int si = 0; si < docLen; si++)
				topics[di][si] = in.readInt();
		}
		docTopicCounts = new int[numDocs][numTopics];
		for (int di = 0; di < ilist.size(); di++)
			for (int ti = 0; ti < numTopics; ti++)
				docTopicCounts[di][ti] = in.readInt();
		int numTypes = ilist.getDataAlphabet().size();
		typeTopicCounts = new int[numTypes][numTopics];
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				typeTopicCounts[fi][ti] = in.readInt();
		tokensPerTopic = new int[numTopics];
		for (int ti = 0; ti < numTopics; ti++)
			tokensPerTopic[ti] = in.readInt();
	}

	public void PrintString(String out, File pf) {

		FileOutputStream ou = null;
		try {
			ou = new FileOutputStream(pf);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		PrintStream p = new PrintStream(ou);
		p.print(out);

	}

	public InstanceList getInstanceList() {
		return ilist;
	}
	
	public void printSampleTopics(StandardLDA lda,String sampleTopicLogFile){
		int[][] topics=lda.getTopics();
		int doc_len = topics.length;
		try {
			FileWriter fw = new FileWriter(sampleTopicLogFile,true);
			BufferedWriter bw = new BufferedWriter(fw);
			for(int i=0;i<doc_len;i++){
				int[] tokens = topics[i];
				int token_len = tokens.length;
				for(int j=0;j<token_len-2;j++){
					bw.write(ilist.get(i).getDataAlphabet().lookupObject(j)+": "+topics[i][j]+"\t\t");
					if(j%5==0&&j!=0){
						bw.newLine();
					}
				}
				bw.newLine();
				bw.newLine();
			}
			bw.flush();
			bw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/** Return a tool for estimating topic distributions for new documents */
	public InferLDATopicsGibbs getInferencer() {
		double [] alpha_lst = new double [numTopics];
		for(int i=0;i<numTopics;i++)
			alpha_lst[i] = alpha; 
		return new InferLDATopicsGibbs(typeTopicCounts, tokensPerTopic,
								   ilist.getDataAlphabet(),
								   alpha_lst, beta, vBeta);
	}
	
	public static void usage()
	{
		System.out.println("CMD");
	}
		
	/*
	 * save P(topic|doc)
	 */
	public void save_theta(String fname, int withPrior) throws IOException {
		class WordProb implements Comparable {
			int wi;
			double p;

			public WordProb(int wi, double p) {
				this.wi = wi;
				this.p = p;
			}

			public final int compareTo(Object o2) {
				if (p > ((WordProb) o2).p)
					return -1;
				else if (p == ((WordProb) o2).p)
					return 0;
				else
					return 1;
			}
		}
		
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		int numDocs = ilist.size();
	    for (int m = 0; m < numDocs; m++) {	    	
	    	FeatureSequence fs = (FeatureSequence) ilist.get(m).getData();
	    	ArrayList<WordProb> lst = new ArrayList<WordProb>();
	    	for (int k = 0; k < numTopics; k++) {
	    		if(withPrior == 1)
			    {
	    			double v = (double)(docTopicCounts[m][k] + alpha) / (fs.getLength() + numTopics * alpha);	
	    			if(v != 0)
	    				lst.add(new WordProb(k, v));
			    }
	    		else
	    		{
	    			double v = (double)(docTopicCounts[m][k]) / (fs.getLength());			    	    		
	    			if(v != 0)
	    				lst.add(new WordProb(k, v));
	    		}
			}
	    	Collections.sort(lst);
	    	for(int i=0;i<lst.size();i++)
	    	{	    		
	    		String str = lst.get(i).wi + ":" + lst.get(i).p + " ";
	    		pw.print(str);
	    	}
			pw.print('\n');			
	    }
	    pw.flush();
	    pw.close();
	}
	
	/*
	 * save P(word|topic)
	 */
	public void save_phi(String fname, int withPrior) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(fname));		
	    for (int k = 0; k < numTopics; k++) {	
	    	pw.println("Topic " + k);
	    	if(withPrior == 1)
	    	{
				for (int n = 0; n < numTypes; n++) {			    
				    double v = (double)(typeTopicCounts[n][k] + beta) / (tokensPerTopic[k] + numTypes * beta);				  
				    if(v != 0)
				    {
				    	String str = n + ":" + String.format("%.4f",v) + " ";
				    	pw.print(str);
				    }
				}
	    	}
	    	else
	    	{
	    		for (int n = 0; n < numTypes; n++) {			    
				    double v = (double)typeTopicCounts[n][k] / tokensPerTopic[k];
				    if(v != 0)
				    {
				    	String str = n + ":" + String.format("%.4f",v) + " ";
				    	pw.print(str);
				    }
				}
	    	}
			pw.print('\n');
	    }
	    pw.flush();
	    pw.close();
	}
	
	public void printTopicPerWord(String fname)throws IOException 
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
	    for (int m = 0; m < numDocs; m++) {
	    	FeatureSequence fs = (FeatureSequence) ilist.get(m).getData();
	    	for (int si = 0; si < topics[m].length; si++) {
				int type = fs.getIndexAtPosition(si);
				String word = ilist.getDataAlphabet().lookupObject(type).toString();
			    pw.print(word+":"+topics[m][si] + " ");
			}
			pw.print('\n');
	    }
	    pw.flush();
	    pw.close();
	}

	 static CommandOption.String para_inferencerFilename = new CommandOption.String
    (InferTopics.class, "inferencer", "FILENAME", true, null,
		 "A serialized topic inferencer from a trained topic model.\n", null);

	static CommandOption.String para_inputFile = new CommandOption.String
		(InferTopics.class, "input", "FILENAME", true, null,
		 "The filename from which to read the list of instances\n" +
		 "for which topics should be inferred.  Use - for stdin.  " +
		 "The instances must be FeatureSequence or FeatureSequenceWithBigrams, not FeatureVector", null);
	
	static CommandOption.String para_topwordsFile = new CommandOption.String
	(InferTopics.class, "topwords", "FILENAME", true, null,
	 "The filename from which to show the top words for each topic", null);
	
	static CommandOption.String para_thetaFile = new CommandOption.String
	(InferTopics.class, "thetafname", "FILENAME", true, null,
	 "The filename from which to save the topic distribution per docs", null);
	
	static CommandOption.Integer para_numIterations = new CommandOption.Integer
    (InferTopics.class, "num-iterations", "INTEGER", true, 100,
     "The number of iterations of Gibbs sampling.", null);
	
	static CommandOption.Integer para_numTopics = new CommandOption.Integer
    (InferTopics.class, "num-topics", "INTEGER", true, 50,
     "The number of topics.", null);
	
	static CommandOption.Integer para_showTopicsInterval = new CommandOption.Integer
	(Vectors2Topics.class, "show-topics-interval", "INTEGER", true, 0,
	 "The number of iterations between printing a brief summary of the topics so far.", null);

	static CommandOption.Integer para_outputModelInterval = new CommandOption.Integer
		(Vectors2Topics.class, "output-model-interval", "INTEGER", true, 0,
		 "The number of iterations between writing the model (and its Gibbs sampling state) to a binary file.  " +
		 "You must also set the --output-model to use this option, whose argument will be the prefix of the filenames.", null);
	
	static CommandOption.Integer para_outputStateInterval = new CommandOption.Integer
	    (Vectors2Topics.class, "output-state-interval", "INTEGER", true, 0,
	     "The number of iterations between writing the sampling state to a text file.  " +
	     "You must also set the --output-state to use this option, whose argument will be the prefix of the filenames.", null);
	
	static CommandOption.Double para_alpha = new CommandOption.Double
		(Vectors2Topics.class, "alpha", "DECIMAL", true, 50.0,
		 "Alpha parameter: smoothing over topic distribution.",null);
	
	static CommandOption.Double para_beta = new CommandOption.Double
		(Vectors2Topics.class, "beta", "DECIMAL", true, 0.01,
		 "Beta parameter: smoothing over unigram distribution.",null);

	static CommandOption.String docWordTopicFile = new CommandOption.String
    (InferTopics.class, "output-doc-word-topic", "FILENAME", true, null,
     "The filename in which to write the inferred topic\n" +
		 "proportions per word position per document.  " +
     "By default this is null, indicating that no file will be written.", null);
	
	static CommandOption.String docTopicsFile = new CommandOption.String
    (InferTopics.class, "output-doc-topics", "FILENAME", true, null,
     "The filename in which to write the inferred topic\n" +
		 "proportions per document.  " +
     "By default this is null, indicating that no file will be written.", null);
	
	static CommandOption.String topicWordFile = new CommandOption.String
    (InferTopics.class, "output-topic-word", "FILENAME", true, null,
     "The filename in which to write the inferred word\n" +
		 "proportions per topic.  " +
     "By default this is null, indicating that no file will be written.", null);
	
	public static void main(String[] args) throws IOException {
		
		/*
		String train_fname = "/home/xiao/test_mallet/topic_input.mallet"; 
		String inferencerFilename = "/home/xiao/test_mallet/topic_inference.mallet";
		String theta_fname = "/home/xiao/test_mallet/topic_theta.mallet";
		//String phi_fname = "/home/xiao/test_mallet/topic_phi.mallet";
		String topwords_per_topic_fname = "/home/xiao/test_mallet/topwords.mallet";
		//String doc_topic_fname = "/home/xiao/test_mallet/doctopis.mallet";
		*/
		
		// Process the command-line options                                                                           
		CommandOption.setSummary (InferTopics.class,
                                  "Use an existing topic model to infer topic distributions for new documents");
        CommandOption.process (InferTopics.class, args);
        
        if (para_inferencerFilename.value == null) {
			System.err.println("You must specify a serialized topic inferencer. Use --help to list options.");
			System.exit(0);
		}

		if (para_inputFile.value == null) {
			System.err.println("You must specify a serialized instance list. Use --help to list options.");
			System.exit(0);
		}
				
		InstanceList ilist = InstanceList.load(new File(para_inputFile.value));

		int numIterations = para_numIterations.value;		
		int numTopics = para_numTopics.value;
		System.out.println("Data loaded.");		

		StandardLDA lda = new StandardLDA(numTopics, para_alpha.value, para_beta.value);
		
		lda.numTopWords = 50;
		
		lda.estimate(ilist, numIterations, 50, para_outputModelInterval.value, null, new Randoms()); 
				
		lda.printToFile(true, new File(para_topwordsFile.value));
		//lda.printDocumentTopics(new File(doc_topic_fname));		
		lda.save_theta(para_thetaFile.value, 0);//p(topic|doc)
		lda.save_phi(topicWordFile.value, 0);//p(word|topic)
		//lda.save_phi(phi_fname);
		
		//save model for inference
		try {

			ObjectOutputStream oos = 
				new ObjectOutputStream(new FileOutputStream(para_inferencerFilename.value));
			oos.writeObject(lda.getInferencer());
			oos.close();

		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		
		//inferencer.printDocumentTopics(docTopicsFile2);
		lda.printTopicPerWord(docWordTopicFile.value);
		
		System.out.println("End learning");
	}
}
