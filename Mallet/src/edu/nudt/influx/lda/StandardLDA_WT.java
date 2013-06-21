package edu.nudt.influx.lda;
/*
 * 初始的LDA实现,由梁政实现,
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
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;
import edu.nudt.influx.Utility.HashProcessor;

//import edu.nudt.lab613.SNA.tc.infoDiffusion.DocTopic;

public class StandardLDA_WT implements Serializable {

	int numTopics; // Number of topics to be fit
	public double alpha; // Dirichlet(alpha,alpha,...) is the distribution over
							// topics
	public double beta; // Prior on per-topic multinomial distribution over
						// words
	double tAlpha;
	double vBeta;
	InstanceList ilist; // the data field of the instances is expected to hold a
						// FeatureSequence
	int[][] topics; // indexed by <document index, sequence index>
	public int numTypes;// number of distinguish words over all documents
	int numTokens;// number of words over all documents including duplicate
					// words
//	int[][] docTopicCounts; // indexed by <document index, topic index>
//	int[][] typeTopicCounts; // indexed by <feature index, topic index>
//	int[] tokensPerTopic; // indexed by <topic index>
	
    public HashProcessor hp = new HashProcessor();
	
	HashMap<Integer,Integer>[] docTopicCounts; // indexed by <document index, topic index>
	HashMap<Integer,Integer>[] docTopicCountsSamples;
	
	HashMap<Integer,Integer>[] typeTopicCounts; // indexed by <feature index, topic index>
	HashMap<Integer,Double>[] typeTopicCountsSamples;
	HashMap<Integer,Double>[] typeTopicCountsAverage;
	
	int[] tokensPerTopic; // indexed by <topic index>
	int[] tokensPerTopicSamples;
	
//	int numIterations;
//	int burnIn;

	public StandardLDA_WT(int numberOfTopics) {
		this(numberOfTopics, 50.0, 0.01);
	}

	public StandardLDA_WT(int numberOfTopics, double alphaSum, double beta) {
		this.numTopics = numberOfTopics;
		this.alpha = alphaSum / numTopics;
		this.beta = beta;
	}

	public void estimate(InstanceList documents,int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r,int burnIn, int numIterations) {
		ilist = documents.shallowClone();
		numTypes = ilist.getDataAlphabet().size();// get the distinctive words
													// in all documents
		int numDocs = ilist.size();// get number of documents
		topics = new int[numDocs][];
//		docTopicCounts = new int[numDocs][numTopics];
//		typeTopicCounts = new int[numTypes][numTopics];
		docTopicCounts = new HashMap[numDocs];
		docTopicCountsSamples = new HashMap[numDocs];	
		hp.initial(docTopicCounts, numDocs);
		hp.initial(docTopicCountsSamples, numDocs);
		
		typeTopicCounts = new HashMap[numTypes];
		typeTopicCountsSamples = new HashMap[numTypes];
		hp.initial(typeTopicCounts, numTypes);
		hp.initial(typeTopicCountsSamples, numTypes);
		
		tokensPerTopic = new int[numTopics];
		tokensPerTopicSamples = new int[numTopics];
		
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
//				docTopicCounts[di][topic]++;
//				typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				hp.addInteger(docTopicCounts[di], topic);
				hp.addInteger(typeTopicCounts[fs.getIndexAtPosition(si)], topic);
				tokensPerTopic[topic]++;
			}
		}

		this.estimate(0, numDocs, showTopicsInterval, outputModelInterval, outputModelFilename, r,burnIn, numIterations);

	}
	
	public void Inferrence(InstanceList additionalDocuments,int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r,int burnIn,int numIterations) {
		if (ilist == null)
			throw new IllegalStateException(
					"Must already have some documents first.");

//		ilist = additionalDocuments;
		int original_numTypes = numTypes; 
		numTypes = additionalDocuments.getDataAlphabet().size();
		vBeta = beta * numTypes;
		int numToken = numTokens;
		numTokens = 0;

		int numDocs = additionalDocuments.size();
		topics = new int[numDocs][];

		// int[][] newDocTopicCounts = new int[numDocs][numTopics];
		// docTopicCounts = newDocTopicCounts;
		// int[][] newTypeTopicCounts = new int[numTypes][numTopics];
		
		//initial the typeTopicCounts
		HashMap<Integer, Integer>[] newTypeTopicCounts = new HashMap[numTypes];
        hp.initial(newTypeTopicCounts, numTypes);
        for(int i=0;i<numTypes;i++){//make judge to see if the new term exist in training dataset
        	String new_term = additionalDocuments.getDataAlphabet().lookupObject(i).toString();
        	int new_term_old_index = ilist.getDataAlphabet().lookupIndex(new_term);
        	if(new_term_old_index<original_numTypes){//exist
        		newTypeTopicCounts[i]=(HashMap<Integer, Integer>) typeTopicCounts[new_term_old_index].clone();
        	}
        }
        typeTopicCounts=newTypeTopicCounts;
        
        tokensPerTopicSamples = new int [numTopics];
        
		docTopicCounts = new HashMap[numDocs]; // indexed by <document
		hp.initial(docTopicCounts, numDocs);
		
		docTopicCountsSamples = new HashMap[numDocs]; // indexed by <document
		hp.initial(docTopicCountsSamples, numDocs);
		
		ilist=additionalDocuments;
		//initially randomly assign topics to each word in the new datasets
		for(int di=0;di<numDocs;di++){
			FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
			int seqLen = fs.getLength();
//			numTokens += seqLen;
			topics[di] = new int[seqLen];
			// Randomly assign tokens to topics
			for (int si = 0; si < seqLen; si++) {
				int topic = r.nextInt(numTopics);
				topics[di][si] = topic;// randomly assign topic to every words
										// in document di
				hp.addInteger(docTopicCounts[di], topic);
				hp.addInteger(docTopicCountsSamples[di], topic);
				hp.addInteger(newTypeTopicCounts[fs.getIndexAtPosition(si)], topic);
				tokensPerTopic[topic]++;
			}
			
		}
		this.estimate(0, numDocs, showTopicsInterval,
				outputModelInterval, outputModelFilename, r,burnIn, numIterations);
	}
/*
	public void addDocuments(InstanceList additionalDocuments, int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r, double w,int burnIn,int numIterations) {
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
		
//		int[][] newDocTopicCounts = new int[numDocs][numTopics];
//		docTopicCounts = newDocTopicCounts;
//		int[][] newTypeTopicCounts = new int[numTypes][numTopics];
		
		HashMap<Integer,Integer>[] newDocTopicCounts; // indexed by <document index, topic index>
		HashMap<Integer,Integer>[] newDocTopicCountsSamples;
		
		HashMap<Integer,Integer>[] newTypeTopicCounts; // indexed by <feature index, topic index>
		HashMap<Integer,Double>[] newTypeTopicCountsSamples;
		
		FeatureSequence fs1;
		for (int di = 0; di < numDocs; di++) {
			fs1 = (FeatureSequence) additionalDocuments.get(di).getData();
			int seqLen = fs1.getLength();
			numTokens += seqLen;
		}
*/
		/*
		 * if(numToken<(numTokens*w)) { w=(numTokens*w)/numToken; } else
		 * if(numToken>(numTokens/w)) { w=numTokens/(numToken*w); }
		 */
/*
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
*/
	/*
	 * Perform several rounds of Gibbs sampling on the documents in the given
	 * range.
	 */
	public void estimate(int docIndexStart, int docIndexLength, int showTopicsInterval, int outputModelInterval,
			String outputModelFilename, Randoms r,int burnIn, int numIterations) {
		for (int iterations = 0; iterations < numIterations; iterations++) {
			if (iterations % 10 == 0)
				System.out.print(iterations);
			else
				System.out.print(".");
			System.out.flush();
			if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0
					&& iterations > 0) {
				System.out.println();
				printTopWords(5, false);
			}
			if (outputModelInterval != 0
					&& iterations % outputModelInterval == 0 && iterations > 0) {
				this.write(new File(outputModelFilename + '.' + iterations));
			}
			sampleTopicsForDocs(docIndexStart, docIndexLength, r,burnIn,iterations);
		}

	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForDocs(int start, int length, Randoms r,int burnIn,int currentIteration) {
		assert (start + length <= docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start + length; di++) {
			sampleTopicsForOneDoc((FeatureSequence) ilist.get(di).getData(),
					topics[di], docTopicCounts[di], docTopicCountsSamples[di],topicWeights, r,burnIn,currentIteration);
		}
	}

	private void sampleTopicsForOneDoc(FeatureSequence oneDocTokens,
			int[] oneDocTopics, // indexed by seq position
//			int[] oneDocTopicCounts, // indexed by topic index
			HashMap<Integer,Integer> oneDocTopicCounts, // indexed by topic index
			HashMap<Integer,Integer> oneDocTopicCountsSamples,
			double[] topicWeights, Randoms r,int burnIn,int currentIteration) {
//		int[] currentTypeTopicCounts;
		HashMap<Integer,Integer> currentTypeTopicCounts = new HashMap<Integer,Integer>();
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			type = oneDocTokens.getIndexAtPosition(si);
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
//			oneDocTopicCounts[oldTopic]--;
//			typeTopicCounts[type][oldTopic]--;
//			tokensPerTopic[oldTopic]--;
			tokensPerTopic[oldTopic]--;
			hp.reduceInteger(oneDocTopicCounts, oldTopic);
			hp.reduceInteger(typeTopicCounts[type], oldTopic);
			// Build a distribution over topics for this token
			Arrays.fill(topicWeights, 0.0);
			topicWeightsSum = 0;
			currentTypeTopicCounts = typeTopicCounts[type];
			for (int ti = 0; ti < numTopics; ti++) {
//				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
//						* ((oneDocTopicCounts[ti] + alpha)); 
				tw = ((hp.getCountInteger(currentTypeTopicCounts, ti) + beta) / (tokensPerTopic[ti] + vBeta))
						* ((hp.getCountInteger(oneDocTopicCounts,ti) + alpha)); // (/docLen-1+tAlpha) is constant across all// (/docLen-1+tAlpha);
																// is constant
																// across all
																// topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
				// System.out.println(si+"---"+ ti+"---"+tw);
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete(topicWeights, topicWeightsSum);
			// System.out.println(si+"---"+ newTopic);
			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
//			oneDocTopicCounts[newTopic]++;
//			typeTopicCounts[type][newTopic]++;
//			tokensPerTopic[newTopic]++;
			hp.addInteger(oneDocTopicCounts, newTopic);
			hp.addInteger(typeTopicCounts[type], newTopic);
			tokensPerTopic[newTopic]++;
			
			if(currentIteration>burnIn-1){
				hp.addInteger(oneDocTopicCountsSamples,newTopic);
				hp.addDouble(typeTopicCountsSamples[type], newTopic);
				tokensPerTopicSamples[newTopic]++;
			}
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

	public HashMap<Integer,Integer>[]  getDocTopicCounts() {
		return docTopicCounts;
	}
	
	public HashMap<Integer,Integer>[]  getDocTopicCountsSamples() {
		return docTopicCountsSamples;
	}

	public HashMap<Integer,Integer>[]  getTypeTopicCounts() {
		return typeTopicCounts;
	}
	public HashMap<Integer,Double>[]  getTypeTopicCountsSamples() {
		return typeTopicCountsSamples;
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
						(double)(hp.getCountInteger(typeTopicCounts[wi],ti) + beta)
						/ (tokensPerTopic[ti] + vBeta));;
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

	public void printToFile(int numWords, boolean useNewLines, File file) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(new FileWriter(file));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		printToFile(numWords, useNewLines, pw);
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
//				wp[wi] = new WordProb(wi, ((double) typeTopicCounts[wi][ti])
//						/ tokensPerTopic[ti]);
				wp[wi] = new WordProb(wi,(hp.getCountDouble(typeTopicCountsSamples[wi],ti))
						/ (tokensPerTopicSamples[ti]));
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

//	public void SprintToDatabase(int numWords, boolean useNewLines, File file) {
//		PrintWriter out = null;
//		Connection conn = null;
//		Statement stmt = null;
//		String word = null;
//		try {
//			out = new PrintWriter(new FileWriter(file));
//			DataBase.openDB();
//			conn = DataBase.getDBConection();
//			stmt = DataBase.getDBStatement(conn);
//		} catch (IOException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		} catch (SQLException e1) {
//			// TODO Auto-generated catch block
//			e1.printStackTrace();
//		}
//		class WordProb implements Comparable {
//			int wi;
//			double p;
//
//			public WordProb(int wi, double p) {
//				this.wi = wi;
//				this.p = p;
//			}
//
//			public final int compareTo(Object o2) {
//				if (p > ((WordProb) o2).p)
//					return -1;
//				else if (p == ((WordProb) o2).p)
//					return 0;
//				else
//					return 1;
//			}
//		}
//
//		WordProb[] wp = new WordProb[numTypes];
//		for (int ti = 0; ti < numTopics; ti++) {
//			for (int wi = 0; wi < numTypes; wi++)
//				wp[wi] = new WordProb(wi,
//						(double) (typeTopicCounts[wi][ti] + beta)
//								/ (tokensPerTopic[ti] + vBeta));
//			Arrays.sort(wp);
//			try {
//				if (useNewLines) {
//					out.println("\nTopic " + ti);
//					for (int i = 0; i < numWords; i++) {
//						String sem = ilist.getDataAlphabet()
//								.lookupObject(wp[i].wi).toString();
//						String sql = "select * from dict_f where first_sem='"
//								+ sem + "' order by count desc";
//						ResultSet rs = DataBase.DBExecuteQuery(stmt, sql);
//						if (rs.next()) {
//							word = rs.getString("w_c");
//						} else {
//							word = sem;
//						}
//						out.println(word + "\t" + wp[i].p);
//					}
//				} else {
//					out.print("Topic " + ti + ": ");
//					for (int i = 0; i < numWords; i++) {
//						String sem = ilist.getDataAlphabet()
//								.lookupObject(wp[i].wi).toString();
//						String sql = "select * from dict_f where first_sem='"
//								+ sem + "' order by count desc";
//						ResultSet rs = DataBase.DBExecuteQuery(stmt, sql);
//						if (rs.next()) {
//							word = rs.getString("w_c");
//						} else {
//							word = sem;
//						}
//						out.println(word + "\t" + wp[i].p);
//					}
//					out.println();
//				}
//			} catch (SQLException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//		}
//		out.close();
//		try {
//			DataBase.closeDBStatement(stmt);
//			DataBase.closeDBConnection(conn);
//			DataBase.closeDB();
//		} catch (SQLException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}

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
//				wp[wi] = new WordProb(wi,
//						(double) (typeTopicCounts[wi][ti] + beta)
//								/ (tokensPerTopic[ti] + vBeta));
				wp[wi] = new WordProb(wi,
						(double) (hp.getCountInteger(typeTopicCounts[wi],ti) + beta)
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

	public void printDocumentTopics(File f,int burnIn,int numIterations) throws IOException {
		printDocumentTopics(new PrintWriter(new FileWriter(f)),burnIn,numIterations);
	}

	public void printDocumentTopics(PrintWriter pw,int burnIn,int numIterations) {
		printDocumentTopics(pw, 0.0, -1,burnIn,numIterations);
		pw.close();
	}

	public void printDocumentTopics(PrintWriter pw, double threshold, int max,int burnIn,int numIterations) {
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
//				topicDist[ti] = (((float) docTopicCounts[di][ti]) / docLen);
				topicDist[ti] = (((double) hp.getCountInteger(docTopicCountsSamples[di],ti)) / (docLen*(numIterations-burnIn)));
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

	public void SprintDocumentTopics(File f,int burnIn,int numIterations) throws IOException {
		SprintDocumentTopics(new PrintWriter(new FileWriter(f)),burnIn,numIterations);
	}

	public void SprintDocumentTopics(PrintWriter pw,int burnIn,int numIterations) {
		SprintDocumentTopics(pw, 0.0, -1,burnIn,numIterations);
		pw.close();
	}

	public void SprintDocumentTopics(PrintWriter pw, double threshold, int max,int burnIn,int numIterations) {
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
//				topicDist[ti] = ((double) (docTopicCounts[di][ti] + alpha) / (docLen + tAlpha));
				topicDist[ti] = ((double) (hp.getCountInteger(docTopicCountsSamples[di],ti)/(numIterations-burnIn) + alpha) / (docLen + tAlpha));
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
//				topicdoc[ti][di] = ((double) (docTopicCounts[di][ti] + alpha) / (topics[di].length + tAlpha));
				topicdoc[ti][di] = ((double) (hp.getCountInteger(docTopicCounts[di],ti) + alpha) / (topics[di].length + tAlpha));
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
//				wp[ti][di] = new WordProb(
//						di,
//						((double) (docTopicCounts[di][ti] + alpha) / (topics[di].length + tAlpha)));
				wp[ti][di] = new WordProb(di,((double) (hp.getCountInteger(docTopicCounts[di],ti) + alpha) / (topics[di].length + tAlpha)));
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
				out.writeInt(hp.getCountInteger(docTopicCounts[di],ti));
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt(hp.getCountInteger(typeTopicCounts[fi],ti));
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
		int  numDocs = ilist.size();
		topics = new int[numDocs][];
		for (int di = 0; di < ilist.size(); di++) {
			int docLen = ((FeatureSequence) ilist.get(di).getData())
					.getLength();
			topics[di] = new int[docLen];
			for (int si = 0; si < docLen; si++)
				topics[di][si] = in.readInt();
		}
		hp.initial(docTopicCounts, numDocs);
		for (int di = 0; di < ilist.size(); di++)
			for (int ti = 0; ti < numTopics; ti++)
				docTopicCounts[di].put(ti, in.readInt());
		int numTypes = ilist.getDataAlphabet().size();
		hp.initial(typeTopicCounts,numTypes);
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				typeTopicCounts[fi].put(ti, in.readInt());
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
	
	public void printSampleTopics(StandardLDA_WT lda,String sampleTopicLogFile){
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

	public static void main(String[] args) throws IOException {
		String mallet_file="F:\\Experiment\\LDAInferrence\\mallet\\selected_sf_training_stemmed_files.mallet";
		InstanceList ilist = InstanceList.load(new File(mallet_file));
		int numIterations = 50;
		int numTopWords = 20;
		int numTopics = 362;
		int burnIn = 48;
		int print_interval = 25;
		
		System.out.println("Training begin......");
		StandardLDA_WT lda = new StandardLDA_WT(numTopics);
		lda.estimate(ilist, print_interval, 0, null, new Randoms(),burnIn,numIterations); // should
																		// be
		String topic_file = "F:\\Experiment\\LDAInferrence\\results\\training.topic";
		String lda_file = "F:\\Experiment\\LDAInferrence\\results\\training.lda";
		lda.printTopWords(numTopWords, true);
		lda.printToFile(numTopWords, true, new File(topic_file));
		lda.printDocumentTopics(new File(lda_file),burnIn,numIterations);
		
		
		int inferNumIterations = 10;
		int inferBurnIn = 9;
		int inferInterval = 5;
		int outputModelInterval=0;
		String infer_topic = "F:\\Experiment\\LDAInferrence\\results\\infer_training.topic";
		String infer_lda = "F:\\Experiment\\LDAInferrence\\results\\infer_training.lda";
		
		System.out.println("Inference begin......");
		String add_mallet_files = "F:\\Experiment\\LDAInferrence\\mallet\\selected_sf_testing_stemmed_files.mallet";
		InstanceList inferDocuments = InstanceList.load(new File(add_mallet_files));
		
		lda.Inferrence(inferDocuments, inferInterval, outputModelInterval, null, new Randoms(), inferBurnIn, inferNumIterations);
		lda.printToFile(numTopWords, true, new File(infer_topic));
		lda.printDocumentTopics(new File(infer_lda),burnIn,numIterations);
	}
}
