package edu.nudt.influx.lda;


import cc.mallet.types.*;
import cc.mallet.util.Randoms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.io.*;
import java.text.NumberFormat;

/**
 * Hierarchical Pachinko Allocation with MLE learning, 
 *  based on Mallet
 * @author 
 */

public class PAMHierarchicalSampling {
	
	//for tree structure
	TopicNode root = new TopicNode();

	//id 2 name map
	HashMap<Integer, String> topic2name = new HashMap<Integer, String>();
	
	public PrintWriter pw = null;
	public String topTopicWordFname;
	double [] alphas;
	double alphaSum;
	double beta;   // Prior on per-topic multinomial distribution over words
	double vBeta; 

	int num_tree_depth;
	// Data
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence
	int numTypes; //number of distinct words in dataset
	int numTokens; //total number of words
	int numDocs; //number of examples
	
	int numTopics; //number of topics in the given hierarchy
	// Gibbs sampling state
	//  (these could be shorts, or we could encode both in one int)
	// indexed by <document index, sequence index>
	// contains the topic path for the word position i in document d, i.e., <d, i>
	int [][][] topics;	
	
	double[] subAlphaSums;
	
	//wroking variables for a topic path
	// Per-document state variables		
	int [][] docTopicCounts; //<document, topic>
	double[] samplingWeights; // the component of the Gibbs update that depends on topics	

	//for p(w|leaf_topic) distribution
	// Per-word type state variables		
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>

	// Histograms for MLE
	int[][] superTopicHistograms; // histogram of # of words per supertopic in documents
	//  eg, [17][4] is # of docs with 4 words in sT 17...
	int[][][] subTopicHistograms; // for each supertopic, histogram of # of words per subtopic

	Runtime runtime;
	NumberFormat formatter;

	public PAMHierarchicalSampling () {
		this (50.0, 0.001);
	}

	public PAMHierarchicalSampling (double alphaSum, double beta) {
		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);
		
		this.beta = beta; // We can't calculate vBeta until we know how many word types...
		this.alphaSum = alphaSum;
		runtime = Runtime.getRuntime();
	}

	public void init()
	{
		numTopics = root.getTreeSize();
		
		//get tree depth
		num_tree_depth = root.getHeight() - 1;//do not count the root
		
		//get all alphas		
		alphas = new double[numTopics];
		root.getAllAlpha(alphas);					
		
		subAlphaSums = new double[numTopics];
		//just fill alphaSum, such as the first level size
		Arrays.fill(subAlphaSums, alphaSum);		
	}
	
	public void estimate (InstanceList documents, int numIterations, int optimizeInterval, 
	                      int showTopicsInterval,
	                      int outputModelInterval, String outputModelFilename, int topWords,
	                      Randoms r) throws IOException
	{
		ilist = documents;
		
		//get number of words
		numTypes = ilist.getDataAlphabet().size ();
		
		vBeta = beta * numTypes;
		
		//get number of documents
		numDocs = ilist.size();
				
		//state variable to hold the topic path for each position in each document
		topics = new int [numDocs][][];
		
		//		Allocate several arrays for use within each document
		//		to cut down memory allocation and garbage collection time	
		
		typeTopicCounts = new int[numTypes][numTopics];//count for <word, topic>
		tokensPerTopic = new int[numTopics];//count for <topic>
		docTopicCounts = new int[numDocs][numTopics];//count for <topic, document>
		
		//for sampling usage
		samplingWeights = new double[numTopics];
		
		//fill each array with zero	
		for (int t = 0; t < numTypes; t++) {
			Arrays.fill(typeTopicCounts[t], 0);	
		}		
		Arrays.fill(tokensPerTopic, 0);		
		Arrays.fill(samplingWeights, 0);						
		
		long startTime = System.currentTimeMillis();

		int maxTokens = 0;

		//		Initialize with random assignments of tokens to topics
		//		and finish allocating this.topics and this.tokens

		int seqLen;

		for (int di = 0; di < numDocs; di++) {

			FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();

			seqLen = fs.getLength();
			if (seqLen > maxTokens) { 
				maxTokens = seqLen;
			}

			numTokens += seqLen;
			topics[di] = new int[seqLen][];			

			// Randomly assign a path to each token
			for (int si = 0; si < seqLen; si++) {
				//Randomly sample a path
				int [] path = root.samplingPath(r);
				//assign the path to the token
				topics[di][si] = new int [num_tree_depth + 1];
				//add path
				for(int p=0;p<path.length;p++)
					topics[di][si][p] = path[p];
				topics[di][si][path.length] = -1;//add a stop sign
				//update the token count for each topic
				for(int p=0;p<path.length;p++)
				{					
					// For each topic, we also need to update the 
					//  word type statistics
					typeTopicCounts[fs.getIndexAtPosition(si) ][path[p]]++;
					tokensPerTopic[path[p]]++;	
					//update doc topic count
					docTopicCounts[di][path[p]]++;
				}									
			}
		}		
		
		System.out.println("total docs: " + numDocs);
		System.out.println("total topics: " + numTopics);
		System.out.println("max tokens: " + maxTokens);
		System.out.println("total tokens: " + numTokens);
				
		superTopicHistograms = new int[numTopics][maxTokens + 1];
		subTopicHistograms = new int[numTopics][numTopics][maxTokens + 1];
		
//		These will be initialized at the first call to 
		clearHistograms(); //in the loop below.
		
		//		Finally, start the sampler!

		for (int iterations = 0; iterations < numIterations; iterations++) {
			long iterationStart = System.currentTimeMillis();

			clearHistograms();
			sampleTopicsForAllDocs (r);

			// There are a few things we do on round-numbered iterations
			//  that don't make sense if this is the first iteration.

			if (iterations > 0) {
				if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0) {
					System.out.println ();
					//printTopWords (5, false);
					
					pw = new PrintWriter(new FileWriter(this.topTopicWordFname + "_" + iterations + ".xml"));										
					//fprintTopWords (5, false, pw);
					printTopWordsAsTreeToFile(root, 0, topWords, pw);
					pw.flush();
					pw.close();
				}
				if (outputModelInterval != 0 && iterations % outputModelInterval == 0) {
					//this.write (new File(outputModelFilename+'.'+iterations));
				}				
				
				if (optimizeInterval != 0 && iterations % optimizeInterval == 0) {
					long optimizeTime = System.currentTimeMillis();
					
					//update the parameter on the tree
					///for(int k=0;k<root.numSubTopics;k++)
					//	learnParameterOnTree(root.subNodes.get(k));
					/*
					for (superTopic = 0; superTopic < numSuperTopics; superTopic++) {
						learnParameters(subAlphas[superTopic],
								subTopicHistograms[superTopic],
								superTopicHistograms[superTopic]);
						subAlphaSums[superTopic] = 0.0;
						for (subTopic = 0; subTopic < numSubTopics; subTopic++) {
							subAlphaSums[superTopic] += subAlphas[superTopic][subTopic];
						}
					}
					*/
					//System.out.print("[o:" + (System.currentTimeMillis() - optimizeTime) + "]");
				}	
							
			}

			if (iterations > 1107) {
				//printWordCounts();
			}

			if (iterations % 10 == 0)
				System.out.println ("<" + iterations + "> ");

			//System.out.print((System.currentTimeMillis() - iterationStart) + " ");

			//else System.out.print (".");
			System.out.flush();
		}

		long seconds = Math.round((System.currentTimeMillis() - startTime)/1000.0);
		long minutes = seconds / 60;	seconds %= 60;
		long hours = minutes / 60;	minutes %= 60;
		long days = hours / 24;	hours %= 24;
		System.out.print ("\nTotal time: ");
		if (days != 0) { System.out.print(days); System.out.print(" days "); }
		if (hours != 0) { System.out.print(hours); System.out.print(" hours "); }
		if (minutes != 0) { System.out.print(minutes); System.out.print(" minutes "); }
		System.out.print(seconds); System.out.println(" seconds");

		//		124.5 seconds
		//		144.8 seconds after using FeatureSequence instead of tokens[][] array
		//		121.6 seconds after putting "final" on FeatureSequence.getIndexAtPosition()
		//		106.3 seconds after avoiding array lookup in inner loop with a temporary variable

	}

	public void learnParameterOnTree(TopicNode nd)
	{
		if(nd.numSubTopics != 0)
		{
			int superTopic = nd.labelIndex;
			double [] subalphas = nd.subAlphas;
			learnParameters(subalphas,
				subTopicHistograms[superTopic],
				superTopicHistograms[superTopic]);
			subAlphaSums[superTopic] = 0.0;
						
			for (int i=0;i<nd.numSubTopics;i++) 
			{				
				subAlphaSums[superTopic] += subalphas[i];
			}
		}
	}
	
	private void clearHistograms() {
		
		for (int superTopic = 0; superTopic < numTopics; superTopic++) {
			Arrays.fill(superTopicHistograms[superTopic], 0);
			for (int subTopic = 0; subTopic < numTopics; subTopic++) {
				Arrays.fill(subTopicHistograms[superTopic][subTopic], 0);
			}
		}	
	}

	/** Use the fixed point iteration described by Tom Minka. */
	public void learnParameters(double[] parameters, int[][] observations, int[] observationLengths) {
		int i, k;

		double parametersSum = 0;

		//		Initialize the parameter sum

		for (k=0; k < parameters.length; k++) {
			parametersSum += parameters[k];
		}

		double oldParametersK;
		double currentDigamma;
		double denominator;

		int[] histogram;

		int nonZeroLimit;
		int[] nonZeroLimits = new int[observations.length];
		Arrays.fill(nonZeroLimits, -1);

		//		The histogram arrays go up to the size of the largest document,
		//		but the non-zero values will almost always cluster in the low end.
		//		We avoid looping over empty arrays by saving the index of the largest
		//		non-zero value.

		for (i=0; i<observations.length; i++) {
			histogram = observations[i];
			for (k = 0; k < histogram.length; k++) {
				if (histogram[k] > 0) {
					nonZeroLimits[i] = k;
				}
			}
		}

		for (int iteration=0; iteration<200; iteration++) {

			// Calculate the denominator
			denominator = 0;
			currentDigamma = 0;

			// Iterate over the histogram:
			for (i=1; i<observationLengths.length; i++) {
				currentDigamma += 1 / (parametersSum + i - 1);
				denominator += observationLengths[i] * currentDigamma;
			}

			/*
   if (Double.isNaN(denominator)) {
	System.out.println(parameterSum);
	for (i=1; i < observationLengths.length; i++) {
	    System.out.print(observationLengths[i] + " ");
	}
	System.out.println();
   }
			 */

			// Calculate the individual parameters

			parametersSum = 0;

			for (k=0; k<parameters.length; k++) {

				// What's the largest non-zero element in the histogram?
				nonZeroLimit = nonZeroLimits[k];

				// If there are no tokens assigned to this super-sub pair
				//  anywhere in the corpus, bail.

				if (nonZeroLimit == -1) {
					parameters[k] = 0.000001;
					parametersSum += 0.000001;
					continue;
				}

				oldParametersK = parameters[k];
				parameters[k] = 0;
				currentDigamma = 0;

				histogram = observations[k];

				for (i=1; i <= nonZeroLimit; i++) {
					currentDigamma += 1 / (oldParametersK + i - 1);
					parameters[k] += histogram[i] * currentDigamma;
				}

				parameters[k] *= oldParametersK / denominator;

				if (Double.isNaN(parameters[k])) {
					System.out.println("parametersK *= " + 
							oldParametersK + " / " +
							denominator);
					for (i=1; i < histogram.length; i++) {
						System.out.print(histogram[i] + " ");
					}
					System.out.println();
				}

				parametersSum += parameters[k];
			}
		}
	}

	/* One iteration of Gibbs sampling, across all documents. */
	private void sampleTopicsForAllDocs (Randoms r)
	{
//		Loop over every word in the corpus
		for (int di = 0; di < numDocs; di++) {

			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
					topics[di], docTopicCounts[di], r);
		}
	}

	
	
	private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens,
	                                    int[][] topic_paths, // indexed by seq position, a sampled topic path for the position	       
	                                    int []oneDocTopicCounts, //indexed by topic
	                                    Randoms r) {

//		long startTime = System.currentTimeMillis();	
		int type, subTopic;
		double cumulativeWeight;

		int docLen = oneDocTokens.getLength();

//		Iterate over the positions (words) in the document

		for (int si = 0; si < docLen; si++) {
			//get word at this position
			type = oneDocTokens.getIndexAtPosition(si);
			
			//get topic path of this position
			int [] path = topic_paths[si];									

			// Remove this token from all counts among the path
			for(int p=0;p<path.length && path[p] != -1;p++)
				oneDocTopicCounts[path[p]] --;
					
			//KDD 11 paper just samples a path
			//but now, we want to sample the topics from top level to the bottom level
			//so we keep booking every <word, topic> count
			for(int p=0;p<path.length && path[p] != -1;p++)
			{
				//book the <word, super topic> count
				typeTopicCounts[type][path[p]]--;
				tokensPerTopic[path[p]]--;
			}
						
			//
			//we iteratively sample a path from the top level to the bottom level
			//at each node, we sample the child node based on the probability estimation as p(t|w,d) = P(w|t)P(t|d)										
					
			int sampled_topic_index = 0;
			int sampled_topic = 0;
			int topic_in_path_index = 0;
			//keep sampling from the root node until we see a leaf node			
			TopicNode currentNode = root;						
			while(currentNode.numSubTopics != 0)
			{				
				cumulativeWeight = 0.0;
				for(int i=0;i<currentNode.numSubTopics;i++)
				{
					subTopic = currentNode.subNodes.get(i).labelIndex;																		
					
					//we do not need to keep the <super topic, sub topic> pair 
					//as we are considering the hierarchy structure where each node can only have one parent topic
					samplingWeights[i] = 
						((double) oneDocTopicCounts[subTopic] + alphas[subTopic])
						 * ((double) typeTopicCounts[type][subTopic] + beta) / ((double) tokensPerTopic[subTopic] + vBeta);
					
					cumulativeWeight += samplingWeights[i]; 
				}
				//sampling a sub topic
				sampled_topic_index = r.nextDiscreteWithSize(samplingWeights, currentNode.numSubTopics, cumulativeWeight);
				//get the topic ID
				sampled_topic = currentNode.subNodes.get(sampled_topic_index).labelIndex;				
				//add it into path
				topic_paths[si][topic_in_path_index] = sampled_topic;
				topic_in_path_index++;
				currentNode = currentNode.subNodes.get(sampled_topic_index);
			}
			//put a stop sign
			topic_paths[si][topic_in_path_index] = -1;
			
			// Put the new <super, sub> pair in the path into the counts
			for(int p=0;p<topic_in_path_index;p++)
			{
				oneDocTopicCounts[topic_paths[si][p]]++;						
				typeTopicCounts[type][topic_paths[si][p]]++;
				tokensPerTopic[topic_paths[si][p]]++;			
			}						
		}

		//		Update the topic count histograms
		//		for dirichlet estimation
		//recusively update all <parent, child> pairs
		//for(int i=0;i<root.numSubTopics;i++)
		//	updateTopicHistogramOnTree(root.subNodes.get(i));				
	}

	public void printTopWords (int numWords, boolean useNewLines) 
	{
		IDSorter[] wp = new IDSorter[numTypes];		
		
		//the actual topics are from 1 to N-1
		for(int topic=1;topic<numTopics;topic++)
		{			
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new IDSorter (wi, (((double) typeTopicCounts[wi][topic]) /
						tokensPerTopic[topic]));
			Arrays.sort (wp);

			StringBuffer topicTerms = new StringBuffer();
			for (int i = 0; i < numWords; i++) {
				topicTerms.append(ilist.getDataAlphabet().lookupObject(wp[i].wi));
				topicTerms.append(" ");
			}
			System.out.println(topic + ": " + topicTerms.toString());			
		}
	}
	
	public void fprintTopWords (int numWords, boolean useNewLines, PrintWriter pw) 
	{
		IDSorter[] wp = new IDSorter[numTypes];		
		
		//the actual topics are from 1 to N-1
		for(int topic=1;topic<numTopics;topic++)
		{			
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new IDSorter (wi, (((double) typeTopicCounts[wi][topic]) /
						tokensPerTopic[topic]));
			Arrays.sort (wp);

			StringBuffer topicTerms = new StringBuffer();
			for (int i = 0; i < numWords; i++) {
				topicTerms.append(ilist.getDataAlphabet().lookupObject(wp[i].wi));
				topicTerms.append(" ");
			}
			pw.println(topic + ": " + topicTerms.toString());			
		}
	}
	
	public String topWodsToString(int topic, int numWords)
	{
		IDSorter[] wp = new IDSorter[numTypes];
		//print word
		for (int wi = 0; wi < numTypes; wi++)
			wp[wi] = new IDSorter (wi, (((double) typeTopicCounts[wi][topic]) /
					tokensPerTopic[topic]));
		Arrays.sort (wp);
		StringBuffer topicTerms = new StringBuffer();
		for (int i = 0; i < numWords; i++) {
			topicTerms.append(ilist.getDataAlphabet().lookupObject(wp[i].wi));
			topicTerms.append(" ");
		}
		return topicTerms.toString();
	}
	
	public void printTopWordsAsTreeToFile(TopicNode nd, int depth, int numWords, PrintWriter pw)
	{
		int topic = nd.labelIndex;		
		for(int i=0;i<depth;i++)
			pw.print("\t");		
		String name = "";
		if(topic != 0)
			name = topic2name.get(topic);
		pw.print("<node id=\"" + nd.labelIndex + "\" name=\"" + name+ "\">\n");
		if(topic != 0)
		{
			for(int i=0;i<depth+1;i++)
				pw.print("\t");		
			String topwords = this.topWodsToString(topic, numWords);		
			pw.println("<topwords>" + topwords + "</topwords>");
		}
		for(int i=0;i<nd.subNodes.size();i++)
		{
			printTopWordsAsTreeToFile(nd.subNodes.get(i), depth + 1, numWords, pw);			
		}		
		for(int i=0;i<depth;i++)
			pw.print("\t");
		pw.print("</node>\n");	
	}	
	
	/*
	public void updateTopicHistogramOnTree(TopicNode nd)
	{
		int superTopic = nd.labelIndex;
		for(int i=0;i<nd.numSubTopics;i++)
		{
			int subTopic = nd.subNodes.get(i).labelIndex;
			superTopicHistograms[superTopic][ superCounts[superTopic] ]++;			
			subTopicHistograms[superTopic][subTopic][superSubCounts[superTopic][subTopic] ]++;
			updateTopicHistogramOnTree(nd.subNodes.get(i));
		}
	}
	*/
	
	public void readID2name(String fname) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(fname));
		String line = "";
		while((line = br.readLine()) != null)
		{
			String [] lines = line.split(":");
			topic2name.put(Integer.parseInt(lines[0]), lines[1]);
		}
		br.close();		
	}		
	
	public void generateTreeWithLeafOnEachIntermediateNode(String inFname, String outFname) throws IOException
	{
		//get all intermedia topic IDs except the root
		ArrayList<Integer> interNodes = new ArrayList<Integer>();
		root.getIntermediaNodes(interNodes);
		
		//make new file 
		PrintWriter pw = new PrintWriter(new FileWriter(outFname));
		BufferedReader br = new BufferedReader(new FileReader(inFname));
		String buffer = "";
		while((buffer = br.readLine()) != null)
		{
			pw.println(buffer);
		}
		br.close();
		//write the new added intermedia nodes
		for(int i=0;i<interNodes.size();i++)
		{
			pw.println(interNodes.get(i) + "," + (numTopics + i));
		}
		pw.flush();
		pw.close();
	}
	
	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException
	{
		args = new String [5];
		//args[0] = "/home/xiao/test_mallet/topic_input.mallet";
		//args[0] = "/home/xiao/datasets/software/sourceforge/sourceforge_mallet.mallet";
		//args[1] = "/home/xiao/workspace/test/sf_hier_par_child_pairs_first_level.txt";
		//args[1] = "/home/xiao/workspace/test/sf_hier_par_child_pairs.txt";
		//args[2] = "/home/xiao/datasets/software/sourceforge/sf_topics_id_string.txt";
		//args[3] = "sourceforge";
		//RCV1
		args[0] = "/home/xiao/datasets/rcv1/mallet/lyrl2004_tokens_train.mallet.new.mallet";
		args[1] = "/home/xiao/datasets/rcv1/mallet/rcv1_hierarchy.par_child";
		args[2] = "/home/xiao/datasets/rcv1/mallet/rcv1_id2cat.map";
		args[3] = "rcv1";
		args[4] = "500";
		
		int numIterations = Integer.parseInt(args[4]);
		int numTopWords = 50;		
		System.out.println ("Data loaded.");
		
		double alphaSum = 50;
		
		PAMHierarchicalSampling pam = new PAMHierarchicalSampling ();
				
		String hierFname = args[1];
		String topic2nameFname = args[2];
		String topTopicWordFname = "/home/xiao/workspace/test/"+args[3]+"/top_words";
		
		//new topic hierarchy where some new nodes are added into the hierarchy
		String newhierFname = "/home/xiao/workspace/test/sf_hier_par_child_leaf_on_intermedia_pairs.txt";								
		
		pam.readID2name(topic2nameFname);
		
		//pam.root = TopicNode.buildHierarchy(newhierFname);
		pam.root = TopicNode.buildHierarchy(hierFname);
		
		pam.root.initTree(alphaSum);
		
		pam.init();
		
		//pam.generateTreeWithLeafOnEachIntermediateNode(hierFname, newhierFname);
		
		pam.topTopicWordFname = topTopicWordFname;
		
		InstanceList ilist = InstanceList.load (new File(args[0]));
		
		pam.estimate (ilist, numIterations, 50, 10, 50, null, numTopWords, new Randoms());  // should be 1100		
	}

	class IDSorter implements Comparable {
		int wi; double p;
		public IDSorter (int wi, double p) { this.wi = wi; this.p = p; }
		public final int compareTo (Object o2) {
			if (p > ((IDSorter) o2).p)
				return -1;
			else if (p == ((IDSorter) o2).p)
				return 0;
			else return 1;
		}
	}

}
