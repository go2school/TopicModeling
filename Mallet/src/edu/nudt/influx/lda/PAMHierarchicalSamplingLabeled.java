package edu.nudt.influx.lda;


import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.*;
import cc.mallet.util.Randoms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeSet;
import java.io.*;
import java.text.NumberFormat;

/**
 * Hierarchical Pachinko Allocation with MLE learning, 
 *  based on Mallet
 * @author 
 */

public class PAMHierarchicalSamplingLabeled implements Serializable  {
	
	//for tree structure
	TopicNode root = new TopicNode();

	//id 2 name map
	HashMap<Integer, String> topic2name = null;
		
	public PrintWriter pw = null;
	public String topTopicWordFname;
	
	double [] alphas;
	double alphaSum;
	double beta;   // Prior on per-topic multinomial distribution over words
	double vBeta; 

	int num_tree_depth;
	// Data
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence
	public ArrayList<TreeSet<Integer>> docLabels = new ArrayList<TreeSet<Integer>>();//the labels of each document  
	
	int numTypes; //number of distinct words in dataset
	int numTokens; //total number of words
	public int numDocs; //number of examples
	
	int numTopics; //number of topics in the given hierarchy
		
	double[] subAlphaSums;
	
	double[] samplingWeights; // the component of the Gibbs update that depends on topics	
	
	//for training 
	public int test_numDocs; //number of examples
	// Gibbs sampling state
	//  (these could be shorts, or we could encode both in one int)
	// indexed by <document index, sequence index>
	// contains the topic path for the word position i in document d, i.e., <d, i>
	int [][][] topics;	
	//wroking variables for a topic path
	// Per-document state variables		
	public int [][] docTopicCounts; //<document, topic>	
	//for p(w|leaf_topic) distribution
	// Per-word type state variables		
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>

	//for inference
	InstanceList test_ilist;
	int [][][] test_topics;	
	//wroking variables for a topic path
	// Per-document state variables		
	public int [][] test_docTopicCounts; //<document, topic>	
	//for p(w|leaf_topic) distribution
	// Per-word type state variables		
	int[][] test_typeTopicCounts; // indexed by <feature index, topic index>
	int[] test_tokensPerTopic; // indexed by <topic index>
	double [][] test_docTopicDistribution;
	double [] topic_thresholds;
	
	// Histograms for MLE
	int[][] superTopicHistograms; // histogram of # of words per supertopic in documents
	//  eg, [17][4] is # of docs with 4 words in sT 17...
	int[][][] subTopicHistograms; // for each supertopic, histogram of # of words per subtopic

	Runtime runtime;
	NumberFormat formatter;

	public PAMHierarchicalSamplingLabeled () {
		this (50.0, 0.001);
	}

	public PAMHierarchicalSamplingLabeled (double alphaSum, double beta) {
		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);
		
		this.beta = beta; // We can't calculate vBeta until we know how many word types...
		this.alphaSum = alphaSum;
		runtime = Runtime.getRuntime();
	}

	public void initTree(String hierFname) throws IOException
	{
		//build tree
		root = TopicNode.buildHierarchy(hierFname);
		
		//setup alpha for each node
		//these alphas are just set without change
		//in the ICML PAM paper, they use a method to estiamte alpha
		root.initTree(alphaSum);		
		
		numTopics = root.getTreeSize();
		
		//get tree depth
		num_tree_depth = root.getHeight() - 1;//do not count the root				
	}
	
	public void initAlpha(double alphaSum)
	{
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

	public void infer_topics (InstanceList documents, int numIterations, int optimizeInterval, 
            int showTopicsInterval,
            int outputModelInterval, String outputModelFilename, int topWords,
            Randoms r) throws IOException
	{
		test_ilist = documents;
			
		//get number of documents
		test_numDocs = test_ilist.size();
			
		//state variable to hold the topic path for each position in each document
		test_topics = new int [test_numDocs][][];
		
		//		Allocate several arrays for use within each document
		//		to cut down memory allocation and garbage collection time			
		test_typeTopicCounts = new int[numTypes][numTopics];//count for <word, topic>
		test_tokensPerTopic = new int[numTopics];//count for <topic>
		test_docTopicCounts = new int[test_numDocs][numTopics];//count for <topic, document>
		
		//for sampling usage
		samplingWeights = new double[numTopics];
		
		//fill each array with zero	
		for (int t = 0; t < numTypes; t++) {
			Arrays.fill(test_typeTopicCounts[t], 0);	
		}		
		Arrays.fill(test_tokensPerTopic, 0);		
		Arrays.fill(samplingWeights, 0);						
			
		int maxTokens = 0;
		int totalTokens = 0;
		
		//		Initialize each token to a topic path		
		int seqLen;
		int type;
		for (int di = 0; di < test_numDocs; di++) {		
			FeatureSequence fs = (FeatureSequence) test_ilist.get(di).getData();
			
			seqLen = fs.getLength();
			if (seqLen > maxTokens) { 
				maxTokens = seqLen;//record the maximal document length
			}
			totalTokens += seqLen;
			
			test_topics[di] = new int[seqLen][];			
			
			// Randomly assign a path to each token
			for (int si = 0; si < seqLen; si++) {				
				type = fs.getIndexAtPosition(si);

				int [] path = null;
				if (!(type < numTypes && typeTopicCounts[type].length != 0))
				{
					//if the word does not appear in the training set
					//randomly sample a path
					path = root.samplingPath(r);
				}
				else
				{	
					//if the word appears in the training set
					//find the most common topic based on the estimated parameter of count(word, topic) from the training data
					path = this.getMostCommonPath(type, r);
				}
				//assign the path to the token
				test_topics[di][si] = path;			
				//update the token count for each topic
				for(int p=0;p<path.length && path[p] != -1;p++)
				{					
					// For each topic, we also need to update the 
					//  word type statistics
					test_typeTopicCounts[type][path[p]]++;
					test_tokensPerTopic[path[p]]++;	
					//update doc topic count
					test_docTopicCounts[di][path[p]]++;
				}									
			}
		}		
		
		System.out.println("total test docs: " + test_numDocs);
		System.out.println("total topics: " + numTopics);
		System.out.println("max document length: " + maxTokens);
		System.out.println("total tokens in the test dataset: " + numTokens);
					
		//		Finally, start the sampler!		
		for (int iterations = 0; iterations < numIterations; iterations++) {									
			for (int di = 0; di < test_numDocs; di++) {
				test_sampleTopicsForOneDoc ((FeatureSequence)test_ilist.get(di).getData(),
						test_topics[di], test_docTopicCounts[di], r);
			}
			if (iterations % 10 == 0)
				System.out.println ("<" + iterations + "> ");
		}		
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
					topics[di], docTopicCounts[di], docLabels.get(di), r);
		}
	}

	private ArrayList<Integer> getLabeledNodeUnder(TopicNode root, TreeSet<Integer> labels)
	{
		ArrayList<Integer> labeledNodes = new ArrayList<Integer>();
		for(int i=0;i<root.subNodes.size();i++)
		{			
			if(labels.contains(root.subNodes.get(i).labelIndex))
				labeledNodes.add(i);
		}
		return labeledNodes;
	}
	
	private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens,
	                                    int[][] topic_paths, // indexed by seq position, a sampled topic path for the position	       
	                                    int [] oneDocTopicCounts, //indexed by topic
	                                    TreeSet<Integer> oneDocTopiclabels, 
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

			// Remove this token from all topic counts on the path
			for(int p=0;p<path.length && path[p] != -1;p++)
				oneDocTopicCounts[path[p]] --;
					
			//KDD 11 paper just samples a path directly 
			//but now, we want to iteratively sample the topics from top level to the bottom level
			//you can think about it as a hierarchical version of the standard LDA
			//we keep booking every <word, topic> count
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
				//get the actual nodes on this level	
				//this is a mimic of extracting the labeled subtree
				ArrayList<Integer> labeledTopicIndex = getLabeledNodeUnder(currentNode, oneDocTopiclabels);
				
				//no labeled labels available, just quit
				if(labeledTopicIndex.size() == 0)
					break;
				
				//compute the weight for the sub nodes
				cumulativeWeight = 0.0;
				for(int i=0;i<labeledTopicIndex.size();i++)
				{
					subTopic = currentNode.subNodes.get(labeledTopicIndex.get(i)).labelIndex; 												
										
					samplingWeights[i] = 
						((double) oneDocTopicCounts[subTopic] + alphas[subTopic])
						 * ((double) typeTopicCounts[type][subTopic] + beta) / ((double) tokensPerTopic[subTopic] + vBeta);
					
					cumulativeWeight += samplingWeights[i]; 
				}
												
				//sampling a sub topic index
				sampled_topic_index = r.nextDiscreteWithSize(samplingWeights, labeledTopicIndex.size(), cumulativeWeight);
				//get the actual topic ID
				sampled_topic = currentNode.subNodes.get(labeledTopicIndex.get(sampled_topic_index)).labelIndex; 	 					
				
				//add it into path
				topic_paths[si][topic_in_path_index] = sampled_topic;
				topic_in_path_index++;
				
				//go to the next level of the tree
				currentNode = currentNode.subNodes.get(labeledTopicIndex.get(sampled_topic_index));
			}
			//put a stop sign
			topic_paths[si][topic_in_path_index] = -1;
			
			// Put the new topics in the path
			for(int p=0;p<topic_in_path_index;p++)
			{
				oneDocTopicCounts[topic_paths[si][p]]++;						
				typeTopicCounts[type][topic_paths[si][p]]++;
				tokensPerTopic[topic_paths[si][p]]++;			
			}						
		}

		//just skip this part as we are not sure about how the parameter updating works
		//		Update the topic count histograms
		//		for dirichlet estimation
		//recusively update all <parent, child> pairs
		//for(int i=0;i<root.numSubTopics;i++)
		//	updateTopicHistogramOnTree(root.subNodes.get(i));				
	}

	private void test_sampleTopicsForOneDoc (FeatureSequence oneDocTokens,           
            //testing data
            int[][] test_topic_paths, // indexed by seq position, a sampled topic path for the position	       
            int [] test_oneDocTopicCounts, //indexed by topic            
            Randoms r) {
	
		//long startTime = System.currentTimeMillis();	
		int type, subTopic;
		double cumulativeWeight;
		
		int docLen = oneDocTokens.getLength();
		
		//Iterate over the positions (words) in the document
		
		for (int si = 0; si < docLen; si++) {
			//get word at this position
			type = oneDocTokens.getIndexAtPosition(si);
			
			//get the estimated topic path of this position
			int [] path = test_topic_paths[si];									
			
			// Remove this token from all topic counts on the path
			for(int p=0;p<path.length && path[p] != -1;p++)
				test_oneDocTopicCounts[path[p]] --;
			
			for(int p=0;p<path.length && path[p] != -1;p++)
			{
				//book the <word, super topic> count
				test_typeTopicCounts[type][path[p]]--;
				test_tokensPerTopic[path[p]]--;
			}
			
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
						((double)test_oneDocTopicCounts[subTopic] + alphas[subTopic])
						 * ((double) typeTopicCounts[type][subTopic] + test_typeTopicCounts[type][subTopic] + beta) / 
						 ((double) tokensPerTopic[subTopic] + test_tokensPerTopic[subTopic] + vBeta);
					
					cumulativeWeight += samplingWeights[i]; 
				}
				//sampling a sub topic
				sampled_topic_index = r.nextDiscreteWithSize(samplingWeights, currentNode.numSubTopics, cumulativeWeight);
				//get the topic ID
				sampled_topic = currentNode.subNodes.get(sampled_topic_index).labelIndex;				
				//add it into path
				test_topic_paths[si][topic_in_path_index] = sampled_topic;
				topic_in_path_index++;
				currentNode = currentNode.subNodes.get(sampled_topic_index);
			}
			//put a stop sign
			test_topic_paths[si][topic_in_path_index] = -1;
			
			// Put the new topics in the path
			for(int p=0;p<topic_in_path_index;p++)
			{
				test_oneDocTopicCounts[test_topic_paths[si][p]]++;						
				test_typeTopicCounts[type][test_topic_paths[si][p]]++;
				test_tokensPerTopic[test_topic_paths[si][p]]++;			
			}						
		}					
	}
	
	/*
	 * 
	 * 
	 * funtion to print the topic assignment for a word in each document
	 */
	public void printTestTopicPathforEachWord(String fname) throws IOException
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		for(int di=0;di<test_numDocs;di++)
		{
			pw.print(di);
			FeatureSequence fs = (FeatureSequence) test_ilist.get(di).getData();
			for(int si=0;si<test_topics[di].length;si++)
			{
				int type = fs.getIndexAtPosition(si);
				pw.print(" " + test_ilist.getDataAlphabet().lookupObject(type) + ":");
				for(int t=0;t<test_topics[di][si].length && test_topics[di][si][t] != -1;t++)
				{
					String topicName = topic2name.get(test_topics[di][si][t]);
					pw.print(topicName + ",");
				}
			}
			pw.print("\n");
		}
		pw.flush();
		pw.close();
	}
	
	public void printTrainTopicPathforEachWord(String fname) throws IOException
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		for(int di=0;di<numDocs;di++)
		{
			pw.print(di);
			FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
			for(int si=0;si<topics[di].length;si++)
			{
				int type = fs.getIndexAtPosition(si);
				pw.print(" " + ilist.getDataAlphabet().lookupObject(type) + ":");
				for(int t=0;t<topics[di][si].length && topics[di][si][t] != -1;t++)
				{
					String topicName = topic2name.get(topics[di][si][t]);
					pw.print(topicName + ",");
				}
			}
			pw.print("\n");
		}
		pw.flush();
		pw.close();
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
	
	/*
	 * read the topic ID to topic name file
	 */
	public void readID2name(String fname) throws IOException
	{
		topic2name = new HashMap<Integer, String>();
		
		BufferedReader br = new BufferedReader(new FileReader(fname));
		String line = "";
		while((line = br.readLine()) != null)
		{
			String [] lines = line.split(":");
			topic2name.put(Integer.parseInt(lines[0]), lines[1]);
		}
		br.close();		
	}		
	
	/*
	 * Helping function for generating a new tree where a leaf node is assigned to each intermedia node
	 * This is just an experimental code. we still do not if it works or not.
	 */
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
	
	/*
	 * read document labels from the file
	 * Important! The label set must be fully annotated (from root to the leaves).
	 */
	public void readLabels(String fileName) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String buf = "";			
		while((buf = br.readLine()) != null)
		{			
			String [] labels = buf.split(" ");
			
			TreeSet<Integer> t = new TreeSet<Integer>();			
			//ignore labels[0], the document ID
			for(int j=1;j<labels.length;j++)
			{
				t.add(Integer.parseInt(labels[j]));
			}			
			docLabels.add(t);
		}
		br.close();
	}
	
	// Serialization

	private static final long serialVersionUID = 1;
	private static final int CURRENT_SERIAL_VERSION = 0;
	private static final int NULL_INTEGER = -1;
	
	private void writeObject (ObjectOutputStream out) throws IOException {
		out.writeInt (CURRENT_SERIAL_VERSION);		
		out.writeInt (numTopics);
		out.writeInt (numTypes);
		out.writeInt (numTokens);
		out.writeInt (numDocs);
		//write alpha
		out.writeDouble (alphaSum);
		for(int i=0;i<alphas.length;i++)
			out.writeDouble(alphas[i]);
		//write beta
		out.writeDouble (beta);		
		out.writeDouble (vBeta);
		//write topic state for each position in each document
		for (int di = 0; di < numDocs; di ++)
		{
			out.writeInt(topics[di].length);
			for (int si = 0; si < topics[di].length; si++)
			{
				//write state length
				out.writeInt(topics[di][si].length);
				for(int t=0;t<topics[di][si].length;t++)				
					out.writeInt (topics[di][si][t]);
			}
		}
		//write <document, topic> count
		for (int di = 0; di < numDocs; di ++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (docTopicCounts[di][ti]);
		//write <word, topic> count
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (typeTopicCounts[fi][ti]);
		//write <topic> count
		for (int ti = 0; ti < numTopics; ti++)
			out.writeInt (tokensPerTopic[ti]);
	}

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {	
		in.readInt ();		
		numTopics = in.readInt();
		numTypes = in.readInt();
		numTokens = in.readInt();
		numDocs = in.readInt();		
		//read alpha
		alphaSum = in.readDouble();
		alphas = new double[numTopics];		
		for(int i=0;i<alphas.length;i++)
			alphas[i] = in.readDouble();
		//read beta
		beta = in.readDouble();
		vBeta = in.readDouble();
		//read topic state
		topics = new int[numDocs][][];
		for (int di = 0; di < numDocs; di++) 
		{
			int docLen = in.readInt();
			topics[di] = new int[docLen][];
			for (int si = 0; si < docLen; si++)
			{
				int stateLen = in.readInt();
				topics[di][si] = new int[stateLen];
				for(int ti=0;ti<stateLen;ti++)
					topics[di][si][ti] = in.readInt();
			}
		}
		docTopicCounts = new int[numDocs][numTopics];
		for (int di = 0; di < numDocs; di++)
			for (int ti = 0; ti < numTopics; ti++)
				docTopicCounts[di][ti] = in.readInt();		
		typeTopicCounts = new int[numTypes][numTopics];
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				typeTopicCounts[fi][ti] = in.readInt();
		tokensPerTopic = new int[numTopics];
		for (int ti = 0; ti < numTopics; ti++)
			tokensPerTopic[ti] = in.readInt();
	}
		
	public int [] getMostCommonPath(int type, Randoms r)
	{
		int [] commonPath = new int [num_tree_depth + 1];
		int topicIndex = 0;
		
		TopicNode curNode = root;
		while(curNode.numSubTopics != 0)
		{
			//find most common topic under this node
			//just check <word, topic> matrix given the current word
			int maxV = -1;
			int maxIndex = -1;
			for(int i=0;i<curNode.numSubTopics;i++)
			{
				int topic = curNode.subNodes.get(i).labelIndex;
				if(typeTopicCounts[type][topic] > maxV)
				{
					maxV = typeTopicCounts[type][topic];
					maxIndex = i;
				}
			}
			int commonTopic = -1;
			if(maxIndex == -1)
			{				
				//random sample a topic
				maxIndex = (int)(r.nextDouble() * curNode.numSubTopics);
				if(maxIndex >= curNode.numSubTopics)
					maxIndex = curNode.numSubTopics - 1;					
			}
			commonTopic = curNode.subNodes.get(maxIndex).labelIndex;
			curNode = curNode.subNodes.get(maxIndex);
			
			commonPath[topicIndex] = commonTopic;
			topicIndex++;
		}
		commonPath[topicIndex] = -1;
		return commonPath;
	}		
	
	/*
	 * 
	 * normalize the topic distribution under the root node
	 */
	public void computeDocumentTopicDistribution(int withPrior, TopicNode root, int di, int [][] test_docTopicCounts)
	{
		//compmute the weight sum of all children
		double sum = 0;
		int topic = 0;
		for(int i=0;i<root.numSubTopics;i++)
		{
			topic = root.subNodes.get(i).labelIndex;
			if(withPrior == 1)
				sum += test_docTopicCounts[di][topic] + alphas[topic];
			else
				sum += test_docTopicCounts[di][topic];
		}
		for(int i=0;i<root.numSubTopics;i++)
		{
			topic = root.subNodes.get(i).labelIndex;
			if(withPrior == 1)
				test_docTopicDistribution[di][topic] = ((double)test_docTopicCounts[di][topic] + alphas[topic]) / sum;
			else
				test_docTopicDistribution[di][topic] = ((double)test_docTopicCounts[di][topic]) / sum;
		}
		
		for(int i=0;i<root.numSubTopics;i++)
			computeDocumentTopicDistribution(withPrior, root.subNodes.get(i), di, test_docTopicCounts);
	}
	
	/*
	 * normalize the topic distribution under the root node
	 */
	public void computeTestDocumenTopicDistribution(int withPrior, TopicNode root, int test_numDocs, int [][] test_docTopicCounts)
	{
		//create document topic distribution
		test_docTopicDistribution = new double[test_numDocs][numTopics];
		for(int di=0;di<test_numDocs;di++)		
			test_docTopicDistribution[di] = new double[numTopics];			
		
		//get document topic distribution
		for(int di=0;di<test_numDocs;di++)
			computeDocumentTopicDistribution(withPrior, root, di, test_docTopicCounts);
	}
	
	/*
	 * get the maximal topic per level
	 */
	public void getHardMaximalTopicClassification(int di, TopicNode root, ArrayList<Integer> predicted_topics)
	{
		int topic = 0;
		//get the maximal topic
		double maxValue = -1;
		int maxIndex = -1;
		if(root.numSubTopics == 0)
			return;
		
		for(int i=0;i<root.numSubTopics;i++)
		{
			topic = root.subNodes.get(i).labelIndex;
			if(test_docTopicDistribution[di][topic] >= maxValue)
			{
				maxValue = test_docTopicDistribution[di][topic];
				maxIndex = i;			
			}
		}
		predicted_topics.add(root.subNodes.get(maxIndex).labelIndex);			
		getHardMaximalTopicClassification(di, root.subNodes.get(maxIndex), predicted_topics);
	}
	
	/*
	 * get the top-k topic per level
	 */
	public void getHardTopkTopicClassification(int topk, int di, TopicNode root, ArrayList<Integer> predicted_topics)
	{
		int topic = 0;
		if(root.numSubTopics == 0)
			return;
		
		//sort the topic per level by distribution
		IDSorter[] wp = new IDSorter[root.numSubTopics];
		for(int i=0;i<root.numSubTopics;i++)
		{
			topic = root.subNodes.get(i).labelIndex;
			wp[i] = new IDSorter(i, test_docTopicDistribution[di][topic]);			
		}
		Arrays.sort (wp);
		//setup the top k distribution
		for(int i=0;i < topk && i<root.numSubTopics;i++)
			predicted_topics.add(root.subNodes.get(wp[i].wi).labelIndex);
		
		for(int i=0;i<topk && i<root.numSubTopics;i++)
			getHardTopkTopicClassification(topk, di, root.subNodes.get(wp[i].wi), predicted_topics);		
	}
		
	/*
	 * get the topic per level by threading
	 */
	public void getHardThreadingTopicClassification(int di, TopicNode root, ArrayList<Integer> predicted_topics)
	{
		int topic = 0;	
		for(int i=0;i<root.numSubTopics;i++)
		{
			topic = root.subNodes.get(i).labelIndex;
			if(test_docTopicDistribution[di][topic] >= topic_thresholds[topic])
			{
				predicted_topics.add(topic);
				getHardThreadingTopicClassification(di, root.subNodes.get(i), predicted_topics);
			}
		}
	}
	
	public void outputHardClassification(TopicNode root, String method, int method_parametr, int test_numDocs, String distributionsFile) throws FileNotFoundException
	{		
		PrintWriter out = new PrintWriter(distributionsFile);		
		for(int di = 0; di < test_numDocs; di ++)
		{
			ArrayList<Integer> predicted_topics = new ArrayList<Integer>();
			
			if(method.equalsIgnoreCase("threshold") == true)
				getHardThreadingTopicClassification(di, root, predicted_topics);
			else if(method.equalsIgnoreCase("maximal") == true)
				getHardMaximalTopicClassification(di, root, predicted_topics);
			else if(method.equalsIgnoreCase("topk") == true)
				getHardTopkTopicClassification(method_parametr, di, root, predicted_topics);
			
			out.print (di); 		

			for (int i=0;i<predicted_topics.size();i++) {
                out.print(" " + predicted_topics.get(i));
            }			
            out.print (" \n");			
		}
		out.flush();
		out.close();
	}		
	
	public void setUpPredictionThreshold(double threshold)
	{
		topic_thresholds = new double [numTopics];
		for(int i=0;i<numTopics;i++)
			topic_thresholds[i] = threshold;
	}
	
	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException, ClassNotFoundException
	{
		args = new String [15];
		//args[0] = "/home/xiao/test_mallet/topic_input.mallet";
		//args[0] = "/home/xiao/datasets/software/sourceforge/sourceforge_mallet.mallet";
		//args[1] = "/home/xiao/workspace/test/sf_hier_par_child_pairs_first_level.txt";
		//args[1] = "/home/xiao/workspace/test/sf_hier_par_child_pairs.txt";
		//args[2] = "/home/xiao/datasets/software/sourceforge/sf_topics_id_string.txt";
		//args[3] = "sourceforge";
		//RCV1
		args[0] = "/home/xiao/datasets/rcv1/mallet/lyrl2004_tokens_train.mallet.new.mallet";
		args[1] = "/home/xiao/datasets/rcv1/mallet/rcv1_train_id_labels.txt.new";
		args[2] = "/home/xiao/datasets/rcv1/mallet/rcv1_hierarchy.par_child";
		//args[2] = "/home/xiao/datasets/rcv1/mallet/rcv1_flat_hierarchy.par_child";
		args[3] = "/home/xiao/datasets/rcv1/mallet/rcv1_id2cat.map";		
		args[4] = "/home/xiao/datasets/rcv1/mallet/rcv1_model.mallet";
		args[5] = "rcv1";
		args[6] = "10";
		args[7] = "infer";
		args[8] = "/home/xiao/workspace/test/rcv1/rcv1_inference";
		args[9] = "0.25";	
		args[10] = "topk";//threshold, maximal. topk
		args[11] = "1";
		args[12] = "/home/xiao/datasets/rcv1/mallet/rcv1_hierarchy.par_child";
		//args[12] = "/home/xiao/datasets/rcv1/mallet/rcv1_flat_hierarchy.par_child";
		args[13] = "no_prior";
		
		//on sourceforge dataset
		args[0] = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.mallet";
		args[1] = "/home/xiao/workspace/enrich/sourceforge/sf_labels.txt";
		args[2] = "/home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt";
		//args[2] = "/home/xiao/datasets/rcv1/mallet/rcv1_flat_hierarchy.par_child";
		args[3] = "/home/xiao/datasets/OSS_data/sourceforge/sourceforge_id2cat.map";		
		args[4] = "/home/xiao/datasets/OSS_data/sourceforge/sourceforge_model.mallet";
		args[5] = "sourceforge";
		args[6] = "300";
		args[7] = "train";
		args[8] = "/home/xiao/workspace/test/sourceforge/sourceforge_inference";
		args[9] = "0.25";	
		args[10] = "topk";//threshold, maximal. topk
		args[11] = "1";
		args[12] = "/home/xiao/datasets/OSS_data/sf_pairwise_hierarchy.txt";
		//args[12] = "/home/xiao/datasets/rcv1/mallet/rcv1_flat_hierarchy.par_child";
		args[13] = "no_prior";
		args[14] = "/home/xiao/datasets/OSS_data/sourceforge/sourceforge_topic_for_word_topic";
		
		int numTopWords = 50;				
		
		double alphaSum = 50;				
		
		String instanceFname = args[0];
		String labelFname = args[1];
		String hierFname = args[2];
		String topic2nameFname = args[3];
		String finalModelFname = args[4];
		String topTopicWordFname = "/home/xiao/workspace/test/"+args[5]+"/top_words";
		int numIterations = Integer.parseInt(args[6]);
		String inferFname = args[8];
		String method = args[7];
		double prediction_threshold = Double.parseDouble(args[9]);
		String hardClassificationMethod = args[10];
		int methodParameter = Integer.parseInt(args[11]);
		int withPrior = (args[13].equalsIgnoreCase("prior"))?1:0;
		String wordTopicFname = args[14];
		//new topic hierarchy where some new nodes are added into the hierarchy
		//String newhierFname = "/home/xiao/workspace/test/sf_hier_par_child_leaf_on_intermedia_pairs.txt";								
				
		if(method.equalsIgnoreCase("infer") == false)
		{
			PAMHierarchicalSamplingLabeled pam = new PAMHierarchicalSamplingLabeled ();
			
			pam.readID2name(topic2nameFname);
					
			pam.initTree(hierFname);
			
			pam.initAlpha(alphaSum);
			
			//pam.generateTreeWithLeafOnEachIntermediateNode(hierFname, newhierFname);
			
			pam.topTopicWordFname = topTopicWordFname;
						
			//read instances		
			InstanceList ilist = InstanceList.load (new File(instanceFname));
			//read instance labels
			pam.readLabels(labelFname);
			//estimate parameters
			System.out.println ("Data loaded.");
			pam.estimate (ilist, numIterations, 50, 10, 50, null, numTopWords, new Randoms());  // should be 1100		
			
			pam.printTrainTopicPathforEachWord(wordTopicFname + ".train");
			
			//pam SAVE object
			try {
	
				ObjectOutputStream oos = 
					new ObjectOutputStream(new FileOutputStream(finalModelFname));
				oos.writeObject(pam);
				oos.close();
			} catch (Exception e) {
				System.err.println(e.getMessage());
			}
		}
		else			
		{
			/*
			 * The test dataset does not need to have the same vocabulary as the training dataset			 
			 */
			
			//build the hierarchy tree
			TopicNode hierTree = new TopicNode();
			hierTree = TopicNode.buildHierarchy(args[12]);
			
			//read pam object		
			PAMHierarchicalSamplingLabeled pui = null;
	
			ObjectInputStream ois = new ObjectInputStream (new FileInputStream(finalModelFname));
			pui = (PAMHierarchicalSamplingLabeled) ois.readObject();
			ois.close();
			
			pui.readID2name(topic2nameFname);
			
			pui.initTree(hierFname);						
				
			pui.topTopicWordFname = topTopicWordFname;
			
			pui.setUpPredictionThreshold(prediction_threshold);
			
			//read instances		
			InstanceList ilist = InstanceList.load (new File(instanceFname));
			
			pui.infer_topics(ilist, numIterations, 50, 10, 50, null, numTopWords, new Randoms());						
			
			//pui.printTrainTopicPathforEachWord(wordTopicFname + ".train");
			pui.printTestTopicPathforEachWord(wordTopicFname + ".test");
			
			//output topics for each document
			pui.computeTestDocumenTopicDistribution(withPrior, hierTree, pui.numDocs, pui.docTopicCounts);			
			pui.outputHardClassification(hierTree, hardClassificationMethod, methodParameter, pui.numDocs, inferFname + ".train");
			
			//output topics for each document
			pui.computeTestDocumenTopicDistribution(withPrior, hierTree, pui.test_numDocs, pui.test_docTopicCounts);			
			pui.outputHardClassification(hierTree, hardClassificationMethod, methodParameter, pui.test_numDocs, inferFname + ".test");
		}
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
		
		public int getID() {return wi;}
		public double getWeight() {return p;}

		/** Reinitialize an IDSorter */
		public void set(int id, double p) { this.wi = id; this.p = p; }
		
	}
}
