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

public class PAMHierarchicalSampingPath {
	
	//for tree structure
	TopicNode root = new TopicNode();
	//<topic_id, parent_topic_id> map
	HashMap<Integer, Integer> parents = new HashMap<Integer, Integer>();
	//<topic_id, topic_node> map
	HashMap<Integer, TopicNode> topic2node = new HashMap<Integer, TopicNode>();
	
	double [] alphas;
	double beta;   // Prior on per-topic multinomial distribution over words
	double vBeta; 

	// Data
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence
	int numTypes; //number of distinct words in dataset
	int numTokens; //total number of words
	int numDocs; //number of examples
	
	int numTopics; //number of topics in the given hierarchy
	int [] allLeafNodes; //all topics on the leaf nodes
	// Gibbs sampling state
	//  (these could be shorts, or we could encode both in one int)
	// indexed by <document index, sequence index>
	// contains the topic path for the word position i in document d, i.e., <d, i>
	int [][][] topics;
	
	//precomputed path
	int [][] paths;
	
	double[] subAlphaSums;
	
	//wroking variables for a topic path
	// Per-document state variables
	int[][] superSubCounts; // # of words per <super, sub>. P(t_c|t_p)
	int [] topicCounts; //# of words per <topic>
		
	int[] superCounts; // # of words per <super>
	double[] superWeights; // the component of the Gibbs update that depends on super-topics
	double[] subWeights;   // the component of the Gibbs update that depends on sub-topics
	double[][] superSubWeights; // unnormalized sampling distribution
	double[][] cumulativeSuperWeights; // a cache of the cumulative weight for each super-topic

	//for p(w|leaf_topic) distribution
	// Per-word type state variables
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	
	
	int[][] typeSubTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerSubTopic; // indexed by <topic index>

	// [for debugging purposes]			
	int[] tokensPerTopic; // indexed by <topic index>
	int[] tokensPerSuperTopic; // indexed by <topic index>
	int[][] tokensPerSuperSubTopic;

	// Histograms for MLE
	int[][] superTopicHistograms; // histogram of # of words per supertopic in documents
	//  eg, [17][4] is # of docs with 4 words in sT 17...
	int[][][] subTopicHistograms; // for each supertopic, histogram of # of words per subtopic

	Runtime runtime;
	NumberFormat formatter;

	public PAMHierarchicalSampingPath (int superTopics, int subTopics) {
		this (superTopics, subTopics, 50.0, 0.001);
	}

	public PAMHierarchicalSampingPath (int superTopics, int subTopics,
	              double alphaSum, double beta) {
		formatter = NumberFormat.getInstance();
		formatter.setMaximumFractionDigits(5);
		
		this.beta = beta; // We can't calculate vBeta until we know how many word types...

		runtime = Runtime.getRuntime();
	}

	public void estimate (InstanceList documents, int numIterations, int optimizeInterval, 
	                      int showTopicsInterval,
	                      int outputModelInterval, String outputModelFilename,
	                      Randoms r)
	{
		ilist = documents;
		
		//get number of words
		numTypes = ilist.getDataAlphabet().size ();
		
		vBeta = beta * numTypes;
		
		//get number of documents
		numDocs = ilist.size();
		
		numTopics = root.getTreeSize() - 1;
		
		//get parents hashmap of this tree
		root.getParents(parents);
		
		root.getNodes(topic2node);
		
		topics = new int [numDocs][][];		
		
		//get all alphas
		alphas = new double[numTopics];
		root.getAllAlpha(alphas);
		
		//get all leaf nodes
		ArrayList<Integer> leaves = new ArrayList<Integer>();
		root.getAllLeaves(leaves);
		allLeafNodes = new int [leaves.size()];
		//make a path from root to each leaf
		paths = new int [numTopics][];				
		for(int i=0;i<leaves.size();i++)
		{
			allLeafNodes[i] = leaves.get(i);
			
			//make a path from root to this leaf		
			ArrayList<Integer> a_path = new ArrayList<Integer>();
			int cur_topic = allLeafNodes[i];
			while(cur_topic != -1)
			{
				a_path.add(cur_topic);
				cur_topic = parents.get(cur_topic);
			}
			int [] root_to_leaf_path = new int [a_path.size()];
			for(int j=0; j < root_to_leaf_path.length; j++)
			{
				root_to_leaf_path[j] = a_path.get(root_to_leaf_path.length - j);
			}
			paths[allLeafNodes[i]] = root_to_leaf_path;
		}
		
		subAlphaSums = new double[root.numSubTopics];
		//just any value, such as the first level size
		Arrays.fill(subAlphaSums, root.numSubTopics);
		
		//		Allocate several arrays for use within each document
		//		to cut down memory allocation and garbage collection time

		superSubCounts = new int[numTopics][numTopics];//count for <super topic, sub topic>
		topicCounts = new int [numTopics];//count for each topic
		superWeights = new double[numTopics];
		subWeights = new double[numTopics];
		superSubWeights = new double[numTopics][numTopics];//weight for <super topic, sub topic>
		cumulativeSuperWeights = new double[numTopics][];
		
		typeSubTopicCounts = new int[numTypes][numTopics];//count for <word, topic>
		tokensPerSubTopic = new int[numTopics];
		tokensPerSuperTopic = new int[numTopics];
		tokensPerSuperSubTopic = new int[numTopics][numTopics];//count for words of <super topic, sub topic>
		
		/*
		superSubCounts = new int[numSuperTopics][numSubTopics];
		superCounts = new int[numSuperTopics];
		superWeights = new double[numSuperTopics];
		subWeights = new double[numSubTopics];
		superSubWeights = new double[numSuperTopics][numSubTopics];
		cumulativeSuperWeights = new double[numSuperTopics];

		typeSubTopicCounts = new int[numTypes][numSubTopics];
		tokensPerSubTopic = new int[numSubTopics];
		tokensPerSuperTopic = new int[numSuperTopics];
		tokensPerSuperSubTopic = new int[numSuperTopics][numSubTopics];
		
		*/

		long startTime = System.currentTimeMillis();

		int maxTokens = 0;

		//		Initialize with random assignments of tokens to topics
		//		and finish allocating this.topics and this.tokens

		int superTopic, subTopic, seqLen;

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
				topics[di][si] = path;
				//update the token count for each topic
				for(int p=0;p<path.length;p++)
					tokensPerTopic[path[p]] ++;							
				
				//get the topic at leaf
				int leafTopic = path[path.length - 1];
				
				// For the leaf-topic, we also need to update the 
				//  word type statistics
				typeSubTopicCounts[fs.getIndexAtPosition(si) ][leafTopic]++;
				tokensPerSubTopic[leafTopic]++;
				
				//for each edge of this path, update the count
				for(int i=0;i<path.length-1;i++)
					tokensPerSuperSubTopic[path[i]][path[i+1]]++;
			}
		}

		System.out.println("max tokens: " + maxTokens);

		//		These will be initialized at the first call to 
		clearHistograms(); //in the loop below.

		
		superTopicHistograms = new int[numTopics][maxTokens + 1];
		subTopicHistograms = new int[numTopics][numTopics][maxTokens + 1];
		
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
				}
				if (outputModelInterval != 0 && iterations % outputModelInterval == 0) {
					//this.write (new File(outputModelFilename+'.'+iterations));
				}				
				
				if (optimizeInterval != 0 && iterations % optimizeInterval == 0) {
					long optimizeTime = System.currentTimeMillis();
					
					//update the parameter on the tree
					for(int k=0;k<root.numSubTopics;k++)
						learnParameterOnTree(root.subNodes.get(k));
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
					System.out.print("[o:" + (System.currentTimeMillis() - optimizeTime) + "]");
				}	
							
			}

			if (iterations > 1107) {
				//printWordCounts();
			}

			if (iterations % 10 == 0)
				System.out.println ("<" + iterations + "> ");

			System.out.print((System.currentTimeMillis() - iterationStart) + " ");

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
					topics[di], r);
		}
	}

	
	
	private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens,
	                                    int[][] topic_paths, // indexed by seq position, a sampled topic path for the position	                                    
	                                    Randoms r) {

//		long startTime = System.currentTimeMillis();

		int[] currentTypeSubTopicCounts;
		int[] currentSuperSubCounts;
		double[] currentSubAlpha;

		int type, subTopic, superTopic;
		double cumulativeWeight, sample;

		int docLen = oneDocTokens.getLength();

		for (int t = 0; t < numTopics; t++) {
			Arrays.fill(superSubCounts[t], 0);
		}

		Arrays.fill(superCounts, 0);

//		populate topic counts for the edges of the path
		for (int si = 0; si < docLen; si++) {
			//get topic path for this position
			int [] path = topic_paths[si];
			//accumulate the p(t_c|t_p) as count(t_c, t_p)
			//and count(t_p)
			for(int p=0;p<path.length-1;p++)
			{
				superSubCounts[ path[p] ][path[p+1] ]++;
				superCounts[ path[p] ]++;
			}
		}

//		Iterate over the positions (words) in the document

		for (int si = 0; si < docLen; si++) {
			//get word at this position
			type = oneDocTokens.getIndexAtPosition(si);
			
			//get topic path of this position
			int [] path = topic_paths[si];
						
			int leafTopic = path[path.length - 1];

			// Remove this token from all counts among the path
			for(int p=0;p<path.length-1;p++)
			{
				superTopic = path[p];
				subTopic = path[p+1];
				
				superSubCounts[superTopic][subTopic]--;
				superCounts[superTopic]--;				
				
				tokensPerSuperTopic[superTopic]--;				
				tokensPerSuperSubTopic[superTopic][subTopic]--;
			}
			//remove this token from the leaf node
			typeSubTopicCounts[type][leafTopic]--;
			tokensPerSubTopic[leafTopic]--;
			
			
			// Build a distribution over super-sub topic pairs 
			//   for this token			
			Arrays.fill(superWeights, 0.0);
			Arrays.fill(subWeights, 0.0);					

			// The conditional probability of each super-sub pair is proportional
			//  to an expression with three parts, one that depends only on the 
			//  super-topic, one that depends only on the sub-topic and the word type,
			//  and one that depends on the super-sub pair.			
			
			
			//calculate each first level only factors			
			for (int index = 0; index < root.numSubTopics; index++) {
				//get fist-level topic ID
				superTopic = root.subNodes.get(index).labelIndex;
				//update first level-topic weight
				superWeights[superTopic] = ((double) superCounts[superTopic] + alphas[superTopic]) /
				((double) superCounts[superTopic] + subAlphaSums[superTopic]);
			}
			
						
			// Next calculate the leaf only factors
			for (int leafTopicIndex = 0; leafTopicIndex < allLeafNodes.length; leafTopicIndex++) {
				//get leaf topic ID
				leafTopic = allLeafNodes[leafTopicIndex];
				//update leaf topic weight
				subWeights[leafTopic] = ((double) typeSubTopicCounts[type][leafTopic] + beta) / 
				((double) tokensPerSubTopic[leafTopic] + vBeta);
			}

			// Finally, put them together

			cumulativeWeight = 0.0;
			
			//for each leaf node, make a path
			//the probability for a path z=<k1, k2, k3, k4, ..., kl> given word t
			//is p(k1)\prod_{j=1}^{l-1} P(k_{j+1}|k_{j})p(t|k_l)
			//so we first estimate p(k1) and p(t|k_l)
			//then we multiple the conditional probability from k1 to kl
			double [] leaf_weight = new double [allLeafNodes.length];
			Arrays.fill(leaf_weight, 1.0);
			for (int leafTopicIndex = 0; leafTopicIndex < allLeafNodes.length; leafTopicIndex++) 
			{
				//get the path from root to this leaf
				int leaf_topic = allLeafNodes[leafTopicIndex];
				int [] root_to_leaf_path = paths[leaf_topic];								
				int first_level_topic = root_to_leaf_path[0];				
				
				//then multiple the first-level weight
				leaf_weight[leafTopicIndex] *= superWeights[first_level_topic];
				
				//we multiple the conditional probability from k1 to kl
				//accumulate p(t_c|t_p)
				for(int i=0;i<root_to_leaf_path.length - 1;i++)
				{
					superTopic = root_to_leaf_path[i];
					subTopic = root_to_leaf_path[i+1];										
					
					leaf_weight[leafTopicIndex] *= 
						((double) superSubCounts[superTopic][subTopic] + alphas[subTopic]) /
						(superCounts[superTopic] + subAlphaSums[superTopic]) ;									
				}	
				//then multiple the leaf weight
				leaf_weight[leafTopicIndex] *= subWeights[leaf_topic];
				
				cumulativeWeight += leaf_weight[leafTopicIndex];
			}	
									
			//sampling a pth
			int leafTopicIndex = r.nextDiscrete (leaf_weight, cumulativeWeight);
			
			// Save the choice into the Gibbs state		
			topic_paths[si] = paths[leafTopicIndex];
			
			// Put the new <super, sub> pair into the counts
			leafTopic = paths[leafTopicIndex][paths[leafTopicIndex].length-1];
			for(int i=0;i<paths[leafTopicIndex].length-1;i++)
			{
				superTopic = paths[leafTopicIndex][i];
				subTopic = paths[leafTopicIndex][i+1];
				
				superSubCounts[superTopic][subTopic]++;
				superCounts[superTopic]++;
				
				tokensPerSuperTopic[superTopic]++;				
				tokensPerSuperSubTopic[superTopic][subTopic]++;
			}	
			typeSubTopicCounts[type][leafTopic]++;
			tokensPerSubTopic[leafTopic]++;
		}

		//		Update the topic count histograms
		//		for dirichlet estimation
		//for each leaf node (a path from root to leaf)
		for (int leafTopicIndex = 0; leafTopicIndex < allLeafNodes.length; leafTopicIndex++) 
		{
			//get the path from root to this leaf
			int leaf_topic = allLeafNodes[leafTopicIndex];
			int [] root_to_leaf_path = paths[leaf_topic];													
			
			for(int i=0;i<root_to_leaf_path.length-1;i++)
			{
				superTopic = root_to_leaf_path[i];
				subTopic = root_to_leaf_path[i+1];
				
				superTopicHistograms[superTopic][ superCounts[superTopic] ]++;
				currentSuperSubCounts = superSubCounts[superTopic];
				subTopicHistograms[superTopic][subTopic][ currentSuperSubCounts[subTopic] ]++;
			}
		}				
	}

	
	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException
	{
		InstanceList ilist = InstanceList.load (new File(args[0]));
		int numIterations = args.length > 1 ? Integer.parseInt(args[1]) : 1000;
		int numTopWords = args.length > 2 ? Integer.parseInt(args[2]) : 20;
		int numSuperTopics = args.length > 3 ? Integer.parseInt(args[3]) : 10;
		int numSubTopics = args.length > 4 ? Integer.parseInt(args[4]) : 10;
		System.out.println ("Data loaded.");
		
		double alphaSum = 50;
		
		PAMHierarchicalSampingPath pam = new PAMHierarchicalSampingPath (numSuperTopics, numSubTopics);
		
		//build topic hierarchy
		String hierFname = "/home/xiao/workspace/test/sf_hier_par_child_pairs.txt";		
		pam.root = TopicNode.buildHierarchy(hierFname);
		pam.root.initTree(alphaSum);
		
		pam.estimate (ilist, numIterations, 50, 0, 50, null, new Randoms());  // should be 1100
		//pam.printTopWords (numTopWords, true);
//		pam.printDocumentTopics (new File(args[0]+".pam"));
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
