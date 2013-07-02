package seeu;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.TreeSet;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

import seeu.utils.Hierarchy;

public class HierarchicalLabeledLDA {
		
	Hierarchy tree = new Hierarchy();
	
	int numTopics; // Number of topics to be fit
	double alpha;  // Dirichlet(alpha,alpha,...) is the distribution over topics
	double beta;   // Prior on per-topic multinomial distribution over words
	double tAlpha;
	double vBeta;
	
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence
	InstanceList llist;  // the data field of the instances is expected to hold a label list
	
	Alphabet text_vocabulary = null;
	Alphabet label_vocabulary = null;
		
	//
	//variables for training text
	//
	
	int[][][] topics; // indexed by <document index, sequence index>, we assume each position can only be assigned one path
	int numTypes;//total number of **unique** terms
	int numTokens;//total number of terms in all documents
	
	int numDocs;//number of text docs
	int numLabelDocs;//number of label instances
	
	int[][] docTopicCounts; // indexed by <document index, topic index>
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>	
	int [][] parent2childTopicCounts; // indexed by <parent topic index, child topic index>
	
	//
	//this is the pseudo added nodes
	//
	int [] docExitTopicCounts; // indexed by <document index, internal topic index>
	int [][] typeExitInternalTopicCounts;//indexed by <feature index, internal topic index>
	int [] exitInternalTopicCounts;//indexed by <internal topic index>		
	
	//
	//variables for training labels
	//
	int [][] docLabels;//indexed by <document index, label index>	
	ArrayList<TreeSet<Integer>> docSetLabels = new ArrayList<TreeSet<Integer>>();
	
	int numLabels;//total number of **unique** labels
	Hashtable<Integer, String> id2labels = new Hashtable<Integer, String>();			
		
	//
	//variables for testing
	//
	
	InstanceList test_ilist;  // the data field of the instances is expected to hold a FeatureSequence
	
	int[][] test_topics; // indexed by <document index, sequence index>	
	int test_numTypes;//total number of terms in all documents
	int test_numTokens;//total number of terms in all documents
	
	int test_numDocs;//number of text docs	
	
	int[][] test_docTopicCounts; // indexed by <document index, topic index>
	int[][] test_typeTopicCounts; // indexed by <feature index, topic index>
	int[] test_tokensPerTopic; // indexed by <topic index>	
	
	double [] buf_sampling_weights;
	
	double block_threshold = 0.5;
	
	public HierarchicalLabeledLDA (int numberOfTopics)
	{
		this (numberOfTopics, 50.0, 0.01);
	}

	public HierarchicalLabeledLDA ()
	{		
	}
	
	public HierarchicalLabeledLDA (int numberOfTopics, double alphaSum, double beta)
	{
		this.numTopics = numberOfTopics;
		this.alpha = alphaSum / numTopics;
		this.beta = beta;
	}

	public void init_hierarchy(Hierarchy hier)
	{
		tree = hier;
		
		//parent2childTopicCounts
		//init parent topic to child topic
		//<parent, <0, child node>
		int [][] parent2childnode = tree.parent2children;
		int [] parentNode = tree.parents;
		
		parent2childTopicCounts = new int [parent2childnode.length][];
		exitInternalTopicCounts = new int [parent2childnode.length];
		for(int i=0;i<parent2childnode.length;i++)
		{
			if(parent2childnode[i] != null)
			{
				//Nodes: [n1, n2, n3, ..., nC]
				//Count: [v1, v2, v3, ..., vC, 0]
				parent2childTopicCounts[i] = new int[parent2childnode[i].length + 1];
				for(int j=0;j<parent2childnode[i].length;j++)
					parent2childTopicCounts[i][j] = 0;
				
				//this is a virtual node
				exitInternalTopicCounts[i] = 0;
			}
		}
	}
	
	public int sampledAChildTopic(int cur_node, Randoms r, int [] topicCountInThisType, double [] weights)
	{
		int [] children = tree.parent2children[cur_node];		
		int parentCount = topicCountInThisType[cur_node];
		int i;
		double sm = 0;
		for(i=0;i<children.length;i++)
		{
			int cti = children[i];
			weights[i] = (double)topicCountInThisType[cti] / parentCount;	
			sm += weights[i]; 
		}
		weights[i+1] = (double)exitInternalTopicCounts[cur_node] / parentCount;
		sm += weights[i+1]; 
		int cti_index = r.nextDiscreteFirstK(weights, children.length + 1, sm);
		if(cti_index == children.length)
			return -1;//exit node
		else
			return children[cti_index];
	}
	
	public int sampledAChildTopic(int cur_node, //current node 
			Randoms r, 
			int [] topicCountInThisType, //topic count for this word
			TreeSet<Integer> labels, //labels for this document
			double [] weights)
	{
		int [] children = tree.parent2children[cur_node];		
		int parentCount = topicCountInThisType[cur_node];
		int i=0, j=0;
		double sm = 0;
		for(i=0;i<children.length;i++)
		{
			int cti = children[i];
			if(labels.contains(cti))
			{
				weights[j] = (double)topicCountInThisType[cti] / parentCount;	
				sm += weights[i];
				j++;
			}
		}
		weights[i+1] = (double)exitInternalTopicCounts[cur_node] / parentCount;
		sm += weights[i+1]; 
		int cti_index = r.nextDiscreteFirstK(weights, children.length + 1, sm);
		if(cti_index == children.length)
			return -1;//exit node
		else
			return children[cti_index];
	}
	
	
	
	public void sampledAPath(Randoms r, 
			int [] path, //output 
			int [] topicCountInThisType, //topic count
			TreeSet<Integer> used_nodes, //used topics
			double [] weights)
	{
		int cur_node = tree.root;
		//not null and not leaf
		int level = 0;
		int next_node = -1;
		while(tree.parent2children[cur_node] != null && tree.parent2children[cur_node].length != 0)
		{
			next_node = sampledAChildTopic(cur_node, r, topicCountInThisType, used_nodes, weights);
			if(next_node == -1)//exit node, stop
			{
				path[level] = -1;
				break;
			}
			else
			{
				path[level] = next_node;
			}
		}
	}
	
	public void init (InstanceList documents, InstanceList labels, Randoms r)
	{
		ilist = documents.shallowClone();
		numTypes = ilist.getDataAlphabet().size ();//get vocabulary size
		numDocs = ilist.size();
		
		//get vocabulary
		text_vocabulary = ilist.getDataAlphabet();
		
		llist = labels.shallowClone();
		numLabels = llist.getDataAlphabet().size();//get label vocabulary
		numLabelDocs =llist.size();
		
		if(numDocs != numLabelDocs)
		{
			System.out.println("Numer of documents and number of labels do not match");
			System.exit(1);
		}
		
		//important!! reset the number of topics as the number of labels		
		numTopics = numLabels;
		
		//decide alpha and beta
		alpha = (double)50 / numTopics;
		beta = 0.1;
		
		//read all labels
		docLabels = new int [numDocs][];
		//get all labels
		FeatureSequence FSlabels;
		for(int di=0;di<numDocs;di++)
		{
			try {
				FSlabels = (FeatureSequence) llist.get(di).getData();
		      } catch (ClassCastException e) {
		        System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
		                            +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
		        throw e;
		      }
			int numLabel = FSlabels.getLength();			
			docLabels[di] = new int[numLabel];
			// Randomly assign tokens to topics
			TreeSet<Integer> label_set = new TreeSet<Integer>();
			for (int si = 0; si < numLabel; si++) {
				docLabels[di][si] = FSlabels.getIndexAtPosition(si);
				id2labels.put(docLabels[di][si], (String)llist.getDataAlphabet().lookupObject(docLabels[di][si]));
				
				label_set.add(docLabels[di][si]);
			}
			
			docSetLabels.add(label_set);
		}
			
		//init working variables
		topics = new int[numDocs][][];
		docTopicCounts = new int[numDocs][numTopics];
		typeTopicCounts = new int[numTypes][numTopics];
		tokensPerTopic = new int[numTopics];
		tAlpha = alpha * numTopics;
		vBeta = beta * numTypes;
		
		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen;
	    FeatureSequence fs;
	    
	    TreeSet<Integer> used_nodes;
	    
	    for (int di = 0; di < numDocs; di++) {
	      try {
	        fs = (FeatureSequence) ilist.get(di).getData();
	      } catch (ClassCastException e) {
	        System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
	                            +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
	        throw e;
	      }
	        
	      	seqLen = fs.getLength();
			
	        numTokens += seqLen;
			topics[di] = new int[seqLen][];
			
			// Randomly assign tokens to a path of topics			
			for (int si = 0; si < seqLen; si++)
			{		
				//sample a topic path
				topics[di][si] = new int [tree.max_level+1];								
				
				tree.sampleAPath(r, topics[di][si], docSetLabels.get(di), block_threshold);
				
				for(int k=0;k<topics[di][si].length && topics[di][si][k] != -1;k++)
				{
					topic = topics[di][si][k];					
					docTopicCounts[di][topic]++;
					typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
					tokensPerTopic[topic]++;
				}
			}
		}    	    		
	}
	
	public void init_test (InstanceList documents, Randoms r)
	{
		test_ilist = documents.shallowClone();		
		test_numDocs = test_ilist.size();								
		
		test_topics = new int[test_numDocs][];
		test_docTopicCounts = new int[test_numDocs][numTopics];
		test_typeTopicCounts = new int[numTypes][numTopics];
		test_tokensPerTopic = new int[numTopics];

		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen;
	    FeatureSequence fs;
	    for (int di = 0; di < test_numDocs; di++) {
	      try {
	        fs = (FeatureSequence) test_ilist.get(di).getData();
	      } catch (ClassCastException e) {
	        System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
	                            +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
	        throw e;
	      }
	      seqLen = fs.getLength();
			test_numTokens += seqLen;
			test_topics[di] = new int[seqLen];
			// Randomly assign tokens to topics
			//following the paper labeled-LDA, this step is exactly the same as LDA
			for (int si = 0; si < seqLen; si++) {
				topic = r.nextInt(numTopics);
				test_topics[di][si] = topic;
				test_docTopicCounts[di][topic]++;
				test_typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				test_tokensPerTopic[topic]++;
			}			
		} 
	}
	
	public void estimate (int numIterations, int showTopicsInterval,
            int outputModelInterval, String outputModelFilename,
	            Randoms r)
	{
		this.estimate(0, numDocs, numIterations, showTopicsInterval, outputModelInterval, outputModelFilename, r);	
	}		
	
	/* Perform several rounds of Gibbs sampling on the documents in the given range. */ 
	public void estimate (int docIndexStart, int docIndexLength,
	                      int numIterations, int showTopicsInterval,
                        int outputModelInterval, String outputModelFilename,
                        Randoms r)
	{
		long startTime = System.currentTimeMillis();
		for (int iterations = 0; iterations < numIterations; iterations++) {
			if (iterations % 10 == 0) System.out.print (iterations);	else System.out.print (".");
			System.out.flush();
			if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0 && iterations > 0) {
				System.out.println ();
				printTopWords (5, false);
			}
      if (outputModelInterval != 0 && iterations % outputModelInterval == 0 && iterations > 0) {
        this.write (new File(outputModelFilename+'.'+iterations));
      }
      sampleTopicsFromLabelsForDocs(docIndexStart, docIndexLength, r);
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
	}	

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsFromLabelsForDocs (int start, int length, Randoms r)
	{
		assert (start+length <= docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start+length; di++) {
			sampleTopicsFromLabelSetForOneDoc ((FeatureSequence)ilist.get(di).getData(),
			                       topics[di], docTopicCounts[di], docLabels[di], topicWeights, r);
		}
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForTestDocs (int start, int length, Randoms r)
	{
		assert (start+length <= test_docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start+length; di++) {
			sampleTopicsForOneTestDoc ((FeatureSequence)test_ilist.get(di).getData(),
					docTopicCounts[di], test_topics[di], test_docTopicCounts[di], topicWeights, r);
		}
	}		
	
	  private void sampleTopicsForOneTestDoc (FeatureSequence oneDocTokens, 			  
	          int[] oneDocTopicCounts, // indexed by topic index
	          int[] test_oneDocTopics, // indexed by seq position
	          int[] test_oneDocTopicCounts, // indexed by topic index
	          double[] topicWeights, Randoms r)
	{
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			type = oneDocTokens.getIndexAtPosition(si);
			oldTopic = test_oneDocTopics[si];
			// Remove this token from all counts
			test_oneDocTopicCounts[oldTopic]--;
			test_typeTopicCounts[type][oldTopic]--;
			test_tokensPerTopic[oldTopic]--;
			// Build a distribution over topics for this token
			//this formula is from WWW 08 paper GibbasLDA
			Arrays.fill (topicWeights, 0.0);
			topicWeightsSum = 0;
			currentTypeTopicCounts = test_typeTopicCounts[type];
			for (int ti = 0; ti < numTopics; ti++) {
				tw = ((typeTopicCounts[type][ti] + currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + test_tokensPerTopic[ti] + vBeta))
				* ((oneDocTopicCounts[ti] + test_oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);
			
			// Put that new topic into the counts
			test_oneDocTopics[si] = newTopic;
			test_oneDocTopicCounts[newTopic]++;
			test_typeTopicCounts[type][newTopic]++;
			test_tokensPerTopic[newTopic]++;
		}
	}
  
	  private void sampleTopicsFromLabelSetForOneDoc (FeatureSequence oneDocTokens, 
			  int[][] oneDocTopics, // indexed by seq position
	          int[] oneDocTopicCounts, // indexed by topic index
	          int[] oneDocLabels, // all labels in this doc
	          double[] topicWeights, Randoms r)
	{
		int[] currentTypeTopicCounts;
		int type, newTopic, oldTopic, newTopicIndex;
		int [] oldTopics;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			//get term type
			type = oneDocTokens.getIndexAtPosition(si);
			
			//get old assigned topics
			oldTopics = oneDocTopics[si];
			
			for(int k=0;k<oldTopics.length && oldTopics[k] != -1; k++)
			{			
				oldTopic = oldTopics[k];
				// Remove this token from all counts
				oneDocTopicCounts[oldTopic]--;
				typeTopicCounts[type][oldTopic]--;
				tokensPerTopic[oldTopic]--;
			}
			
			//hierarchically sampling a new topic from root to the leaf
			
			int cur_node = tree.root;
			int sampled_node = -1;
			int level = 0;
			int [] used_children = new int [200];
			int num_used_children = 0;
			while(tree.parent2children[cur_node].length != 0 && cur_node != -1)
			{
				//get nodes at this level
				for(int ci=0;ci<tree.parent2children[cur_node].length;ci++)
				{
					int cti = tree.parent2children[cur_node][ci];
					if(this.docSetLabels.contains(cti))
					{
						used_children[num_used_children] = cti;
						num_used_children++;
					}
				}
				//add exit node
				used_children[num_used_children] = -1;
				num_used_children++;
				
				//sampling from the topics
				Arrays.fill (topicWeights, 0.0);
				topicWeightsSum = 0;
				currentTypeTopicCounts = typeTopicCounts[type];
				for(int li=0;li<oneDocLabels.length;li++)
				{
					int ti = oneDocLabels[li];
					tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
							* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
					topicWeightsSum += tw;
					topicWeights[li] = tw;
				}	
				
				cur_node = sampled_node;				
				level++;
			}
			
			// Build a distribution over topics from the training labels for this token
			Arrays.fill (topicWeights, 0.0);
			topicWeightsSum = 0;
			currentTypeTopicCounts = typeTopicCounts[type];
			for(int li=0;li<oneDocLabels.length;li++)
			{
				int ti = oneDocLabels[li];
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
						* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[li] = tw;
			}			
			/*
			for (int ti = 0; ti < numTopics; ti++) {
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
				* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
			}
			*/
			// Sample a topic assignment from this distribution
			newTopicIndex = r.nextDiscreteFirstK (topicWeights, oneDocLabels.length, topicWeightsSum);
			
			newTopic = oneDocLabels[newTopicIndex];
			
			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
			oneDocTopicCounts[newTopic]++;
			typeTopicCounts[type][newTopic]++;
			tokensPerTopic[newTopic]++;
		}
	}
  
	  /* Perform several rounds of Gibbs sampling on the documents in the given range. */ 
		public void test (int docIndexStart, int docIndexLength,
		                      int numIterations, int showTopicsInterval,	                        
	                        Randoms r)
		{
			long startTime = System.currentTimeMillis();
			for (int iterations = 0; iterations < numIterations; iterations++) {
				if (iterations % 10 == 0) System.out.print (iterations);	else System.out.print (".");
				System.out.flush();
				if (showTopicsInterval != 0 && iterations % showTopicsInterval == 0 && iterations > 0) {
					System.out.println ();
					printTopWords (5, false);
				}
	      
				sampleTopicsForTestDocs(docIndexStart, docIndexLength, r);
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
		}
		
	public int[][] getDocTopicCounts(){
		return docTopicCounts;
	}
	
	public int[][] getTypeTopicCounts(){
		return typeTopicCounts;
	}

	public int[] getTokensPerTopic(){
		return tokensPerTopic;
	}

	public void printTopWords (int numWords, boolean useNewLines)
	{
		class WordProb implements Comparable {
			int wi;
			double p;
			public WordProb (int wi, double p) { this.wi = wi; this.p = p; }
			public final int compareTo (Object o2) {
				if (p > ((WordProb)o2).p)
					return -1;
				else if (p == ((WordProb)o2).p)
					return 0;
				else return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb (wi, ((double)typeTopicCounts[wi][ti]) / tokensPerTopic[ti]);
			Arrays.sort (wp);
			if (useNewLines) {
				System.out.println ("\nTopic "+ti);
				for (int i = 0; i < numWords; i++)
					System.out.println (text_vocabulary.lookupObject(wp[i].wi).toString() + " " + wp[i].p);
			} else {
				System.out.print ("Topic "+ti+": ");
				for (int i = 0; i < numWords; i++)
					System.out.print (text_vocabulary.lookupObject(wp[i].wi).toString() + " ");
				System.out.println();
			}
		}
	}

	public void writeTopWords (String fname, int numWords, boolean useNewLines) throws IOException
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		
		class WordProb implements Comparable {
			int wi;
			double p;
			public WordProb (int wi, double p) { this.wi = wi; this.p = p; }
			public final int compareTo (Object o2) {
				if (p > ((WordProb)o2).p)
					return -1;
				else if (p == ((WordProb)o2).p)
					return 0;
				else return 1;
			}
		}

		WordProb[] wp = new WordProb[numTypes];
		for (int ti = 0; ti < numTopics; ti++) {
			for (int wi = 0; wi < numTypes; wi++)
				wp[wi] = new WordProb (wi, ((double)typeTopicCounts[wi][ti]) / tokensPerTopic[ti]);
			Arrays.sort (wp);
			if (useNewLines) {
				pw.println ("\nTopic "+ti);
				for (int i = 0; i < numWords; i++)
					pw.println (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " " + wp[i].p);
			} else {
				System.out.print ("Topic "+ti+": ");
				for (int i = 0; i < numWords; i++)
					System.out.print (text_vocabulary.lookupObject(wp[i].wi).toString() + " ");
				pw.println();
			}
		}
		
		pw.flush();
		pw.close();
	}
	
	public void writeTopicsAtPositionInDocuments(int [][] topics, String fname) throws IOException
	{
		//output topics variables
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		for(int di=0;di<topics.length;di++)
		{
			for(int si=0;si<topics[di].length-1;si++)				
				pw.print(topics[di][si] + " ");
			pw.print(topics[di][topics[di].length-1] + "\n");
		}
		pw.flush();
		pw.close();
	}
	
  public void printDocumentTopics (File f) throws IOException
  {
    printDocumentTopics (new PrintWriter (new FileWriter (f)));
  }

  public void printDocumentTopics (PrintWriter pw) {
    printDocumentTopics (pw, 0.0, -1);
  }

  public void printDocumentTopics (PrintWriter pw, double threshold, int max)
  {
    pw.println ("#doc source topic proportion ...");
    int docLen;
    double topicDist[] = new double[topics.length];
    for (int di = 0; di < topics.length; di++) {
      pw.print (di); pw.print (' ');
			if (ilist.get(di).getSource() != null){
				pw.print (ilist.get(di).getSource().toString()); 
			}
			else {
				pw.print("null-source");
			}
			pw.print (' ');
      docLen = topics[di].length;
      for (int ti = 0; ti < numTopics; ti++)
        topicDist[ti] = (((float)docTopicCounts[di][ti])/docLen);
      if (max < 0) max = numTopics;
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
        pw.print (maxindex+" "+topicDist[maxindex]+" ");
        topicDist[maxindex] = 0;
      }
      pw.println (' ');
    }
  }

  public void writeDocumentTopics (String fname, int[][] topics, int[][] docTopicCounts, double threshold, int max) throws IOException
  {
	PrintWriter pw = new PrintWriter(new FileWriter(fname));
    pw.println ("#doc source topic proportion ...");
    int docLen;
    double topicDist[] = new double[topics.length];
    for (int di = 0; di < topics.length; di++) {
      pw.print (di); pw.print (' ');			
      docLen = topics[di].length;
      for (int ti = 0; ti < numTopics; ti++)
        topicDist[ti] = (((float)docTopicCounts[di][ti])/docLen);
      if (max < 0) max = numTopics;
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
        pw.print (maxindex+" "+topicDist[maxindex]+" ");
        topicDist[maxindex] = 0;
      }
      pw.println (' ');
    }
    pw.flush();
    pw.close();
  }

  public void printState (File f) throws IOException
  {
	  PrintWriter writer = new PrintWriter (new FileWriter(f));
	  printState (writer);
	  writer.close();
  }


  public void printState (PrintWriter pw)
  {
	  Alphabet a = ilist.getDataAlphabet();
	  pw.println ("#doc pos typeindex type topic");
	  for (int di = 0; di < topics.length; di++) {
		  FeatureSequence fs = (FeatureSequence) ilist.get(di).getData();
		  for (int si = 0; si < topics[di].length; si++) {
			  int type = fs.getIndexAtPosition(si);
			  pw.print(di); pw.print(' ');
			  pw.print(si); pw.print(' ');
			  pw.print(type); pw.print(' ');
			  pw.print(a.lookupObject(type)); pw.print(' ');
			  pw.print(topics[di][si]); pw.println();
		  }
	  }
  }

  
  public void writeLabelMap(String fname) throws IOException
  {
	  PrintWriter pw = new PrintWriter (new FileWriter(fname));
	  for (Integer key: id2labels.keySet()) {
		    pw.println(key + " " + id2labels.get(key));		     
		  }	
	  pw.flush();
	  pw.close();
  }
  
  public void write (File f) {
    try {
      ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream(f));
      oos.writeObject(this);
      oos.close();
    }
    catch (IOException e) {
      System.err.println("Exception writing file " + f + ": " + e);
    }
  }
	
	public InstanceList getInstanceList ()
	{
		return ilist;
	}			
	
	public void writeModel(ObjectOutputStream out) throws IOException 
	{		
		//write count data
		out.writeInt (numDocs);
		out.writeInt (numTopics);
		out.writeInt (numLabels);
		out.writeInt (numTypes);
		out.writeInt (numTokens);
		//write LDA parameters
		out.writeDouble (alpha);
		out.writeDouble (beta);
		out.writeDouble (tAlpha);
		out.writeDouble (vBeta);
		//write assigned topic at each word position
		for (int di = 0; di < numDocs; di ++)
		{
			out.writeInt(topics[di].length);
			for (int si = 0; si < topics[di].length; si++)
				out.writeInt (topics[di][si]);
		}
		//write topic count in each document
		for (int di = 0; di < numDocs; di ++)				
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (docTopicCounts[di][ti]);		
		//write word,topic count
		for (int fi = 0; fi < numTypes; fi++)
			for (int ti = 0; ti < numTopics; ti++)
				out.writeInt (typeTopicCounts[fi][ti]);
		//write number of tokens per topic
		for (int ti = 0; ti < numTopics; ti++)
			out.writeInt (tokensPerTopic[ti]);
	}	
	
	public void readModel(ObjectInputStream in) throws IOException, ClassNotFoundException 
	{				
		//read count
		numDocs = in.readInt();
		numTopics = in.readInt();
		numLabels = in.readInt();
		numTypes = in.readInt();
		numTokens = in.readInt();
		//read LDA parameters
		alpha = in.readDouble();
		beta = in.readDouble();
		tAlpha = in.readDouble();
		vBeta = in.readDouble();
		//read important model parameters		
		topics = new int[numDocs][];
		for (int di = 0; di < numDocs; di++) {
			int docLen = in.readInt();
			topics[di] = new int[docLen];
			for (int si = 0; si < docLen; si++)
				topics[di][si] = in.readInt();
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
	
	public void writeTextVocabulary(ObjectOutputStream out) throws IOException 
	{
		//write text data vocabulary
		out.writeObject (ilist.getDataAlphabet());
	}
	
	public void readTextVocabulary(ObjectInputStream in) throws IOException, ClassNotFoundException 
	{
		//read text vocabulary
		text_vocabulary = (Alphabet) in.readObject ();
	}
	
	public void writeLabelVocabulary(ObjectOutputStream out) throws IOException 
	{
		//write label vocabulary
		out.writeObject (llist.getDataAlphabet());
	}
	
	public void readLabelVocabulary(ObjectInputStream in) throws IOException, ClassNotFoundException 
	{
		//read label vocabulary
		label_vocabulary = (Alphabet) in.readObject ();
	}
	
	public static void printOption()
	{
		String options = "Usage: LabeledLDA [-mode <train|test>] [-train_text <train mallet file>]"
				+ " [-train_label <train labels>] [-test_text <test text>] [-model_out_folder <folder name>] [-prediction_out_folder <folder name>]"
				+ " [-iteration <number of iteration>] [-top_words <number of top words>]";
	    System.err.println(options);
	    return ;	   
	}
	
	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException, ClassNotFoundException
	{		
		String mode = "";
		String train_text_fname = "";
		String train_label_fname = "";
		String test_text_fname = "";	
		String model_out_folder = "";
		String prediction_out_folder = "";
		int numIterations = 1000;
		int numTopWords = 20;		
		
		String model_fname = "";
		String text_vocabulary_fname = "";		
		String topword_fname = "";
		String topic_position_fname = "";		
		String test_topic_position_fname = "";
		String test_topic_document_fname = "";
		
		// TODO Auto-generated method stub
		    for (int i = 0; i < args.length; i++) {
		    	if (args[i].equals("-mode")) {
		    		mode = args[++i];
		      }else if (args[i].equals("-train_text")) {
		    	  train_text_fname = args[++i];
		      }		      
		      else if (args[i].equals("-train_label")) {
		    	  train_label_fname = args[++i];
		      }
		      else if (args[i].equals("-test_text")) {
		    	  test_text_fname = args[++i];
		      }
		      else if (args[i].equals("-model_out_folder")) {
		    	  model_out_folder = args[++i];
		      }
		      else if (args[i].equals("-prediction_out_folder")) {
		    	  prediction_out_folder = args[++i];
		      }
		      else if (args[i].equals("-iteration")) {
		    	  numIterations = Integer.parseInt(args[++i]);
		      }
		      else if (args[i].equals("-top_words")) {
		    	  numTopWords = Integer.parseInt(args[++i]);
		      }
		    }		 		 		 						
			
		if(mode.equalsIgnoreCase("train"))
		{
			if(train_text_fname.equals("") || train_label_fname.equals(""))
			{
				System.err.println("train text (<-train_text> or <-train_label>) file not set");
				 printOption();
				 return;
			}
			
			if(model_out_folder.equals(""))
			{
				 System.err.println("<-model_out_folder> not set");
				 printOption();
				 return;
			}
			
			model_fname = model_out_folder + "/train_model_raw_parameters.model";
			text_vocabulary_fname = model_out_folder + "/train_model_text.vocabulary";				
			topword_fname = model_out_folder + "/train_model_top_words.model";//top k words for a topic and its probabilities
			topic_position_fname = model_out_folder + "/train_model_topics_position.model";//one MCMC chain for the topic assignment at each word
			
			InstanceList ilist = InstanceList.load (new File(train_text_fname));//documents
			
			System.out.println ("Data loaded.");
			
			System.out.println ("Start training...");
			
			LabeledLDA lda = new LabeledLDA ();
			
			lda.init (ilist, train_label_fname, new Randoms());
			
			lda.estimate(0, ilist.size(), numIterations, 50, 0, null, new Randoms());
			
			lda.writeTopWords (topword_fname, numTopWords, true);
			
			lda.writeTopicsAtPositionInDocuments(lda.topics, topic_position_fname);
					
			//write model parameters
			ObjectOutputStream oos = new ObjectOutputStream (new FileOutputStream (model_fname));
			lda.writeModel(oos);
			oos.close();
			
			//write text vocabulary
			ObjectOutputStream oos_text = new ObjectOutputStream (new FileOutputStream (text_vocabulary_fname));
			lda.writeTextVocabulary(oos_text);
			oos_text.close();
		}
		else if(mode.equalsIgnoreCase("test"))
		{
			if(test_text_fname.equals(""))
			{
				System.err.println("test text (-test_text) file not set");
				 printOption();
				 return;
			}
			
			if(prediction_out_folder.equals(""))
			{
				 System.err.println("<-prediction_out_folder> not set");
				 printOption();
				 return;
			}
			
			model_fname = model_out_folder + "/train_model_raw_parameters.model";
			text_vocabulary_fname = model_out_folder + "/train_model_text.vocabulary";
			test_topic_position_fname = prediction_out_folder + "/test_model_topics_position.test_model";//one MCMC chain for the topic assignment at each word
			test_topic_document_fname =  prediction_out_folder + "/test_model_topic_per_document.test_model";//accumuate topic assignment for a document
			
			InstanceList test_ilist = InstanceList.load (new File(test_text_fname));//documents
			
			LabeledLDA lda = new LabeledLDA ();
						
			//read model
			ObjectInputStream ois = new ObjectInputStream (new FileInputStream(model_fname));
	        lda.readModel(ois);
			ois.close();
			
			lda.init_test(test_ilist, new Randoms());
			
			//read text vocabulary			
			ObjectInputStream ois2 = new ObjectInputStream (new FileInputStream(text_vocabulary_fname));
	        lda.readTextVocabulary(ois2);
			ois2.close();
						
	        //do inference on the test dataset
	        lda.test(0, test_ilist.size(), numIterations, 50, new Randoms());
	        
	        //write results	        				
			lda.writeTopicsAtPositionInDocuments(lda.test_topics, test_topic_position_fname);
			
			//write topicd predicted for documents
			lda.writeDocumentTopics (test_topic_document_fname, lda.test_topics, lda.test_docTopicCounts, 0.0, -1);
		}		
	}
}
