/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

package seeu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.TreeSet;
import java.io.*;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

/**
 * Latent Dirichlet Allocation.
 * @author Xiao Li
 */

// Think about support for incrementally adding more documents...
// (I think this means we might want to use FeatureSequence directly).
// We will also need to support a growing vocabulary!

public class StandardLDA implements Serializable {

	int numTopics; // Number of topics to be fit
	double alpha;  // Dirichlet(alpha,alpha,...) is the distribution over topics
	double beta;   // Prior on per-topic multinomial distribution over words
	double tAlpha;
	double vBeta;
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence		
	
	Alphabet text_vocabulary = null;
	
	double [] topicWeights = null;
	
	//
	//variables for training
	//
	
	int[][] topics; // indexed by <document index, sequence index>
	int numTypes;//total number of **unique** terms
	int numTokens;//total number of terms in all documents

	int numDocs;//number of text docs
	
	int[][] docTopicCounts; // indexed by <document index, topic index>
	int[][] typeTopicCounts; // indexed by <feature index, topic index>
	int[] tokensPerTopic; // indexed by <topic index>	
	
	int [] docLength;//doc length, indexed by document index
	
	int [] all_topics;//store all used topic index
	//
	//variables for testing
	//
	
	public InstanceList test_ilist;  // the data field of the instances is expected to hold a FeatureSequence
	
	int[][] test_topics; // indexed by <document index, sequence index>	
	int test_numTypes;//total number of terms in all documents
	int test_numTokens;//total number of terms in all documents
	
	int test_numDocs;//number of text docs	
	
	int[][] test_docTopicCounts; // indexed by <document index, topic index>
	int[][] test_typeTopicCounts; // indexed by <feature index, topic index>
	int[] test_tokensPerTopic; // indexed by <topic index>	
	
	double [][] theta;// LDA output, Theta
	double [][] phi;// LDA output Phi
	
	double [][] test_theta;// LDA output, Theta
	
	boolean isFirstTimePerplexy = true;
	boolean show_perplexity_on_test = false;
	
	public StandardLDA (int numberOfTopics)
	{
		this (numberOfTopics, 50.0, 0.01);
	}

	public StandardLDA ()
	{
		
	}
	
	public StandardLDA (int numberOfTopics, double alphaSum, double beta)
	{
		this.numTopics = numberOfTopics;
		this.alpha = alphaSum / numTopics;//as same as "finding scientific topics", divied by number of topics
		this.beta = beta;
	}
	
	public void init (InstanceList documents,  Randoms r) throws IOException
	{
		ilist = documents.shallowClone();
		numTypes = ilist.getDataAlphabet().size ();//get vocabulary size
		numDocs = ilist.size();
		
		//get vocabulary
		text_vocabulary = ilist.getDataAlphabet();		
				
		topicWeights = new double[numTopics];
		
		//decide alpha and beta
		//following the paper "Finding scientific topics"	
		//alpha and beta are initlized in constructor function		
		tAlpha = alpha * numTopics;
		vBeta = beta * numTypes;
		
		topics = new int[numDocs][];
		docTopicCounts = new int[numDocs][numTopics];
		typeTopicCounts = new int[numTypes][numTopics];
		tokensPerTopic = new int[numTopics];
		docLength = new int [numDocs];
		
		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen;
	    FeatureSequence fs;
	    for (int di = 0; di < numDocs; di++) {
	      try {
	        fs = (FeatureSequence) ilist.get(di).getData();
	      } catch (ClassCastException e) {
	        System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
	                            +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
	        throw e;
	      }
	        seqLen = fs.getLength();
	        
	        docLength[di] = seqLen;
	        
			numTokens += seqLen;
			topics[di] = new int[seqLen];
			// Randomly assign tokens to topics					
			for (int si = 0; si < seqLen; si++) {								
				//for this document, sample a topic								
				topic = r.nextInt(numTopics);				
				//update variables
				topics[di][si] = topic;
				docTopicCounts[di][topic]++;
				typeTopicCounts[fs.getIndexAtPosition(si)][topic]++;
				tokensPerTopic[topic]++;
			}
		}    	    		
	}
	
	public void init_test (InstanceList documents, Randoms r, boolean using_model)
	{
		test_ilist = documents.shallowClone();		
		test_numDocs = test_ilist.size();								
		
		test_topics = new int[test_numDocs][];
		test_docTopicCounts = new int[test_numDocs][numTopics];
		test_typeTopicCounts = new int[numTypes][numTopics];
		test_tokensPerTopic = new int[numTopics];

		topicWeights = new double [numTopics];
		
		all_topics = new int [numTopics];
		for(int i=0;i<numTopics;i++)
			all_topics[i] = i;
		
		// Initialize with random assignments of tokens to topics
		// and finish allocating this.topics and this.tokens
		int topic, seqLen, token;
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
				
				token = fs.getIndexAtPosition(si);
				//use the infered parameter to estimate
				if(using_model)
				{
					topic = sampleATopic(si, token, all_topics, docTopicCounts[di], topicWeights, r);
				}
				else
					//we do not use smart sampling here
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
      sampleTopicsForAllDocs(r);
      	     
	      if(show_perplexity_on_test)
	      {
	    	  computeTheta();
	    	  computePhi();
	    	  double p = perplexity(ilist, phi, theta);
				System.out.println("Perplexity " + p);
	      }
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
	public void sampleTopicsForAllDocs (Randoms r)
	{
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = 0; di < numDocs; di++) {
			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
			                       topics[di], docTopicCounts[di], topicWeights, r);
		}
	}

	/* One iteration of Gibbs sampling, across all documents. */
	public void sampleTopicsForDocs (int start, int length, Randoms r)
	{
		assert (start+length <= docTopicCounts.length);
		double[] topicWeights = new double[numTopics];
		// Loop over every word in the corpus
		for (int di = start; di < start+length; di++) {
			sampleTopicsForOneDoc ((FeatureSequence)ilist.get(di).getData(),
			                       topics[di], docTopicCounts[di], topicWeights, r);
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
	
	
  private void sampleTopicsForOneDoc (FeatureSequence oneDocTokens, int[] oneDocTopics, // indexed by seq position
	                                    int[] oneDocTopicCounts, // indexed by topic index
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
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
			oneDocTopicCounts[oldTopic]--;
			typeTopicCounts[type][oldTopic]--;
			tokensPerTopic[oldTopic]--;
			// Build a distribution over topics for this token			
			topicWeightsSum = 0;
			currentTypeTopicCounts = typeTopicCounts[type];
			for (int ti = 0; ti < numTopics; ti++) {
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
				      * ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete (topicWeights, topicWeightsSum);

			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
			oneDocTopicCounts[newTopic]++;
			typeTopicCounts[type][newTopic]++;
			tokensPerTopic[newTopic]++;
		}
	}
	
	  private void sampleTopicsForOneTestDoc (FeatureSequence oneDocTokens, 			  
	          int[] oneDocTopicCounts, // indexed by topic index
	          int[] test_oneDocTopics, // indexed by seq position
	          int[] test_oneDocTopicCounts, // indexed by topic index
	          double[] topicWeights, Randoms r)
	{
		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopicIndex, newTopic;
		double topicWeightsSum;
		int docLen = oneDocTokens.getLength();
		double tw;
		// Iterate over the positions (words) in the document
		for (int si = 0; si < docLen; si++) {
			type = oneDocTokens.getIndexAtPosition(si);
			//get current assignment
			oldTopic = test_oneDocTopics[si];
			// Remove this token from all counts
			test_oneDocTopicCounts[oldTopic]--;
			test_typeTopicCounts[type][oldTopic]--;
			test_tokensPerTopic[oldTopic]--;
			// Build a distribution over topics for this token
			//this formula is from WWW 08 paper GibbasLDA
			
			topicWeightsSum = 0;
			currentTypeTopicCounts = test_typeTopicCounts[type];
			for (int li = 0; li < all_topics.length; li++) {
				int ti = all_topics[li];
				tw = ((typeTopicCounts[type][ti] + currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + test_tokensPerTopic[ti] + vBeta))
				* ((oneDocTopicCounts[ti] + test_oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[ti] = tw;
			}
			// Sample a topic assignment from this distribution
			newTopicIndex = r.nextDiscreteFirstK (topicWeights, all_topics.length, topicWeightsSum);
			
			newTopic = all_topics[newTopicIndex];
			
			// Put that new topic into the counts
			test_oneDocTopics[si] = newTopic;
			
			//update all counts
			test_oneDocTopicCounts[newTopic]++;
			test_typeTopicCounts[type][newTopic]++;
			test_tokensPerTopic[newTopic]++;
		}
	}	 
	  
	  public int sampleATopic(int si,//token position 
			  int type,//token
			  int [] topics,//a set of topics
	          int[] oneDocTopicCounts, // indexed by topic index	    
	          double[] topicWeights, Randoms r)
	  {
		    int[] currentTypeTopicCounts;
			int newTopicIndex;
			double topicWeightsSum;
			
			double tw;
			
			// Build a distribution over topics from all topics

			topicWeightsSum = 0;
			
			currentTypeTopicCounts = typeTopicCounts[type];
			for(int li=0;li<topics.length;li++)
			{
				//get the topic index
				int ti = topics[li];
				//estimate the topic weight
				tw = ((currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + vBeta))
						* ((oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
				topicWeightsSum += tw;
				topicWeights[li] = tw;
			}			
			
			// Sample a topic assignment from this distribution
			newTopicIndex = r.nextDiscreteFirstK (topicWeights, topics.length, topicWeightsSum);
			
			//return the topic index
			return topics[newTopicIndex];
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
				
				 if(show_perplexity_on_test)
			      {
					 computeTestTheta();
					  computePhi();
				      double p = perplexity(test_ilist, phi, test_theta);
						System.out.println("Perplexity " + p);
			      }
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
	
	public void computeTheta()
	{
		//code from GibbasLDA++
		theta = new double [numDocs][numTopics];
		for(int di=0;di<numDocs;di++)
		{			
			for(int li=0;li<all_topics.length;li++)
			{
				int ti = all_topics[li];
				theta[di][ti] = (((double)docTopicCounts[di][ti]) + alpha)/(docLength[di] + tAlpha);
			}
		}				
	}
	
	public void computeTestTheta()
	{
		//code from GibbasLDA++
		test_theta = new double [test_numDocs][numTopics];
		for(int di=0;di<test_numDocs;di++)
		{			
			FeatureSequence oneDocTokens = (FeatureSequence)test_ilist.get(di).getData();
			int docLen = oneDocTokens.getLength();
			for(int li=0;li<all_topics.length;li++)
			{
				int ti = all_topics[li];
				test_theta[di][ti] = ((double)test_docTopicCounts[di][ti] + docTopicCounts[di][ti] + alpha)/(docLen + tAlpha);
				
				if(test_theta[di][ti] == 0.0)
				{
					System.out.println("bad  " + di + " " + ti + " " + test_theta[di][ti] + " " + test_docTopicCounts[di][ti] + " " + alpha);
				}
			}
			
		}	
		
		/*
		 * 
		 * tw = ((typeTopicCounts[type][ti] + currentTypeTopicCounts[ti] + beta) / (tokensPerTopic[ti] + test_tokensPerTopic[ti] + vBeta))
				* ((oneDocTopicCounts[ti] + test_oneDocTopicCounts[ti] + alpha)); // (/docLen-1+tAlpha); is constant across all topics
		 */
	}
	
	public void computePhi()
	{
		//code from GibbasLDA++
		phi = new double [numTypes][numTopics];
		for(int wi=0;wi<numTypes;wi++)
		{			
			for(int li=0;li<all_topics.length;li++)
			{
				int ti = all_topics[li];
				phi[wi][ti] = (((double)typeTopicCounts[wi][ti]) + beta)/(tokensPerTopic[ti] + vBeta);
			}
		}
	}
	
	
	public Hashtable<Integer, Integer> getAllWordOcc(FeatureSequence oneDocTokens)
	{
		//get all word and occs
		int docLen = oneDocTokens.getLength();
		// Iterate over the positions (words) in the document
		Hashtable<Integer, Integer> word_occ = new Hashtable<Integer, Integer>();
		for (int si = 0; si < docLen; si++) {
			int wi = oneDocTokens.getIndexAtPosition(si);//get word
			if(word_occ.containsKey(wi) == false)
				word_occ.put(wi, 1);
			else
			{
				int occ = word_occ.get(wi);
				word_occ.put(wi, occ+1);
			}
		}
		return word_occ;
	}
	
	public double perplexity(InstanceList instance_list, double [][] phi, double [][] theta)
	{		
		double ret = 0;
		//
		//from JMLR 03 LDA
		//=exp(-\frac{\sum_{i=1}^Mlog(p(w_d))}{\sum_{i=1}^MN_i})
		//
		//p(w_d)=\sum_tp(w|t)p(t|d)=\sum_t phi(w,t)*theta(t,d)
		//
		//
		double up = 0, down = 0;
			
		for (int di = 0; di < instance_list.size(); di++) {		
			FeatureSequence oneDocTokens = (FeatureSequence)instance_list.get(di).getData();
			
			//get occ of each word
			Hashtable<Integer, Integer> word_occ = getAllWordOcc(oneDocTokens);
			
			double twlog = 0;
			// Iterate over the positions (words) in the document
			for (Integer wi: word_occ.keySet()) {
				int occ = word_occ.get(wi);
				//compute inner product of \sum_t p(w|t)p(t|d)
				double v = 0;
				for(int li=0;li<all_topics.length;li++)
				{
					int ti = all_topics[li];//get topic
					//compute phi
					double v_phi = phi[wi][ti];
					//compute theta
					double v_theta = theta[di][ti];
					v += v_theta * v_phi;
				}			
				twlog += Math.log(v) * occ;
			}
			//update 
			up += twlog;
			down += oneDocTokens.getLength();
		}
		
		ret = Math.exp(-up/down);

		return ret;
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
				pw.println ("Topic "+ti);
				for (int i = 0; i < numWords; i++)
					pw.println (ilist.getDataAlphabet().lookupObject(wp[i].wi).toString() + " " + wp[i].p);
				pw.println("\n");
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
		//write doc length
		for(int di=0;di<numDocs;di++)
			out.writeInt(docLength[di]);
	}	
	
	public void readModel(ObjectInputStream in) throws IOException, ClassNotFoundException 
	{				
		//read count
		numDocs = in.readInt();
		numTopics = in.readInt();		
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
		//read doc length
		docLength = new int [numDocs];
		for(int di=0;di<numDocs;di++)
			docLength[di] = in.readInt();
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
	
	public static void printOption()
	{
		String options = "Usage: LabeledLDA [-mode <train|test>] [-train_text <train mallet file>]"
				+ " [-test_text <test text>] [-use_model_init_test <if using model parameter in testing>] [-show_perplexity_on_test <if show perplexity on test in testing>] [-model_out_folder <folder name>] [-prediction_out_folder <folder name>]"
				+ " [-iteration <number of iteration>] [-top_words <number of top words>]";
	    System.err.println(options);
	    return ;	   
	}
	
	// Recommended to use mallet/bin/vectors2topics instead.
	public static void main (String[] args) throws IOException, ClassNotFoundException
	{		
		String mode = "";
		String train_text_fname = "";
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
		int num_of_topics = 0;
		
		boolean using_model = false;
		boolean show_perplexity_on_test = false;
		
		// TODO Auto-generated method stub
		    for (int i = 0; i < args.length; i++) {
		    	if (args[i].equals("-mode")) {
		    		mode = args[++i];
		      }else if (args[i].equals("-train_text")) {
		    	  train_text_fname = args[++i];
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
		      else if (args[i].equals("-use_model_init_test")) {
		    	  using_model = Boolean.parseBoolean(args[++i]);
		      }
		      else if (args[i].equals("-show_perplexity_on_test")) {
		    	  show_perplexity_on_test = Boolean.parseBoolean(args[++i]);
		      }
		      else if (args[i].equals("-num_topics")) {
		    	  num_of_topics = Integer.parseInt(args[++i]);
		      }
		    }		 		 		 						
			
		if(mode.equalsIgnoreCase("train"))
		{
			if(train_text_fname.equals(""))
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
			
			if(num_of_topics == 0)
			{
				System.err.println("<-num_topics> not set");
				 printOption();
				 return;
			}
			
			model_fname = model_out_folder + "/train_model_raw_parameters.model";
			text_vocabulary_fname = model_out_folder + "/train_model_text.vocabulary";				
			topword_fname = model_out_folder + "/train_model_top_words.model";//top k words for a topic and its probabilities
			topic_position_fname = model_out_folder + "/train_model_topics_position.model";//one MCMC chain for the topic assignment at each word
			
			InstanceList ilist = InstanceList.load (new File(train_text_fname));//training documents						
			
			System.out.println ("Data loaded.");
			
			System.out.println ("Start training...");
			
			StandardLDA lda = new StandardLDA (num_of_topics);
			
			lda.init (ilist, new Randoms());				
			
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
			
			StandardLDA lda = new StandardLDA ();
						
			lda.show_perplexity_on_test = show_perplexity_on_test;
			
			//read model
			ObjectInputStream ois = new ObjectInputStream (new FileInputStream(model_fname));
	        lda.readModel(ois);
			ois.close();
			
			//read test data and initialize inference
			lda.init_test(test_ilist, new Randoms(), using_model);
			
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
		else
		{
			printOption();
			 return;
		}
	}

}
