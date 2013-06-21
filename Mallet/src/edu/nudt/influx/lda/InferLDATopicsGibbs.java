package edu.nudt.influx.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import cc.mallet.topics.TopicInferencer;
import cc.mallet.topics.tui.InferTopics;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;
import cc.mallet.util.CommandOption;
import cc.mallet.util.Randoms;

public class InferLDATopicsGibbs extends TopicInferencer{

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
	
	public InferLDATopicsGibbs(int[][] typeTopicCounts, int[] tokensPerTopic,
			Alphabet alphabet, double[] alpha, double beta, double betaSum) {
		super(typeTopicCounts, tokensPerTopic, alphabet, alpha, beta, betaSum);
		// TODO Auto-generated constructor stub
		r = new Randoms();	
	}		
	
	// double Vbeta = 0;	
	    
	int numIterations = 50;
	int sampleInterval = 10;
	int burnInIterations = 10;
	double docTopicsThreshold = 0.0;
	int docTopicsMax = 50;
	
	Randoms r;
	
	InstanceList ilist;			
	
	double [] p;
	// for inference only    
    int newDocs;
    int [][] new_docWordsTopics;//indexed by <document index, feature index>
	int[][] new_docTopicCounts; // indexed by <document index, topic index>
	int[][] new_typeTopicCounts; // indexed by <feature index, topic index>
	int[] new_tokensPerTopic; // indexed by <topic index>
	int[] new_docLengths; //index by <document index>
	
	public void inference()
	{	
		//repeatedly sample a topic for the word at each position in each document
		for(int iter = 0; iter < numIterations; iter++)
		{
			for(int di =0;di<newDocs;di++)
			{
				FeatureSequence oneDocTokens = (FeatureSequence) ilist.get(di).getData();
				int docLen = oneDocTokens.getLength();
				for (int si = 0; si < docLen; si++) 
				{
					int type = oneDocTokens.getIndexAtPosition(si);
					int topic = inf_sampling(di, si, type);
					new_docWordsTopics[di][si] = topic;
					if(di == 0 && (iter == numIterations - 1 || iter == 0))
					{
						System.out.print(ilist.getDataAlphabet().lookupObject(si).toString() + ":"  + topic  + " ");
					}
				}
				
				if(di == 0)
				{
					//System.out.println("");
					
				//System.out.print(iter + " " + new_docLengths[di]  + ": ");
				//	for(int i=0;i<numTopics;i++)
				//		System.out.print(new_docTopicCounts[di][i]  + " ");
				//	System.out.println(" ");
				}
			}
		}
	}
	
	int getMostCommonTopic(int type, Randoms r)
	{		
		int maxValue = -1;
		int maxIndex = -1;
		for(int i=0;i<numTopics;i++)
		{
			if(typeTopicCounts[type][i] > maxValue)
			{
				maxValue = typeTopicCounts[type][i];
				maxIndex = i;
			}
		}
		if(maxIndex == -1)
		{				
			//random sample a topic
			maxIndex = r.nextInt(numTopics);			
		}
		return maxIndex;
	}
	
	public void init_inf(InstanceList testInstances) throws Exception
	{		
		ilist = testInstances.shallowClone();
		//get total number of docs in the new dataset
		newDocs = ilist.size();
		//create <word_index, topic_index>
		new_typeTopicCounts = new int [numTypes][];
		for(int i=0;i<numTypes;i++)
			new_typeTopicCounts[i] = new int [numTopics];
		//create <doc_index, topic_index>
		new_docTopicCounts = new int [newDocs][];
		for(int i=0;i<newDocs;i++)
			new_docTopicCounts[i] = new int [numTopics];
		//create <doc_index, word_index>
		new_docWordsTopics = new int [newDocs][];		
		//create <topic_index> sum of tokens assigned to this topic
		new_tokensPerTopic = new int [numTopics];
		//create <doc_index> sum of total docs
		new_docLengths = new int [newDocs];
		//temporarility array for computation
		p = new double [numTopics];
		
		//assign the most common topic to each word
		for(int m=0;m<newDocs;m++)
		{
			FeatureSequence fs = (FeatureSequence) ilist.get(m).getData();
			int docLen = fs.getLength();
			new_docWordsTopics[m] = new int [docLen];
			for(int n=0;n<docLen;n++)
			{
				int type = fs.getIndexAtPosition(n);
				
				//get topic
				int topic = -1;
				if (!(type < numTypes && typeTopicCounts[type].length != 0))
				{
					//if the word does not appear in the training set
					//randomly sample a path
					topic = r.nextInt(numTopics);	
					System.out.println("not appearing ");
				}
				else
				{	
					//if the word appears in the training set
					//find the most common topic based on the estimated parameter of count(word, topic) from the training data
					//do not assign the most common topic to the word
					//topic = this.getMostCommonTopic(type, r);
					topic = r.nextInt(numTopics);	
					//System.out.println("not appearing ");
				}
				
				new_docWordsTopics[m][n] = topic;
				new_typeTopicCounts[type][topic] ++;
				new_docTopicCounts[m][topic] ++;
				new_tokensPerTopic[topic] ++;
			}
			new_docLengths[m] = docLen;		
		}
	}
	//infer the topics for document d and word n 
	public int inf_sampling(int m, int n, int type)
	{		
		int topic = new_docWordsTopics[m][n];
	    
		int w = type;
	    int _w = type;
	    
	    //remove current assignment
	    new_typeTopicCounts[_w][topic] -= 1;
	    new_docTopicCounts[m][topic] -= 1;
	    new_tokensPerTopic[topic] -= 1;
	    	   
	    // do multinomial sampling via cumulative method
	    double cumulativeWeight = 0;
	    for (int k = 0; k < numTopics; k++) 
	    {
	    	p[k] = (typeTopicCounts[w][k] + new_typeTopicCounts[_w][k] + beta) / (tokensPerTopic[k] + new_tokensPerTopic[k] + betaSum) *
			    (new_docTopicCounts[m][k] + alpha[k]);// / (new_docLengths[m] + Kalpha);
	    	cumulativeWeight += p[k];
	    }
	  
	    //sammple a topic
	    topic = r.nextDiscreteWithSize(p, numTopics, cumulativeWeight);
	    
	    // add newly estimated z_i to count variables	    
	    
	    new_typeTopicCounts[_w][topic] += 1;
	    new_docTopicCounts[m][topic] += 1;
	    new_tokensPerTopic[topic] += 1;	    
	    
	    return topic;
	}
	
	public void save_newtheta(String fname, int withPrior) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
	    for (int m = 0; m < newDocs; m++) {	    	
	    	ArrayList<WordProb> lst = new ArrayList<WordProb>();
	    	
	    	if(withPrior == 1)
	    	{
		    	for (int k = 0; k < numTopics; k++) {	    	
				    double v = (double)(new_docTopicCounts[m][k] + alpha[k]) / (new_docLengths[m] + numTopics * alpha[k]);
				    
				    if(v != 0)
				    	lst.add(new WordProb(k, v));			    			    
				}
	    	}
	    	else
	    	{
	    		for (int k = 0; k < numTopics; k++) {	    	
				    double v = (double)(new_docTopicCounts[m][k]) / new_docLengths[m];
				    
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
		
	public void printTopicPerWord(String fname)throws IOException 
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
	    for (int m = 0; m < newDocs; m++) {
	    	FeatureSequence fs = (FeatureSequence) ilist.get(m).getData();
	    	for (int si = 0; si < new_docWordsTopics[m].length; si++) {
				int type = fs.getIndexAtPosition(si);
				String word = ilist.getDataAlphabet().lookupObject(type).toString();
			    pw.print(word+":"+new_docWordsTopics[m][si] + " ");
			}
			pw.print('\n');
	    }
	    pw.flush();
	    pw.close();
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
	
	public void output_doc_topic_by_threshold(String fname, double threshold) throws IOException {
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
	    for (int m = 0; m < newDocs; m++) {
			for (int k = 0; k < numTopics; k++) {
			    double v = (new_docTopicCounts[m][k] + alpha[k]) / (new_docLengths[m] + numTopics * alpha[k]);			    
			    String str = v + " ";
			    pw.print(str);
			}
			pw.print('\n');
	    }
	    pw.flush();
	    pw.close();
	}	
	
	public void printDocumentTopics(String fname) throws IOException {
		printDocumentTopics(new PrintWriter(new FileWriter(fname)));
	}

	public void printDocumentTopics(PrintWriter pw) {
		printDocumentTopics(pw, 0.0, -1);
		pw.close();
	}
	
	public void printDocumentTopics(PrintWriter pw, double threshold, int max) {
		pw.println("#doc source topic proportion ...");
		int docLen;
		double topicDist[] = new double[numTopics];
		for (int di = 0; di < new_docWordsTopics.length; di++) {
			pw.print(di);
			pw.print(' ');
			if (ilist.get(di).getSource() != null) {
				pw.print(ilist.get(di).getSource().toString());
			} else {
				pw.print("null-source");
			}
			pw.print(' ');
			docLen = new_docWordsTopics[di].length;
			for (int ti = 0; ti < numTopics; ti++)
				topicDist[ti] = (((float) new_docTopicCounts[di][ti]) / docLen);
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
	
	public static InferLDATopicsGibbs read (File f) throws Exception {

		InferLDATopicsGibbs inferencer = null;

		ObjectInputStream ois = new ObjectInputStream (new FileInputStream(f));
        inferencer = (InferLDATopicsGibbs) ois.readObject();
		ois.close();

        return inferencer;
    }
	
	
	
	 static CommandOption.String inferencerFilename = new CommandOption.String
     (InferTopics.class, "inferencer", "FILENAME", true, null,
		 "A serialized topic inferencer from a trained topic model.\n", null);
	 
	 static CommandOption.Integer withPrior = new CommandOption.Integer
     (InferTopics.class, "withprior", "INTEGER", true, 0,
		 "Do we need to add the prior value into the probability distribution computation. (Default is 0)\n", null);

	static CommandOption.String inputFile = new CommandOption.String
		(InferTopics.class, "input", "FILENAME", true, null,
		 "The filename from which to read the list of instances\n" +
		 "for which topics should be inferred.  Use - for stdin.  " +
		 "The instances must be FeatureSequence or FeatureSequenceWithBigrams, not FeatureVector", null);
	
	static CommandOption.String docTopicsFile = new CommandOption.String
     (InferTopics.class, "output-doc-topics", "FILENAME", true, null,
      "The filename in which to write the inferred topic\n" +
		 "proportions per document.  " +
      "By default this is null, indicating that no file will be written.", null);
	
	static CommandOption.String docWordTopicFile = new CommandOption.String
    (InferTopics.class, "output-doc-word-topic", "FILENAME", true, null,
     "The filename in which to write the inferred topic\n" +
		 "proportions per word position per document.  " +
     "By default this is null, indicating that no file will be written.", null);
	
	static CommandOption.Integer para_iteration = new CommandOption.Integer
    (InferTopics.class, "iteration", "INTEGER", true, 100,
   		 "The number of iterations. (Default is 100)\n", null);
 
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*
		String inferencerFilename = "/home/xiao/test_mallet/topic_inference.mallet";
		String testdata_fname = "/home/xiao/test_mallet/topic_input.mallet"; 
		String docTopicsFile = "/home/xiao/test_mallet/doc_topics_output.txt";
		String docTopicsFile2 = "/home/xiao/test_mallet/doc_topics_output2.txt";
		*/
		
		 // Process the command-line options                                                                           
		CommandOption.setSummary (InferTopics.class,
                                  "Use an existing topic model to infer topic distributions for new documents");
        CommandOption.process (InferTopics.class, args);
		
		if (inferencerFilename.value == null) {
			System.err.println("You must specify a serialized topic inferencer. Use --help to list options.");
			System.exit(0);
		}

		if (inputFile.value == null) {
			System.err.println("You must specify a serialized instance list. Use --help to list options.");
			System.exit(0);
		}
		
		if (docTopicsFile.value == null) {
			System.err.println("You must specify a text per-document topic file. Use --help to list options.");
			System.exit(0);
		}
		
		// TODO Auto-generated method stub		
		try {
			InstanceList instances = InstanceList.load (new File(inputFile.value));
			InferLDATopicsGibbs inferencer = InferLDATopicsGibbs.read(new File(inferencerFilename.value));		
			
			inferencer.init_inf(instances);
			
			inferencer.save_newtheta(docTopicsFile.value + "_0", withPrior.value);
			
			inferencer.printToFile(50, true, new File("/home/xiao/tmp"));
			
			inferencer.numIterations = para_iteration.value;
			
			System.out.println("Finish init inference");
			System.out.println("Start inference");
			inferencer.inference();
			System.out.println("Finish inference");
			//save p(topic|doc)
			
			inferencer.save_newtheta(docTopicsFile.value, withPrior.value);
//			inferencer.printDocumentTopics(docTopicsFile2);
			inferencer.printTopicPerWord(docWordTopicFile.value);
			System.out.println("End inference");

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

}
