package seeu;

import java.util.Arrays;
import java.util.Hashtable;
import java.io.*;

import cc.mallet.types.*;
import cc.mallet.util.ArrayUtils;
import cc.mallet.util.Randoms;

public class lda {

	int numTopics; // Number of topics to be fit
	double alpha;  // Dirichlet(alpha,alpha,...) is the distribution over topics
	double beta;   // Prior on per-topic multinomial distribution over words
	double tAlpha;
	double vBeta;
	InstanceList ilist;  // the data field of the instances is expected to hold a FeatureSequence	
	int numTypes;
	int numTokens;
	
	// indexed by <document index, sequence index>
	// m: document index
	// i: position index
	//int[][] topics;
	Hashtable<Integer, Integer>[] docPositionTopicCounts;
	
	// indexed by <document index, topic index>
	// m: document index
	// z: topic index
	//This matrix is good enough
	int[][] docTopicCounts; 
	
	// indexed by <feature index, topic index>
	// z: topic index
	// t: term index
	// as term is very large, we'd better not use array
	Hashtable<Integer, Integer>[] topicTypeCounts; 
	
	// indexed by <topic index>
	int[] tokensPerTopic; 
	
	int [][] topics;
	int [][] typeTopicCounts;
	
	public void increaseCount(Hashtable<Integer, Integer> tb, int word_index)
	{
		if(tb.contains(word_index))
		{
			Integer b = tb.get(word_index);
			tb.put(word_index, b+1);
			
		}
	}
	
	public void decreaseCount(Hashtable<Integer, Integer> tb, int word_index)
	{
		if(tb.contains(word_index))
		{
			Integer b = tb.get(word_index);
			tb.put(word_index, b-1);
			
		}
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
			}
			if (outputModelInterval != 0
					&& iterations % outputModelInterval == 0 && iterations > 0) {
				//this.write(new File(outputModelFilename + '.' + iterations));
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
			//get old topic
			oldTopic = oneDocTopics[si];
			// Remove this token from all counts
			oneDocTopicCounts[oldTopic]--;
			//get <topic, word> count
			decreaseCount(topicTypeCounts[oldTopic], type);			
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
				// System.out.println(si+"---"+ ti+"---"+tw);
			}
			// Sample a topic assignment from this distribution
			newTopic = r.nextDiscrete(topicWeights, topicWeightsSum);
			// System.out.println(si+"---"+ newTopic);
			// Put that new topic into the counts
			oneDocTopics[si] = newTopic;
			oneDocTopicCounts[newTopic]++;
			typeTopicCounts[type][newTopic]++;
			tokensPerTopic[newTopic]++;
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
