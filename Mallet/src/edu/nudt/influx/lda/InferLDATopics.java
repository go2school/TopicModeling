package edu.nudt.influx.lda;

import java.io.File;

import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.InstanceList;

public class InferLDATopics {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String inferencerFilename = "/home/xiao/workspace/enrich/ohloh/lda_model.mallet";
		String testdata_fname = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.mallet"; 
		String docTopicsFile = "/home/xiao/test_mallet/doc_topics_output.txt";
		
		int numIterations = 100;
		int sampleInterval = 10;
		int burnInIterations = 10;
		double docTopicsThreshold = 0.0;
		int docTopicsMax = 50;
		// TODO Auto-generated method stub
		try {
			
			TopicInferencer inferencer = 
				TopicInferencer.read(new File(inferencerFilename));

			InstanceList instances = InstanceList.load (new File(testdata_fname));

			inferencer.writeInferredDistributions(instances, new File(docTopicsFile),
												  numIterations, sampleInterval,
												  burnInIterations,
												  docTopicsThreshold, docTopicsMax);
			

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
		}
	}

}
