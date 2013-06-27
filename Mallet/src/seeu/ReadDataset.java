package seeu;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;

public class ReadDataset {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		//String fname = "/media/DataVolume1/datasets/20news/new_dataset/20_news_text.no_stop.stemming.mallet";
		//String fname = "/media/DataVolume1/datasets/20news/new_dataset/5_folds/20_news_0_fold_train_text.mallet";
		String fname = "/media/DataVolume1/datasets/rcv1/new_dataset/5_folds/rcv1_0_fold_train_text.mallet";
		//String fname = "/media/DataVolume1/datasets/ohsumed/new_dataset/ohsumed_mallet_text.txt.no_stop.stemming.no_rare.3.mallet";
		InstanceList ilist = InstanceList.load (new File(fname));//documents
		Alphabet voc = ilist.getDataAlphabet();
		String voctext = voc.toString();
		//PrintWriter pw = new PrintWriter(new FileWriter("/media/DataVolume1/datasets/ohsumed/new_dataset/ohsumed_text.voc_train"));
		PrintWriter pw = new PrintWriter(new FileWriter("/media/DataVolume1/datasets/rcv1/new_dataset/rcv1_text.voc_train"));
		pw.print(voctext);
		pw.flush();
		pw.close();
		
		//check dataset 		
		int numDocs = ilist.size();
		FeatureSequence FSlabels;
		System.out.println(numDocs);
		for(int di=0;di<numDocs && di < 1;di++)
		{			
			try {
				FSlabels = (FeatureSequence) ilist.get(di).getData();
		      } catch (ClassCastException e) {
		        System.err.println ("LDA and other topic models expect FeatureSequence data, not FeatureVector data.  "
		                            +"With text2vectors, you can obtain such data with --keep-sequence or --keep-bisequence.");
		        throw e;
		      }
			int numLabel = FSlabels.getLength();						
			// Randomly assign tokens to topics
			for (int si = 0; si < numLabel; si++) {
				int type = FSlabels.getIndexAtPosition(si);
				String word = voc.lookupObject(type).toString();
				System.out.print(type + ":" + word + " " );
			}		
			System.out.println("");
		}
				
		
	}

}
