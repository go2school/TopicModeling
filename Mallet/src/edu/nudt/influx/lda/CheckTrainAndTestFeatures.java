package edu.nudt.influx.lda;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.types.FeatureVectorSequence.Iterator;

public class CheckTrainAndTestFeatures {

	public static void printDict(String fname, Alphabet tr_alpha, int numTypes) throws IOException{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		for(int i=0;i<numTypes;i++)
		{
			pw.println(i + " " + tr_alpha.lookupObject(i));
		}
		pw.flush();
		pw.close();
	}
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String trFname = "/home/xiao/datasets/OSS_data/ohinferencefiles.mallet";
		String teFname = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.mallet";
		String dictTr = "/home/xiao/workspace/enrich/ohloh/ohloh.dict";
		String dictTe = "/home/xiao/workspace/enrich/sourceforge/sourceforge.dict";
		
		InstanceList ilistTrain = InstanceList.load (new File(trFname));		
		InstanceList ilistTest = InstanceList.load (new File(teFname));		
		printDict(dictTr, ilistTrain.getDataAlphabet(), ilistTrain.getDataAlphabet().size());
		printDict(dictTe, ilistTest.getDataAlphabet(), ilistTest.getDataAlphabet().size());
	}

}
