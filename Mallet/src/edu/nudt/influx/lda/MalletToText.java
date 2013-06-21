package edu.nudt.influx.lda;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;

public class MalletToText {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		args = new String [3];
		
		String fname = args[0];
		String outfname1 = args[1];
		String outfname2 = args[2];
		
		fname = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.dict";
		
		fname = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.dict";
		//outfname2 = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.dict";
		//outfname2 = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/ohinferencefiles.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/ohinferencefiles.dict";
		//outfname2 = "/home/xiao/datasets/OSS_data/ohinferencefiles.data";
		
		InstanceList instances = InstanceList.load (new File(fname));
		int numTypes = instances.getDataAlphabet().size();
		PrintWriter pw1 = new PrintWriter(new FileWriter(outfname1));		
		for(int wi=0;wi<numTypes;wi++)
		{
			String word = instances.getDataAlphabet().lookupObject(wi).toString();			
			pw1.println(word + " " + (wi + 1));			
		}
		pw1.flush();
		pw1.close();		
	}

}
