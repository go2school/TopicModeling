package edu.nudt.influx.lda;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import cc.mallet.types.FeatureSequence;
import cc.mallet.types.InstanceList;

public class MalletToTxtAndData {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		args = new String [3];
		
		String fname = args[0];
		String outfname1 = args[1];
		String outfname2 = args[2];
		
		
		
		fname = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Test_FcFilesTagRemovedPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Test_OhFilesTagRemovedPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/ohinferencefiles.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/ohinferencefiles.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/ohinferencefiles.data";
		
		fname = "/home/xiao/datasets/OSS_data/Test_fc_SfFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_fc_SfFilesTagRemovedPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Test_fc_SfFilesTagRemovedPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/SfTrainingFiles/Train_SfFilesWithTagPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/Test_oh_SfFilesTagRemovedPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Test_oh_SfFilesTagRemovedPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Test_oh_SfFilesTagRemovedPiped.data";		
		
		fname = "/home/xiao/datasets/OSS_data/AllTrainTogetherPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/AllTrainTogetherPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/AllTrainTogetherPiped.data";		
		
		fname = "/home/xiao/datasets/OSS_data/ohinferencefiles.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/ohinferencefiles.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/ohinferencefiles.data";
		
		fname = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.data";
		
		fname = "/home/xiao/datasets/OSS_data/SfTrainingFiles/Train_SfFilesTagRemoved.mallet";
		outfname1 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.txt";
		outfname2 = "/home/xiao/datasets/OSS_data/Train_SfFilesWithTagPiped.data";
		
		InstanceList instances = InstanceList.load (new File(fname));
		PrintWriter pw1 = new PrintWriter(new FileWriter(outfname1));
		PrintWriter pw2 = new PrintWriter(new FileWriter(outfname2));
		for(int di=0;di<instances.size();di++)
		{
			pw1.print(di);
			pw2.print(di);
			FeatureSequence oneDocTokens = (FeatureSequence) instances.get(di).getData();
			int docLen = oneDocTokens.getLength();					
			for(int si=0;si<docLen;si++)
			{
				int type = oneDocTokens.getIndexAtPosition(si);
				pw1.print(" " + instances.getDataAlphabet().lookupObject(type));
				pw2.print(" " + (type + 1));
			}
			pw1.println("");
			pw2.println("");
		}
		pw1.flush();
		pw1.close();
		pw2.flush();
		pw2.close();
	}

}
