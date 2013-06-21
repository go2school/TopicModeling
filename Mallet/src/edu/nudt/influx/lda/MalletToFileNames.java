package edu.nudt.influx.lda;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import cc.mallet.types.InstanceList;

public class MalletToFileNames {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String fname = args[0];
		String outfname = args[1];
		
		InstanceList instances = InstanceList.load (new File(fname));
		PrintWriter pw = new PrintWriter(new FileWriter(outfname));
		for(int i=0;i<instances.size();i++)
		{
			pw.println(instances.get(i).getName() + " " + i);
		}
		pw.flush();
		pw.close();
	}

}
