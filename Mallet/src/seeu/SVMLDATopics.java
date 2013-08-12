package seeu;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.io.FileReader;
import java.io.IOException;

import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Set;

import java.util.TreeSet;

import de.bwaldvogel.liblinear.Model;

public class SVMLDATopics {

	public static int [][] parent2children;
	public static int [] parents;
	
	public static int [][] readLabels(String fname) throws IOException
	{
		//read label
		int num_ex = 0;
		BufferedReader br = new BufferedReader(new FileReader(fname));
		String buf = "";
		while((buf = br.readLine()) != null)
		{
			num_ex++;
		}
		br.close();
		
		int [][] labels = new int [num_ex][];
		br = new BufferedReader(new FileReader(fname));
		int index = 0;
		while((buf = br.readLine()) != null)
		{
			String [] fields = buf.split(" ");
			int numLabels = Integer.parseInt(fields[1]);
			labels[index] = new int [numLabels];
			for(int i=2;i<fields.length;i++)
				labels[index][i-2] = Integer.parseInt(fields[i]);
			index++;
		}
		br.close();
		
		return labels;
	}
	
	public static int [][] readTopics(String fname) throws IOException
	{
		//read number of ex
		int num_ex = 0;
		BufferedReader br = new BufferedReader(new FileReader(fname));
		String buf = "";
		while((buf = br.readLine()) != null)
		{
			num_ex++;
		}
		br.close();
		
		//read topics
		int [][] topics = new int [num_ex][];
		br = new BufferedReader(new FileReader(fname));
		int index = 0;
		while((buf = br.readLine()) != null)
		{
			String [] fields = buf.split(" ");
			int numLabels = fields.length;
			topics[index] = new int [numLabels];
			for(int i=0;i<fields.length;i++)
				topics[index][i] = Integer.parseInt(fields[i]);
			index++;
		}
		br.close();
		
		return topics;
	}
	
	/*
	 * In the hierarchy, every label is added by one
	 */
	public static void readParentChildHierarchy(String fname, int root, TreeSet<Integer> all_labels) throws NumberFormatException, IOException
	{
		int maxCatID = -1;
		String buf = "";
		Hashtable<Integer, TreeSet<Integer>> p2c = new Hashtable<Integer, TreeSet<Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(fname));	
		while((buf = br.readLine()) != null)
		{
			String [] fields = buf.split(",");
			int parent = Integer.parseInt(fields[0]);
			int child = Integer.parseInt(fields[1]);
			if(parent > maxCatID)
				maxCatID = parent;
			if(child > maxCatID)
				maxCatID = child;
			//add all labels
			if(parent != root)
				all_labels.add(parent);
			all_labels.add(child);
			//add p2c
			if(p2c.containsKey(parent))
			{
				TreeSet<Integer> c = p2c.get(parent);
				c.add(child);
			}
			else
			{
				TreeSet<Integer> c = new TreeSet<Integer>();
				c.add(child);
				p2c.put(parent, c);
			}
		}
		br.close();
		
		parents = new int [maxCatID + 1];
		Arrays.fill(parents, -1);
		
		br = new BufferedReader(new FileReader(fname));	
		while((buf = br.readLine()) != null)
		{
			String [] fields = buf.split(",");
			int parent = Integer.parseInt(fields[0]);
			int child = Integer.parseInt(fields[1]);	
			
			parents[child] = parent;
		}
		br.close();	
		
		parent2children = new int [maxCatID + 1][];
		Set<Integer> keys = p2c.keySet();
        for(Integer key: keys){
        	TreeSet<Integer> c = p2c.get(key);
        	parent2children[key] = new int [c.size()];
        	Iterator<Integer> it = c.iterator();
        	int index = 0;
        	while(it.hasNext())
        	{
        		int l = it.next();
        		parent2children[key][index] = l;
        		index++;
        	}
        }
	}
		
	public static int getMaxTopicID(int [][] topics)
	{
		int maxID = -1;
		for(int i=0;i<topics.length;i++)
			for(int j=0;j<topics[i].length;j++)
				if(maxID < topics[i][j])
					maxID = topics[i][j];
		return maxID;
	}
	
	public static int [] getAllLabels(int [][] labels)
	{
		TreeSet<Integer> allLabels = new TreeSet<Integer>();
		for(int i=0;i<labels.length;i++)
			for(int j=0;j<labels[i].length;j++)
				allLabels.add(labels[i][j]);
		
		int [] ret_labels = new int [allLabels.size()];		
		int ti = 0;
		Iterator<Integer> it = allLabels.iterator();
		while(it.hasNext()) {
			Integer a = it.next();	
			ret_labels[ti] = a;
		    ti++;
		}
		return ret_labels;
	}	
	
	public static void saveModels(String odir, int [] allLabels, Hashtable<Integer, Model> allModels) throws IOException
	{
		for(int i=0;i<allLabels.length;i++)
		{
			int label = allLabels[i];
			Model model = allModels.get(label);
			if(model != null)
			{
				File f = new File(odir + "/" + label + ".model");
				model.save(f);
			}
		}
	}
	
	public static void loadModels(String odir, TreeSet<Integer> allLabels, Hashtable<Integer, Model> allModels) throws IOException
	{
		Iterator<Integer> it = allLabels.iterator();
		while(it.hasNext())
		{
			int label = it.next()-1;
			Model model = new Model();
			File f = new File(odir + "/" + label + ".model");
			model = Model.load(f);	
			allModels.put(label, model);
		}
	}
	
	public static void writePrediction(int [][] prediction, String fname) throws IOException
	{
		PrintWriter pw = new PrintWriter(new FileWriter(fname));
		for(int i=0;i<prediction.length;i++)
		{
			pw.write(i + " " + prediction[i].length);
			for(int j=0;j<prediction[i].length;j++)
				pw.write(" " + prediction[i][j]);
			pw.write("\n");
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
				
		String trainLabelFname = "";
		String featureFname = "";
		String hierFname = "";
		String outDir = "";		
		String mode = "";
	
		// TODO Auto-generated method stub
	    for (int i = 0; i < args.length; i++) {
    	  if (args[i].equals("-mode")) {
    		  mode = args[++i];
	      }		      
	      else if (args[i].equals("-label")) {
	    	  trainLabelFname = args[++i];
	      }		      
	      else if (args[i].equals("-topics")) {
	    	  featureFname = args[++i];
	      }
	      else if (args[i].equals("-odir")) {
	    	  outDir = args[++i];
	      }	
	      else if (args[i].equals("-hier")) {
	    	  hierFname = args[++i];
	      }	
	    	  
	    }	
		 
		if(mode.equalsIgnoreCase("train"))
		{
			int [][] labels = readLabels(trainLabelFname);
			int [][] topics = readTopics(featureFname);
			TreeSet<Integer> all_labels = new TreeSet<Integer>();
			readParentChildHierarchy(hierFname, 0, all_labels);//in the relations, each label is added by one
			
			int maxTopicID = getMaxTopicID(topics);//get the maxmal topic feature ID
			int [] allLabels = getAllLabels(labels);//get all labels from the training data\
			
			SVMFeeder s = new SVMFeeder();
			
			Hashtable<Integer, Model> allModels = s.trainHierarchicalClassifier(labels, topics, parents, maxTopicID, allLabels);
			saveModels(outDir, allLabels, allModels);
		}
		else if(mode.equalsIgnoreCase("test"))
		{
			int [][] topics = readTopics(featureFname);
			TreeSet<Integer> all_labels = new TreeSet<Integer>();
			readParentChildHierarchy(hierFname, 0, all_labels);//in the relations, each label is added by one
			int maxTopicID = getMaxTopicID(topics);
					
			SVMFeeder s = new SVMFeeder();
			Hashtable<Integer, Model> allModels = new Hashtable<Integer, Model>();
			loadModels(outDir, all_labels, allModels);
			
			int [][] predicted_labels = s.testHierarchicalClassifier(allModels, topics, parent2children, maxTopicID);
			
			//write predicted labels
			writePrediction(predicted_labels, outDir + "/prediction");			
		}
	}
}
