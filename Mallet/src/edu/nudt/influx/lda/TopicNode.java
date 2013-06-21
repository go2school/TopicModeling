package edu.nudt.influx.lda;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import cc.mallet.util.Randoms;


public class TopicNode {

	//PAM variables
	public double [] subAlphas;
	public double subAlphaSum;
	public double alpha;
	
	public int numSubTopics = 0;
	public int [] subCounts;
	
	//public HashMap<Integer, Double> subWeights = new HashMap<Integer, Double>();
	
	//for tree data structure
	public ArrayList<TopicNode> subNodes = new ArrayList<TopicNode>();
	public String name = "EMPTY";
	public int labelIndex = 0;
	public int parent = -1;	
	public int subTreeSize = -1;
	public int height = 0;
	
	public static int maxNodes = 1000;
	
	//build tree from <par, child> pair file
	public static TopicNode buildHierarchy(String parChildPairfile) throws IOException
	{
		//make buffer for all nodes
		TopicNode [] tmp = new TopicNode[maxNodes];
		for(int i=0;i<tmp.length;i++)
		{
			tmp[i] = new TopicNode();
			tmp[i].labelIndex = i;
		}
		
		//read <par, child> pair of nodes
		int par = 0, child = 0;
		BufferedReader br = new BufferedReader(new FileReader(parChildPairfile));
		String buffer = null;
		while((buffer = br.readLine()) != null)
		{
			String [] lst = buffer.split(",");
			par = Integer.parseInt(lst[0]);
			child = Integer.parseInt(lst[1]);
			tmp[par].subNodes.add(tmp[child]);
			tmp[par].numSubTopics ++;
			tmp[child].parent = par;
			tmp[child].height = tmp[par].height + 1;
		}
		br.close();
		
		//find root
		int root = child;
		while(tmp[root].parent != -1)
		{
			root = tmp[root].parent;
		}
		
		TopicNode rootNode = tmp[root];
		
		return rootNode;
	}
	
	public void getNodes(HashMap<Integer, TopicNode> topic2node)
	{
		topic2node.put(this.labelIndex, this);
		for(int i=0;i<numSubTopics;i++)
		{			
			subNodes.get(i).getNodes(topic2node);
		}
	}
	
	public void getParents(HashMap<Integer, Integer> parents)
	{
		for(int i=0;i<numSubTopics;i++)
		{
			parents.put(subNodes.get(i).labelIndex, labelIndex);
			subNodes.get(i).getParents(parents);
		}
	}
	
	//allocate data structure for the subtree
	public void initTree(double alphaSum)
	{	
		this.subAlphaSum = alphaSum;
		
		if(numSubTopics != 0)
		{								
			subCounts = new int [numSubTopics];
			subAlphas = new double [numSubTopics];
			Arrays.fill(subAlphas, alphaSum / numSubTopics);
			for(int i=0;i<numSubTopics;i++)
			{	
				subNodes.get(i).initTree(alphaSum);
				subNodes.get(i).alpha = alphaSum / numSubTopics;
			}
		}		
	}		
	
	public void getAllAlpha(double [] alphas)
	{
		for(int i=0;i<numSubTopics;i++)
		{
			int labelIndex = subNodes.get(i).labelIndex;
			double alpha = subNodes.get(i).alpha;
			alphas[labelIndex] = alpha; 
			
			subNodes.get(i).getAllAlpha(alphas);
		}
	}
	
	public int getHeight()
	{		
		int maxHeight = 0;
		for(int i=0;i<subNodes.size();i++)
		{
			int h = subNodes.get(i).getHeight();
			if(h > maxHeight)
				maxHeight = h;
		}
		return maxHeight + 1;
	}
	
	public void getIntermediaNodes(ArrayList<Integer> interNodes)
	{
		if(this.subNodes.size() != 0)
		{
			if(this.labelIndex != 0)
				interNodes.add(this.labelIndex);
			for(int i=0;i<this.subNodes.size();i++)
			{
				subNodes.get(i).getIntermediaNodes(interNodes);
			}
		}
	}
	
	public void getAllLeaves(ArrayList<Integer> leaves)
	{
		for(int i=0;i<this.numSubTopics;i++)
		{
			if(subNodes.get(i).numSubTopics == 0)
				leaves.add(subNodes.get(i).labelIndex);
			else
				subNodes.get(i).getAllLeaves(leaves);
		}
	}
	
	public void printTree(int depth)
	{
		for(int i=0;i<depth;i++)
			System.out.print("\t");
		System.out.print(this.labelIndex);
		System.out.print("\n");
		for(int i=0;i<subNodes.size();i++)
		{
			subNodes.get(i).printTree(depth + 1);
		}		
	}
	
	public void printTreeToFile(int depth, PrintWriter pw)
	{
		for(int i=0;i<depth;i++)
			pw.print("\t");
		if(subNodes.size() == 0)
			pw.print("<node id=\"" + this.labelIndex + "\" height=\"" + this.height + "\"/>\n");
		else
		{
			pw.print("<node id=\"" + this.labelIndex + "\" height=\"" + this.height + "\">\n");			
			for(int i=0;i<subNodes.size();i++)
			{
				subNodes.get(i).printTreeToFile(depth + 1, pw);
			}		
			for(int i=0;i<depth;i++)
				pw.print("\t");
			pw.print("</node>\n");
		}
	}	
	
	public int getTreeSize()
	{
		int sum = 1;
		for(int i=0;i<this.subNodes.size();i++)
			sum += this.subNodes.get(i).getTreeSize();
		this.subTreeSize = sum;
		return sum;
	}
	
	public int [] samplingPath(Randoms r)
	{
		int [] path = null;
		ArrayList<Integer> tmpList = new ArrayList<Integer>();
		TopicNode curNode = this;
		while(curNode.numSubTopics != 0)
		{
			//sample a node
			int nd = r.nextInt(curNode.numSubTopics);
			TopicNode nextNode = curNode.subNodes.get(nd);
			tmpList.add(nextNode.labelIndex);
			curNode = nextNode;
		}
		path = new int [tmpList.size()];
		for(int i=0;i<path.length;i++)
			path[i] = tmpList.get(i);
		return path;
	}
	
	public static void test() throws IOException
	{
		TopicNode tmp = new TopicNode();
		String hierFname = "/home/xiao/workspace/test/sf_hier_par_child_pairs.txt";
		String xmlHierFname = "/home/xiao/workspace/test/sf_hier_par_child_pairs.xml";
		TopicNode root = tmp.buildHierarchy(hierFname);
		
		root.initTree(0.5);
		int tree_size = root.getTreeSize();
		System.out.println(tree_size);
		root.printTree(0);
		PrintWriter pw = new PrintWriter(new FileWriter(xmlHierFname));		
		root.printTreeToFile(0, pw);
		pw.flush();
		pw.close();
		
		Randoms r = new Randoms();
		int [] path = null;
		for(int j=0;j<1000;j++)
		{
			path = root.samplingPath(r);
			for(int i=0;i<path.length;i++)
				System.out.print(path[i] + " ");
			//if(path.length >= 2)
			//	System.out.println("More than two");
			System.out.println("");
		}
		
		System.out.println("Height " + root.getHeight());
	}
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub	
		test();
	}

}
