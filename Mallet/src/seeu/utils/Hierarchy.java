package seeu.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.TreeSet;

import cc.mallet.util.Randoms;

public class Hierarchy {

	//use array for fast index and search
	public int [][] parent2children; 
	public int [] parents;
	public int root;
	public int max_level;
	public int num_nodes;
	public int [] level;
	
	public Hierarchy(){}
	
	public int [][] getParent2Children()
	{
		return parent2children;
	}
	
	public int [] getParents()
	{
		return parents;
	}
	
	public void sampleAPath(Randoms r, int [] path)
	{
		//sample a path from root
		int current_node = root;
		int level = 0;
		while(parent2children[current_node] != null)
		{
			//System.out.println("aa " + current_node + " " + parent2children[current_node].length);
			int num_children = parent2children[current_node].length;
			int child_index = r.nextInt(num_children + 1);
			if(child_index != num_children)//not fall on exit node
			{
				path[level] = parent2children[current_node][child_index];
				current_node = parent2children[current_node][child_index];
				level ++;
			}			
		}
	}
	
	public void sampleAPath(Randoms r, int [] path, 
				TreeSet<Integer> used_nodes, double block_threshold)
	{		
		//sample a path from root
		int current_node = root;
		int level = 0;
		int [] childrens = new int [200];
		while(parent2children[current_node] != null)
		{
			//System.out.println("aa " + current_node + " " + parent2children[current_node].length);
			int num_children = parent2children[current_node].length;
			int num_used_children = 0;
			for(int i=0;i<num_children;i++)
			{
				int cti = parent2children[current_node][i];
				if(used_nodes.contains(cti))
				{
					childrens[num_used_children] = cti;
					num_used_children++;
				}
			}
			
			int child_index = r.nextInt(num_used_children);
			int sampled_cti = childrens[child_index];
			
			path[level] = sampled_cti;
			level++;
			
			//check if go deeper
			if(r.nextDouble() >= block_threshold)
			{
				//go deeper
				current_node = sampled_cti;
			}
			else
				break;
		}
		path[level] = -1;
	}
	
	public void readTree_parent2child(String fname) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(fname));
		String line = "";
		TreeSet<Integer> all_nodes = new TreeSet<Integer>();
		//read all nodes
		ArrayList<Integer> parent_nodes = new ArrayList<Integer>(); 
		ArrayList<Integer> child_nodes = new ArrayList<Integer>();
		Hashtable<Integer, ArrayList<Integer>> p2c = new Hashtable<Integer, ArrayList<Integer>>(); 
		int max_id = -1;
		while((line = br.readLine()) != null)
		{
			String [] fields = line.split(" ");			
			int parent = Integer.parseInt(fields[0]);
			int child = Integer.parseInt(fields[1]);
			all_nodes.add(parent);
			all_nodes.add(child);
			parent_nodes.add(parent);
			child_nodes.add(child);
			if(parent > max_id)
				max_id = parent;
			if(child > max_id)
				max_id = child;
			
			if(p2c.containsKey(parent))
			{
				p2c.get(parent).add(child);				
			}
			else
			{
				ArrayList<Integer> c = new ArrayList<Integer>();
				c.add(child);
				p2c.put(parent, c);
			}			
		}
		br.close();
		
		System.out.println(max_id + " " + all_nodes.size());
		
		parent2children = new int [max_id+1][];
		parents = new int [max_id+1];		
		for(int i=0;i<max_id+1;i++)
		{
			parents[i] = -1;
		}
		for (Integer key: p2c.keySet())
		{
			int parent = key;
			ArrayList<Integer> c = p2c.get(key);
			parent2children[parent] = new int[c.size()];
			for(int j=0;j<c.size();j++)
			{
				parent2children[parent][j] = c.get(j);
				parents[c.get(j)] = parent;
				root = parent;//for find root
			}			
		}
		//find root
		int p = root;
		while(parents[p] != -1)
			p = parents[p];
		root = p;
		
		//get number of all nodes
		num_nodes = all_nodes.size();
		
		//get level for each node
		level = new int [max_id+1];
		for(int i=0;i<parents.length;i++)
			level[i] = -1;
		max_level = -1;
		for(int i=0;i<parents.length;i++)
		{
			if(parents[i] != -1)
			{
				p = i;
				level[i] ++;
				while(parents[p] != -1)
				{
					p = parents[p];
					level[i] ++;
				}
				if(level[i] > max_level)
					max_level = level[i];
			}
		}
	}
	
	void printTree()
	{
		for(int i=0;i<parent2children.length;i++)
		{
			if(parent2children[i] != null)
			{
				System.out.print(i);
				for(int j=0;j<parent2children[i].length;j++)
					System.out.print(" " + parent2children[i][j]);
				System.out.println("");
			}
		}
		for(int i=0;i<parents.length;i++)
		{
			if(parents[i] != -1)
				System.out.println(i + " " + parents[i] + " " + level[i]);
		}
		System.out.println("Root is " + root + ", max level is " + max_level);
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		Hierarchy h = new Hierarchy();
		h.readTree_parent2child("D:/mallet-2.0.7/workspace/test_tree.txt");
		//h.printTree();
	
		TreeSet<Integer> used_nodes = new TreeSet<Integer>();
		used_nodes.add(2);
		used_nodes.add(5);
		//used_nodes.add(6);
		
		int [] path = new int [h.max_level + 1];
		for(int k=0;k<10;k++)
		{
			h.sampleAPath(new Randoms(), path, used_nodes, 0.5);
			System.out.print(k + "  => ");
			for(int i=0;i<path.length && path[i] != 0;i++)
				System.out.print(path[i] + " ");
			System.out.println("");
		}
	}

}
