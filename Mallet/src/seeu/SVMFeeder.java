package seeu;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class SVMFeeder {

	public static void swapFirstNegative(Problem problem)
	{
		if(problem.l > 0 && problem.y[0] < 0)
		{
			int i;
			for(i=1;i<problem.l;i++)
			{
				if(problem.y[i] > 0)
					break;
			}
			if(i != problem.l)
			{
				Feature [] f = problem.x[0];
				double y = problem.y[0];
				
				problem.x[0] = problem.x[i];
				problem.y[0] = problem.y[i];
				
				problem.x[i] = f;
				problem.y[i] = y;
			}
		}
	}
	
	
	public static Problem exampleSequenceToProblem(int [] y, double [][] data)
	{
		int num_ex = y.length;
		Problem problem = new Problem();
		problem.l = num_ex;
		problem.n = data[0].length;
		problem.x = new FeatureNode[num_ex][];
		problem.y = new double [num_ex];
		for(int i=0;i<num_ex;i++)
		{
			problem.y[i] = y[i];

			//check non-zero features
			int num_non_zero_features = 0;
			for(int j=0;j<data[i].length;j++)
			{
				if(data[i][j] != 0)
					num_non_zero_features++;
			}
			problem.x[i] = new FeatureNode [num_non_zero_features];
			int fs_index = 0;
			for(int j=0;j<data[i].length;j++)
			{
				if(data[i][j] != 0)
				{
					problem.x[i][fs_index] = new FeatureNode(j+1, data[i][j]);
					fs_index++;
				}
			}			
		}
		return problem;
	}
	
	/*
	 * topics: topic assignment at each position of a document, the topic index is starting at zero
	 * vx: features of documents
	 * numTopics: the total number of topics
	 */
	public static void topicSequenceToFeatures(boolean isTF, int [][] topics, List<Feature[]> vx,
					int maxTopicID, int [] buf)
	{
		vx.clear();
		
		int num_x = topics.length;
		for(int i=0;i<num_x;i++)
		{
			//reset buffer
			Arrays.fill(buf, 0);			
			
			//count topic frequency
			for(int j=0;j<topics[i].length;j++)
			{
				buf[topics[i][j]] ++;
			}
			
			//count number of non-empty words
			int numWords = 0;
			for(int t=0;t<maxTopicID+1;t++)
				if(buf[t] != 0)				
					numWords++;
			
			//create SVM features
			Feature[] x = new FeatureNode[numWords];
			int index = 0;
			for(int t=0;t<maxTopicID+1;t++)
			{
				if(buf[t] != 0)
				{
					if(isTF == true)
						x[index] = new FeatureNode(t+1, (double)buf[t]/topics[i].length);//t+1, because the feature ID must start at one
					else
						x[index] = new FeatureNode(t+1, buf[t]);//t+1, because the feature ID must start at one
					index++;
				}
			}
			
			vx.add(x);
		}
	}	
	
	/*
	 * docLabels: labels for all docs
	 * topics: infered topics by LDA
	 * parents: hierarchical relations of labels
	 * maxTopicID: the maximal topic ID index
	 * all_labels: all labels in the dataset
	 */
	public Hashtable<Integer, Model> trainHierarchicalClassifier(int [][] docLabels, int [][] topics, 
						int [] parents, int maxTopicID, int [] all_labels)
	{
		Hashtable<Integer, Model> all_models = new Hashtable<Integer, Model>();
		
		int num_ex = topics.length;
		
		//make problem
		int [] buf = new int [maxTopicID + 1];
		int [] tmp_y = new int [num_ex];
		Arrays.fill(tmp_y, 1);
		List<Feature[]> vx = new ArrayList<Feature[]>();
		SVMFeeder.topicSequenceToFeatures(true, topics, vx, maxTopicID, buf);		
		Problem problem = SVMFeeder.constructProblem(tmp_y, vx, maxTopicID+1, -1);		 
		
		//for each label, train a SVM
		int [] class_lst = new int [num_ex];	
		for(int i=0;i<all_labels.length;i++)
		{
			int label = all_labels[i] + 1;
			int parent_label = parents[label];
			Arrays.fill(class_lst, 0);
			
			System.out.println("Train " + (label - 1));
			
			//localize dataset
			int numd_training_ex = makeBinaryLabelLits(docLabels, label-1, parent_label-1, class_lst);
			//make problem
			if(numd_training_ex > 0)
			{
				Problem label_problem = constructProblem(problem, numd_training_ex, class_lst);
				SVMFeeder.swapFirstNegative(label_problem);
				
				//train the model
				Model model = trainBinarySVM(label_problem);
				//save the model
				all_models.put(label-1, model);
			}
		}
		
		return all_models;
	}

	public int [][] testHierarchicalClassifier(Hashtable<Integer, Model> all_models, 
				int [][] topics, int [][] parent2children, int maxTopicID)
	{		
		int num_ex = topics.length;
		
		int [][] test_labels = new int [num_ex][];
		//make problem
		int [] buf = new int [maxTopicID + 1];
		int [] tmp_y = new int [num_ex];
		Arrays.fill(tmp_y, 1);
		List<Feature[]> vx = new ArrayList<Feature[]>();
		SVMFeeder.topicSequenceToFeatures(true, topics, vx, maxTopicID, buf);		
		Problem problem = SVMFeeder.constructProblem(tmp_y, vx, maxTopicID+1, -1);		 
		
		TreeSet<Integer> predicted_labels = new TreeSet<Integer>();
		//from top level to the bottom level, do prediction
		for(int index_ex = 0;index_ex < problem.l;index_ex++)
		{
			Feature [] ex = problem.x[index_ex];
			
			predicted_labels.clear();
			makePrediction(0, parent2children, ex, all_models, predicted_labels);
			
			test_labels[index_ex] = new int [predicted_labels.size()];
			Iterator<Integer> it = predicted_labels.iterator();
			int l_index = 0;
			while(it.hasNext())
			{
				int label = it.next();
				test_labels[index_ex][l_index] = label;
				l_index ++ ;
			}			
		}
		return test_labels;
	}
	
	public void makePrediction(int root, int [][] parent2children, Feature [] ex, Hashtable<Integer, Model> all_models,
			TreeSet<Integer> predicted_labels)
	{
		//remember!! 
		//in parent2children, every label is added by one
		//so in classifiers, we need to subtract it by one
		if(parent2children[root] != null)
		{
			for(int i=0;i<parent2children[root].length;i++)
			{
				int label = parent2children[root][i] - 1;
				Model model = all_models.get(label);
				if(model != null)
				{
					int predict_label = (int) Linear.predict(model, ex);
					if(predict_label > 0)
					{
						predicted_labels.add(label);
						makePrediction(parent2children[root][i], parent2children, ex, all_models, predicted_labels);
					}
				}
			}
		}
	}
	
	/*
	 * Used for generating binary label set for each node of hierarchy
	 */
	public int makeBinaryLabelLits(int [][] docLabels, int pos_label, int parent_pos_label, int [] class_lst)
	{
		int num_ex = 0;
		int num_pos = 0;
		int num_neg = 0;
		for(int i=0;i<docLabels.length;i++)
		{
			int is_parent_pos = 0;
			int is_pos = 0;
			for(int j=0;j<docLabels[i].length;j++)
			{
				if(pos_label == docLabels[i][j])
				{
					is_pos = 1;
				}
				else if(parent_pos_label == docLabels[i][j])
				{
					is_parent_pos = 1;
				}
			}
			//either first level or contains the parent label
			if(is_parent_pos == 1 || parent_pos_label == -1)
			{
				if(is_pos == 1)
				{
					class_lst[i] = 1;
					num_pos++;
				}
				else
				{
					class_lst[i] = -1;
					num_neg++;
				}
				num_ex++;
			}
		}
		System.out.println(pos_label + ": " + num_pos + ", " + num_neg + ", " + num_ex);
		return num_ex;
	}
	
	/*
	 * make binary dataset by selecting pos and negative data
	 * +1 in class_lst: pos
	 * -1 in class_lst: neg
	 */
	public static Problem constructProblem(Problem problem, int num_ex, int [] class_lst)
	{
		Problem new_problem = new Problem();
		new_problem.l = num_ex;
		new_problem.n = problem.n;
		new_problem.y = new double [num_ex];
		new_problem.x = new Feature[num_ex][];
		int ex_index = 0;
		for(int i=0;i<problem.l;i++)
		{
			if(class_lst[i] == 1)
			{
				new_problem.x[ex_index] = problem.x[i];
				new_problem.y[ex_index] = 1;
				ex_index++;
			}
			else if(class_lst[i] == -1)
			{
				new_problem.x[ex_index] = problem.x[i];
				new_problem.y[ex_index] = -1;
				ex_index++;
			}
		}
		return new_problem;
		
	}
	
	public static Problem constructProblem(int [] vy, List<Feature[]> vx, int max_index, double bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.length;
        prob.n = max_index;
        if (bias >= 0) {
            prob.n++;
        }
        prob.x = new Feature[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.get(i);

            if (bias >= 0) {
                assert prob.x[i][prob.x[i].length - 1] == null;
                prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
            }
        }

        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy[i];

        return prob;
    }
	
	
	public static Model trainBinarySVM(int maxTopicID, int [] y, List<Feature[]> vx)
	{		
		Problem problem = constructProblem(y, vx, maxTopicID+1, -1);
		 
		swapFirstNegative(problem);				
		
		SolverType solver = SolverType.L2R_L2LOSS_SVC_DUAL; // -s 0
		double C = 1.0;    // cost of constraints violation
		double eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);
		
		return model;
	}
	
	public static Model trainBinarySVM(Problem problem)
	{		
		swapFirstNegative(problem);				
		
		SolverType solver = SolverType.L2R_L2LOSS_SVC_DUAL; // -s 0
		double C = 1.0;    // cost of constraints violation
		double eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);
		
		return model;
	}
	
	public static void saveModel(Model model, String fname) throws IOException
	{
		File modelFile = new File(fname);
		model.save(modelFile);
	}
	
	public static Model loadModel(String fname) throws IOException
	{
		File modelFile = new File(fname);
		Model model = Model.load(modelFile);
		return model;
	}
	
	public static void predict(Model model, Problem problem, int [] ret)
	{
		for(int i=0;i<problem.l;i++)
		{
			double prediction = Linear.predict(model, problem.x[i]);
			ret[i] = (int)prediction;
		}
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 * @throws InvalidInputDataException 
	 */
	public static void main(String[] args) throws IOException, InvalidInputDataException {
		// TODO Auto-generated method stub
		Problem problem = Problem.readFromFile(new File("d:/a6a.txt"), -1);	
		
		//double data [][] = {{1,2,0},{3,0,1},{0,2,3},{0,0,1}};
		//Problem problem = exampleSequenceToProblem(y, data);
		
		/*
		int [] y = {1, -1, 1, -1};//label	
		int [][] topics = {{0,1,2,3,4,5,6,6,3,2,1,2}, {8,3,2,1,2,3,0,0,0,0,2,2,3}, {1,2,3,3,4,2,3,2,1}, {7,5,4,2,3,2}};//features
		int maxTopicID = 8;
		int [] buf = new int [maxTopicID + 1];
		
		List<Feature[]> vx = new ArrayList<Feature[]>();
		topicSequenceToFeatures(true, topics, vx, maxTopicID, buf);//all old topics are increased by one	
		Problem problem = constructProblem(y, vx, maxTopicID+1, -1);
		 */
		
		swapFirstNegative(problem);				
		
		
		//Model model = trainBinarySVM(maxTopicID, y, vx);
		
		
		SolverType solver = SolverType.L2R_L2LOSS_SVC_DUAL; // -s 0
		double C = 1.0;    // cost of constraints violation
		double eps = 0.01; // stopping criteria

		Parameter parameter = new Parameter(solver, C, eps);
		Model model = Linear.train(problem, parameter);
		
		//File modelFile = new File("model");
		//model.save(modelFile);
		// load model or use it directly
		//model = Model.load(modelFile);
		
		for(int i=0;i<problem.l && i< 5;i++)
		{
			double prediction = Linear.predict(model, problem.x[i]);
		
			System.out.println(prediction + ", " + problem.y[i]);
		}
	}

}
