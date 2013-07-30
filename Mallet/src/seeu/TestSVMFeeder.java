package seeu;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Problem;

public class TestSVMFeeder {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		int [] y = {1, -1, 1, -1};//label	
		int [][] topics = {{0,1,2,3,4,5,6,6,3,2,1,2}, {8,3,2,1,2,3,0,0,0,0,2,2,3}, {1,2,3,3,4,2,3,2,1}, {7,5,4,2,3,2}};//features
		int maxTopicID = 8;
		int [] buf = new int [maxTopicID + 1];
	
		List<Feature[]> vx = new ArrayList<Feature[]>();
		SVMFeeder.topicSequenceToFeatures(true, topics, vx, maxTopicID, buf);		
		Problem problem = SVMFeeder.constructProblem(y, vx, maxTopicID+1, -1);		 
		SVMFeeder.swapFirstNegative(problem);		
		
		Model model = SVMFeeder.trainBinarySVM(maxTopicID, y, vx);
		SVMFeeder.saveModel(model, "model");
		model = SVMFeeder.loadModel("model");
		int [] ret = new int [problem.l];
		SVMFeeder.predict(model, problem, ret);
		
		for(int i=0;i<problem.l;i++)
		{			
			System.out.println(ret[i]);
		}
		
	}

}
