package seeu;

public class IDSorter  implements Comparable {
	int wi; double p;
	public IDSorter (int wi, double p) { this.wi = wi; this.p = p; }
	public final int compareTo (Object o2) {
		if (p > ((IDSorter) o2).p)
			return -1;
		else if (p == ((IDSorter) o2).p)
			return 0;
		else return 1;
	}
	
	public int getID() {return wi;}
	public double getWeight() {return p;}

	/** Reinitialize an IDSorter */
	public void set(int id, double p) { this.wi = id; this.p = p; }
	
}