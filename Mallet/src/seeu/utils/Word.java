package seeu.utils;

public class Word implements Comparable {
	int wi;
	double p;
	public Word (int wi, double p) { this.wi = wi; this.p = p; }
	public final int compareTo (Object o2) {
		if (wi > ((Word)o2).wi)
			return -1;
		else if (wi == ((Word)o2).wi)
			return 0;
		else return 1;
	}
}