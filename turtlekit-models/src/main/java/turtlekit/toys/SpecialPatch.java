package turtlekit.toys;

import turtlekit.kernel.Patch;

public class SpecialPatch extends Patch {
	
	private int value = 30;

	/**
	 * @return the value
	 */
	public int getValue() {
		return value;
	}

	/**
	 * @param value the value to set
	 */
	public void setValue(int value) {
		this.value = value;
	}

}
