package turtlekit.toys;

import turtlekit.kernel.TKEnvironment;

public class SpecialEnvironment extends TKEnvironment<SpecialPatch>{
	
	private int value = 10;

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
