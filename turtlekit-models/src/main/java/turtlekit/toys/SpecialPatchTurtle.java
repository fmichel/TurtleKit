package turtlekit.toys;

import turtlekit.kernel.Turtle;

public class SpecialPatchTurtle extends Turtle<SpecialPatch>{
	
	
	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("fake");
		randomLocation();
		randomHeading();
	}
	
	@Override
	public SpecialEnvironment getEnvironment() {
		return getLauncher().getEnvironment();
	}
	
	
	private String fake() {
		wiggle();
		System.err.println(getPatch().getValue());
		System.err.println(getEnvironment().getValue());
		return "fake";
	}

	public static void main(String[] args) {
		executeThisTurtle(10
				,"--environment",SpecialEnvironment.class.getName()
				);
	}

}
