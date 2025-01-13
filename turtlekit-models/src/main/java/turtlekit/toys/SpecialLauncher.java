package turtlekit.toys;

import turtlekit.kernel.TurtleKit;

public class SpecialLauncher extends TurtleKit {
	
	@Override
	protected void onLaunchSimulatedAgents() {
		for (int i = 0; i < 100; i++) {
			launchAgent(new SpecialPatchTurtle());
		}
	}

	public static void main(String[] args) {
		executeThisLauncher("--patchClass",SpecialPatch.class.getName());
	}

}
