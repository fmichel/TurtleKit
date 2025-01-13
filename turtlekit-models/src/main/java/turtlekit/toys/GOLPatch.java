package turtlekit.toys;

import javafx.scene.paint.Color;

public class GOLPatch extends turtlekit.kernel.Patch {
	Color nextState = Color.WHITE;

	public Color getNextState() {
		return nextState;
	}

	public void setNextState(Color nextState) {
		this.nextState = nextState;
	}
}