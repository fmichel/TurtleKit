package turtlekit.kernel;

import javafx.scene.paint.Color;

public class DefaultTurtle extends Turtle<Patch> {

//	@Override
//	protected void onStart() {
//		setColor(Color.RED);
//		randomHeading();
//		randomLocation();
//	}
	
	@Override
	protected void onActivation() {
		super.onActivation();
		setColor(Color.RED);
		randomHeading();
		randomLocation();
	}
	
	public static void main(String[] args) {
		executeThisTurtle(10);
	}

}