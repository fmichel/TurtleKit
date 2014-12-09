package turtlekit.langtonAnts;

import java.awt.Color;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.StatesPerSecondCharter;
import turtlekit.viewer.TKDefaultViewer;

public class LangtonAnt extends Turtle {
	
	@Override
	protected void activate() {
		super.activate();
		setNextAction("doIt");
		home();
		fd(0.5);
		setHeading(90);
		fd(0.5);
	}
	
	public String doIt(){
		if(getPatchColor() == Color.BLACK){
			setPatchColor(Color.WHITE);
			turnLeft(90);
		}
		else {
			setPatchColor(Color.BLACK);
			turnRight(90);
		}
		fd(1);
		return "doIt";
	}

	public static void main(String[] args) {
		executeThisTurtle(1
				,Option.envDimension.toString(),"1000,1000"
				,Option.renderingInterval.toString(),"550"
				,Option.viewers.toString(),StatesPerSecondCharter.class.getName()+";"+TKDefaultViewer.class.getName()
				,Option.startSimu.toString()
				);
	}

}
