package turtlekit.digitalart;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.TKDefaultViewer;
import turtlekit.viewer.TimeUnitsPerSecondCharter;

public class LangtonAnt extends Turtle {
	
	private Pheromone<Float> presence;

	@Override
	protected void activate() {
		super.activate();
		setNextAction("doIt");
		home();
  		presence = getEnvironment().getPheromone("presence",0.0f,0.9f);//TODO need milli precision
	}
	
	public String doIt(){
		presence.set(xcor(), ycor(), 10000f);
		return "doIt";
	}

	public static void main(String[] args) {
		executeThisTurtle(1
				,Option.envDimension.toString(),"512,512"
				,Option.viewers.toString(),TimeUnitsPerSecondCharter.class.getName()+";"+TKDefaultViewer.class.getName()
				,Option.startSimu.toString()
				,Option.cuda.toString()
				
				);
	}

}
