package turtlekit.ants;

import static javafx.scene.paint.Color.GREEN;

import madkit.simulation.EngineAgents;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;

@EngineAgents(viewers = {AntViewer.class})
public class AntWorker extends DefaultTurtle {
	private Pheromone<Float> worker;

	@Override
	protected void onActivation() {
		super.onActivation();
		setColor(GREEN);
		worker = getEnvironment().getPheromone("worker",0.05f,0.8f);
		changeNextBehavior("work");
		home();
	}
	
	@Override
	protected void onStart() {
	}
	
	private void work(){
//		System.err.println(worker);
		worker.incValue(xcor(), ycor(), 5000);
	}

	public static void main(String[] args) {
		executeThisTurtle(1
				, "--width", "512", "--height", "512"
				, "--start"
//				,"--agentLogLevel","ALL"
//				,"--headless"
//				,"--cuda"
		// , startSimu.toString()
//				,"-v",AntViewer.class.getName()
//				,"-v",FXPheroViewer.class.getName()
//				,"-v",AntViewer.class.getName()
//				,"-v",TKViewer.class.getName()
		);
	}


}
