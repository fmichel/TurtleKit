/**
 * 
 */
package turtlekit.ants;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.BLUE;
import static javafx.scene.paint.Color.GREEN;
import static javafx.scene.paint.Color.YELLOW;

import java.util.concurrent.atomic.AtomicInteger;

import turtlekit.kernel.Patch;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.jfx.FXPheroViewer;
import madkit.simulation.EngineAgents;

/**
 * @author fab
 */

@EngineAgents(viewers = { AntViewer.class })
public class Ant extends DefaultTurtle {

	private Pheromone<Float> presence;
	Pheromone<Float> worker;
	private Pheromone<Float> path;
	public static AtomicInteger collected = new AtomicInteger(0);

	@Override
	protected void onActivation() {
		super.onActivation();
//		getLogger().setLevel(Level.FINEST);
		setColor(GREEN);
		home();
		randomHeading();
		presence = getEnvironment().getPheromone("presence", 0.001f, 0.009f);// TODO need milli precision
//		presence = getEnvironment().getSobelPheromone("presence", 0.01f, 0.009f);// TODO need milli precision
//		worker = getEnvironment().getPheromone("worker",1,80);
		changeNextBehavior("searchForFood");

		path = getEnvironment().getPheromone("path", 0.01f, 0.009f);// TODO need milli precision
}

	@Override
	protected void onStart() {
		randomHeading();
		for (int i = 0; i < 20; i++) {
			emitPresence();
			fd(1);
		}
	}

	public void searchForFood() {
		worker = getPheromone("worker");
		towardsMinGradientField(presence);
		emitPresence();
//		if (presence.get(xcor(), ycor()) < 5) {
//			advertise(0, 2);
//		}
		wiggle(10);
		if (getPatchColor() == YELLOW) {
//			Object o = null;
//			o.toString();
			int foodQty = (int) getPatch().getMark("food");
			if (foodQty-- > 0) {
				getPatch().dropObject("food", Integer.valueOf(foodQty));
			} else {
				setPatchColor(BLACK);
			}
			advertise(0, 3);
			presence.set(xcor(), ycor(), -10f);
			setColor(BLUE);
			changeNextBehavior("returnToNest");
		}
	}

	private void advertise(float d, int radius) {
		for (Patch p : getPatch().getNeighbors(radius, true)) {
			presence.set(p.x, p.y, d);
		}
	}

	private void returnToNest() {
		final int currentBehaviorCount = getCurrentBehaviorCount();
		if (currentBehaviorCount > 4) {
//			advertise(currentBehaviorCount * 0.1f, 1);
			advertise(0, 1);
		}
		emitPath();
		// presence.set(xcor(), ycor(), 0);
		towardsMaxGradientField(worker);
		wiggle(10);
		if (xcor() == getEnvironment().getWidth() / 2 && ycor() == getEnvironment().getHeight() / 2) {
//			System.err.println(collected.incrementAndGet());
			setColor(GREEN);
			changeNextBehavior("searchForFood");
		}
	}

	private void emitPresence() {
		presence.incValue(xcor(), ycor(), getCurrentBehaviorCount() * .1f);
	}

	private void emitPath() {
		path.incValue(xcor(), ycor(), 1000f);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(2000, "--turtles", AntWorker.class.getName() + ",1", "--width", "512", "--height", "512",
				"--start"
//				,"--agentLogLevel","ALL"
//				,"--headless"
				, "--cuda"
		// , startSimu.toString()
//				,"-v",AntViewer.class.getName()
//				,"-v",FXPheroViewer.class.getName()
//				,"-v",AntViewer.class.getName()
//				,"-v",TKViewer.class.getName()
		);
	}

}
