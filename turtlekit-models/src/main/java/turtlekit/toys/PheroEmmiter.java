package turtlekit.toys;


import javafx.scene.paint.Color;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;

public class PheroEmmiter extends DefaultTurtle {
	
	private Pheromone<?> pheromone;

	@Override
	protected void onActivation() {
		pheromone = getEnvironment().getPheromone("presence", 0.2f, 0.6f);
		getLogger().info(() -> pheromone.toString());
		super.onActivation();
		changeNextBehavior("fly");
		setColor(Color.color((prng().nextDouble()),prng().nextDouble(), prng().nextDouble()));
	}
	
	
	private void fly() {
		wiggle();
		pheromone.incValue(xcor(), ycor(), 100000);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(1
				,"--width","700"
				,"--height","700"
//				,"--noLog"
//				, "--cuda"
				,"--start"
				,"--tkLogLevel","ALL"
				);
	}

}
