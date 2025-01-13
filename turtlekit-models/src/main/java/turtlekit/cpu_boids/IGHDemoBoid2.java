package turtlekit.cpu_boids;

import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;
import static javafx.scene.paint.Color.BLUE;
import static javafx.scene.paint.Color.CYAN;
import static javafx.scene.paint.Color.GREEN;
import static javafx.scene.paint.Color.RED;
import static javafx.scene.paint.Color.WHITE;
import static turtlekit.cpu_boids.FlockingModel.BOID_FOV;
import static turtlekit.cpu_boids.FlockingModel.GROUP_SIZE;
import static turtlekit.cpu_boids.FlockingModel.MAX_ALIGN_TURN;
import static turtlekit.cpu_boids.FlockingModel.MAX_NEIGHBORS;

import java.util.ArrayList;
import java.util.List;


/**
 * BirdFlocking represents the "Bird" of the simulation
 * <p>
 * A "Bird" agent is characterized by:
 * <ul>
 * <li>A Field of View</li>
 * <li>A Speed</li>
 * <li>Rotation Angles</li>
 * <li>A neighbor list</li>
 * </ul>
 * </p>
 * 
 * @author Emmanuel Hermellin
 * @author Fabien Michel
 * @version 0.1
 * 
 * @see turtlekit.kernel.DefaultTurtle
 * 
 */

public class IGHDemoBoid2 extends AbstractBoid {

	protected AbstractBoid nearestBirdInFov;
	protected AbstractBoid nearestBird;
	protected List<AbstractBoid> neighbors;
	protected List<AbstractBoid> neighborsInFov;

	public IGHDemoBoid2() {
		super("wander");
	}

	@Override
	protected void onActivation() {
		super.onActivation();
		randomLocation();
		randomHeading();

	}

	private void updatePerceptions() {
		nearestBird = null;
		nearestBirdInFov = null;
		neighborsInFov = new ArrayList<>(0);
		neighbors = getPatch().getTurtles((int) BOID_FOV, false, AbstractBoid.class);

		if (!neighbors.isEmpty()) {
			final int nb = (int) MAX_NEIGHBORS;
			final int size = neighbors.size();
			if (size > nb) {
				neighbors = neighbors.subList(0, nb > size ? size : nb);
			}
			nearestBird = neighbors.get(0);
//	    for (AbstractBoid other : neighbors) {
//		if (isInVisionCone(other, BOID_VISION_CONE)) {
//		    neighborsInFov.add(other);
//		}
//	    }
//	    nearestBirdInFov = neighborsInFov.isEmpty() ? null : neighborsInFov.get(0);
		}
	}

	/**
	 * No neighbors :(
	 */
	public void wander() {
		setColor(CYAN);
		updatePerceptions();
		if (!neighbors.isEmpty())
			changeNextBehavior("flock");
		alignHeadingTo(prng().nextInt(360));
		move();
	}

	/**
	 * Let us flock
	 */
	public String flock() {
		setColor(BLUE);
		updatePerceptions();

		if (neighbors.isEmpty()) {
			accelerate();
			move();
			return "flock";
		}
//        final double distanceToNeighbors = averageDistanceToNeighbors(neighbors);
//        if (distanceToNeighbors > maxDistanceFromNeighbors()) {
//            headTowardOthers(neighbors);
//            accelerate();
//            move();
//        }

		if (distance(nearestBird) < minDistanceFromNeighbors()) {
			return "separate";
		}

		if (neighbors.size() > GROUP_SIZE)
			return "cohere";
		accelerate();
		return "align";
	}

	public String separate() {
		this.setColor(RED);
		avoidDirection(towards(nearestBird), prng().nextInt((int) MAX_ALIGN_TURN));
		adaptSpeed(nearestBird.getSpeed());
		move();
		return "flock";
	}

	public String align() {
		this.setColor(BLUE);
		adaptSpeed(nearestBird.getSpeed());
		move();
		return "flock";
	}

	/**
	 * Agent behavior : cohesion The agents try to make a group during their
	 * movement
	 */
	public String cohere() {
		this.setColor(GREEN);
		if (!getPatchOtherTurtles().isEmpty()) {
			setColor(WHITE);
		}

		float globalHeading = (float) computeBoidsMeanDirection(neighbors);

		alignHeadingTo(globalHeading, MAX_ALIGN_TURN);
		adaptSpeed(computeBoidsAverageSpeed(neighbors));
		move();

		int currentBehaviorCount = getCurrentBehaviorCount();
		if (currentBehaviorCount > 9) {
			return "flock";
		}
		return "cohere";
	}

	public void adaptSpeed(double targetSpeed) {
		if (targetSpeed > getSpeed()) {
			accelerate();
		} else {
			decelerate();
		}
	}

	/**
	 * 
	 */
	private void randomizeBehavior() {
		if (getCurrentBehaviorCount() > 50) {
			setCurrentBehaviorCount(0);
			alignHeadingTo(prng().nextInt(360));
			setSpeed(prng().nextInt((int) modelMaxSpeed()));
		}
	}

	public void alignDirection() {
		alignHeadingTo(computeBoidsMeanDirection(neighbors));
	}

	public void alignSpeed() {
		if (computeBoidsAverageSpeed(neighbors) > getSpeed()) {
			accelerate();
		} else {
			decelerate();
		}
	}

	public static void main(String[] args) {
		executeThisTurtle(
				1000
//				,"--headless"
				,"--noLog"
				,"--width","500"
				,"--height","500"
				);

		double cosSum = cos(toRadians(90)) + 2 * cos(toRadians(0));
		System.err.println(cosSum);
		double sinSum = sin(toRadians(90)) + 2 * sin(toRadians(0));
		System.err.println(sinSum);
		System.err.println(toDegrees(atan2(sinSum, cosSum)));
		System.err.println(toDegrees(atan2(sinSum / 3, cosSum / 3)));

		cosSum = cos(toRadians(48)) + 2 * cos(toRadians(350));
		System.err.println(cosSum);
		sinSum = sin(toRadians(48)) + 2 * sin(toRadians(350));
		System.err.println(sinSum);
		System.err.println(toDegrees(atan2(sinSum, cosSum)));
		System.err.println(toDegrees(atan2(sinSum / 3, cosSum / 3)));

		System.err.println(toDegrees(atan2(0, 0)));

	}
}