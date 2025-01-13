package turtlekit.cpu_boids;

import static javafx.scene.paint.Color.GREEN;
import static javafx.scene.paint.Color.RED;
import static javafx.scene.paint.Color.WHITE;
import static turtlekit.cpu_boids.FlockingModel.BOID_FOV;
import static turtlekit.cpu_boids.FlockingModel.BOID_VISION_CONE;
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

public class IGHDemoBoid extends AbstractBoid {

	protected AbstractBoid nearestBirdInFov;
	protected AbstractBoid nearestBird;
	protected List<AbstractBoid> neighbors;
	protected List<AbstractBoid> neighborsInFov;

	public IGHDemoBoid() {
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
			neighbors = neighbors.subList(0, nb > size ? size : nb);
			nearestBird = neighbors.get(0);
			for (AbstractBoid other : neighbors) {
				if (isInVisionCone(other, BOID_VISION_CONE)) {
					neighborsInFov.add(other);
				}
			}
			nearestBirdInFov = neighborsInFov.isEmpty() ? null : neighborsInFov.get(0);
		}
	}

	/**
	 * No neighbors :(
	 */
	public void wander() {
		setColor(RED);
		updatePerceptions();
		if (!neighbors.isEmpty())
			changeNextBehavior("flock");
		alignHeadingTo(prng().nextInt(360));
		move();
	}

	/**
	 * Let us flock
	 */
	public void flock() {
		setColor(GREEN);
		if (!getPatchOtherTurtles().isEmpty()) {
			setColor(WHITE);
		}
		updatePerceptions();

		if (neighbors.isEmpty()) {
			changeNextBehavior("wander");
		}

		setSpeed(computeBoidsAverageSpeed(neighbors));

		if (nearestBirdInFov != null)
			accelerate();

//        final double distanceToNearest = distance(nearestBird);

		final double distanceToNeighbors = averageDistanceToNeighbors(neighbors);

		if (distanceToNeighbors < minDistanceFromNeighbors()) {
			avoidDirection(computeHeadingTowardOthers(neighborsInFov));
			move();
		}
		if (distanceToNeighbors > maxDistanceFromNeighbors()) {
//			headTowardOthers(neighbors);
			headLikeOthers(neighbors);
			accelerate();
			move();
		}

//        randomizeBehavior();
//        headTowardOthers(neighbors);
		alignSpeed();
		alignDirection();
		move();
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
				,"--width","300"
				,"--height","300"
				);
		// ,Option.startSimu.toString()
//				startSimu.toString());
//
	}
}