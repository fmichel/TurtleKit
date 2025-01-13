package turtlekit.cpu_boids;

import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;
import static turtlekit.cpu_boids.FlockingModel.BOID_FOV;
import static turtlekit.cpu_boids.FlockingModel.BOID_VISION_CONE;
import static turtlekit.cpu_boids.FlockingModel.GROUP_SIZE;
import static turtlekit.cpu_boids.FlockingModel.MAX_ALIGN_TURN;
import static turtlekit.cpu_boids.FlockingModel.MAX_NEIGHBORS;

import java.util.ArrayList;
import java.util.List;

import javafx.scene.paint.Color;

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

public class FollowingBoids extends AbstractBoid {

	protected AbstractBoid nearestBirdInFov;
	protected AbstractBoid nearestBird;
	protected List<AbstractBoid> neighbors;
	protected List<AbstractBoid> neighborsInFov;

	public FollowingBoids() {
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
		setColor(Color.CYAN);
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
		setColor(Color.BLUE);
		updatePerceptions();
		if (neighbors.isEmpty()) {
			accelerate();
			move();
			changeNextBehavior("flock");
		}
//        final double distanceToNeighbors = averageDistanceToNeighbors(neighbors);
//        if (distanceToNeighbors > maxDistanceFromNeighbors()) {
//            headTowardOthers(neighbors);
//            accelerate();
//            move();
//        }

//        if(distance(nearestBird) < minDistanceFromNeighbors()) {
//            return "separate";
//        }

		if (nearestBirdInFov != null && distance(nearestBirdInFov) < minDistanceFromNeighbors()) {
			changeNextBehavior("separate");
		} else {
			alignHeadingTo(computeBoidsMeanDirection(neighborsInFov), MAX_ALIGN_TURN);
			accelerate();
		}

		if (neighbors.size() > GROUP_SIZE)
			changeNextBehavior("cohere");
		changeNextBehavior("align");
	}

	public String separate() {
		this.setColor(Color.RED);
		avoidDirection(towards(nearestBirdInFov), prng().nextInt((int) MAX_ALIGN_TURN));
//	adaptSpeed(nearestBird.getSpeed());
		decelerate();
		move();
		return "flock";
	}

	public String align() {
		updatePerceptions();
		this.setColor(Color.BLUE);
		if (nearestBird != null) {
			adaptSpeed(nearestBird.getSpeed());
		}
		move();
		return "flock";
	}

	/**
	 * Agent behavior : cohesion The agents try to make a group during their
	 * movement
	 */
	public String cohere() {
		this.setColor(Color.GREEN);
		if (!getPatchOtherTurtles().isEmpty()) {
			setColor(Color.WHITE);
		}

		float globalHeading = (float) computeBoidsMeanDirection(neighbors);

		alignHeadingTo(globalHeading, MAX_ALIGN_TURN);
		adaptSpeed(computeBoidsAverageSpeed(neighbors));
		move();

		int currentBehaviorCount = getCurrentBehaviorCount();
		if (currentBehaviorCount > 10) {
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
		executeThisTurtle(500
				,"--model",FlockingModel.class.getName()
				);

		double cosSum = cos(toRadians(90)) + 2 * cos(toRadians(0));
		double sinSum = sin(toRadians(90)) + 2 * sin(toRadians(0));
		System.err.println(toDegrees(atan2(sinSum, cosSum)));
	}
}