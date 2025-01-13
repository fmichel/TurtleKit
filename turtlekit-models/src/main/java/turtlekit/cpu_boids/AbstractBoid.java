package turtlekit.cpu_boids;

import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sin;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;
import static turtlekit.cpu_boids.FlockingModel.MAX_ACCELERATION;
import static turtlekit.cpu_boids.FlockingModel.MAX_DISTANCE;
import static turtlekit.cpu_boids.FlockingModel.MAX_SPEED;
import static turtlekit.cpu_boids.FlockingModel.MAX_TURN;
import static turtlekit.cpu_boids.FlockingModel.MIN_DISTANCE;
import static turtlekit.cpu_boids.FlockingModel.MIN_SPEED;

import java.util.List;

import madkit.simulation.EngineAgents;
import turtlekit.kernel.DefaultTurtle;

@EngineAgents(model=FlockingModel.class)
public class AbstractBoid extends DefaultTurtle {

	static private double speed = 1.0f;

	public AbstractBoid(String initMethod) {
		changeNextBehavior(initMethod);
	}

	public AbstractBoid() {
		super();
	}

	public double computeBoidsMeanDirection(List<AbstractBoid> boids) {
		double cosSum = boids.parallelStream().mapToDouble(b -> cos(b.getHeadingInRadians())).sum();
		double sinSum = boids.parallelStream().mapToDouble(b -> sin(b.getHeadingInRadians())).sum();
		return toDegrees(atan2(sinSum, cosSum));
	}

	public double computeBoidsAverageSpeed(List<AbstractBoid> boids) {
		return boids.parallelStream().mapToDouble(b -> b.getSpeed()).average().orElse(speed);
	}

	public void headTowardOthers(List<AbstractBoid> boids) {
		alignHeadingTo(computeHeadingTowardOthers(boids));
	}

	public void headLikeOthers(List<AbstractBoid> boids) {
		alignHeadingTo(computeBoidsMeanDirection(boids));
	}

	public double computeHeadingTowardOthers(List<AbstractBoid> boids) {
		double cosSum = boids.parallelStream().mapToDouble(b -> cos(toRadians(towards(b)))).sum();
		double sinSum = boids.parallelStream().mapToDouble(b -> sin(toRadians(towards(b)))).sum();
		return toDegrees(atan2(sinSum, cosSum));
	}

	public double computeHeadingMean(List<AbstractBoid> boids) {
		double cosSum = boids.parallelStream().mapToDouble(b -> b.getCosinus()).sum();
		double sinSum = boids.parallelStream().mapToDouble(b -> b.getSinus()).sum();
		return toDegrees(atan2(sinSum, cosSum));
	}

	public double averageDistanceToNeighbors(List<AbstractBoid> boids) {
		return boids.parallelStream().mapToDouble(b -> distance(b)).average().orElse(20);
	}

	public void alignHeadingTo(double target) {
		double angleToOther = signedTwoAnglesDifference(getHeading(), target);
		final int nextInt = prng().nextInt((int) MAX_TURN);
		// final int nextInt = MAX_TURN;
		if (angleToOther <= 0) {
			turnRight(nextInt);
		} else {
			turnLeft(nextInt);
		}
	}

	public void alignHeadingTo(double target, double turn) {
		double angleToOther = signedTwoAnglesDifference(getHeading(), target);
		if (angleToOther <= 0) {
			turnRight(turn);
		} else {
			turnLeft(turn);
		}
	}

	public void decelerate() {
		setSpeed(max(speed - getMaxAcceleration(), MIN_SPEED));
	}

	private float getMaxAcceleration() {
		return ((float) MAX_ACCELERATION) / 100;
	}

	public void accelerate() {
		setSpeed(min(speed + getMaxAcceleration(), modelMaxSpeed()));
	}

	public void avoidDirection(double target) {
		double angleToOther = signedTwoAnglesDifference(getHeading(), target);
		final int nextInt = prng().nextInt((int) MAX_TURN);
		if (angleToOther >= 0) {
			turnRight(nextInt);
		} else {
			turnLeft(nextInt);
		}
	}

	public void avoidDirection(double target, double turn) {
		double angleToOther = signedTwoAnglesDifference(getHeading(), target);
		if (angleToOther >= 0) {
			turnRight(turn);
		} else {
			turnLeft(turn);
		}
	}

	/**
	 * Return the Speed
	 * 
	 * @return speed
	 */
	public double getSpeed() {
		return speed;
	}

	/**
	 * Set the Speed of the agent
	 * 
	 * @param d The new Speed
	 * 
	 * @see BirdFlockingCPU#align()
	 * @see BirdFlockingCPU#cohesion()
	 */
	protected void setSpeed(double d) {
		this.speed = max(min(modelMaxSpeed(), d), MIN_SPEED);
	}

	protected float modelMaxSpeed() {
		return ((float) MAX_SPEED) / 10;
	}

	/**
	 * @return
	 */
	protected float maxDistanceFromNeighbors() {
		return ((float) MAX_DISTANCE) / 10;
	}

	/**
	 * @return
	 */
	protected float minDistanceFromNeighbors() {
		return ((float) MIN_DISTANCE) / 10;
	}

	/**
	 * @return
	 */
	protected float maxAcceleration() {
		return (float) (MAX_ACCELERATION / 100);
	}

	/**
	 * Activate the agent
	 * 
	 * @see DefaultTurtle#onActivation()
	 */
	protected void onActivation() {
		super.onActivation();
		home();
		// setHeading(0);
		randomHeading();
		changeNextBehavior("wander");
	}

	/**
	 * Agent behavior : move
	 */
	public void move() {
		fd(speed);
	}

	/**
	 * Random heading for the agent
	 */
	public void headingChange() {
		randomHeading();
	}

	/**
	 * Random heading for the agent
	 */
	protected void changeHeading(int turn) {
		if (turn > 0) {
			final int nextInt = prng().nextInt(turn);
			if (prng().nextBoolean()) {
				turnRight(nextInt);
			} else {
				turnLeft(nextInt);
			}
		}
	}

	public boolean isInVisionCone(DefaultTurtle t, double visionConeAngle) {
		return differenceTwoAngles(getHeading(), towards(t)) <= visionConeAngle / 2;
	}

	/**
	 * Test pour connaitre si je suis dans l'intervalle
	 */
	public boolean amIInTheInterval(double otherHeading, int interval) {
		double diffTwoAngle = differenceTwoAngles(getHeading(), otherHeading);
		return (diffTwoAngle <= interval);
	}

	/**
	 * Connaitre la différence entre deux angles (sans prise en compte des signes)
	 */
	public double differenceTwoAngles(double targetA, double targetB) {
		double d = Math.abs(targetA - targetB) % 360;
		return d > 180 ? 360 - d : d;
	}

	/**
	 * Adapter la direction en fonction de la différence d'heading entre l'agent et la
	 * cible
	 */
	public void changeHeadingReduceInterval(double otherHeading, int turn) {
		// //GOOD for GPU
		double differenceAngle = differenceTwoAngles(getHeading(), otherHeading);
		int turnAngle = prng().nextInt(turn);
		double temp = differenceTwoAngles((getHeading() + turnAngle), otherHeading);

		if (temp > differenceAngle) {
			setHeading(getHeading() - turnAngle);
		} else {
			setHeading(getHeading() + turnAngle);
		}
	}

	/**
	 * Connaitre la différence entre deux angles (avec prise en compte des signes)
	 */
	public double signedTwoAnglesDifference(double from, double to) {
		double a = (to - from) % 360;
		if (a < -180)
			a += 360;
		if (a > 180)
			a -= 360;
		return a;
	}

	public static void main(String[] args) {
		AbstractBoid a = new AbstractBoid();
		a.setHeading(0);
		System.err.println(a.differenceTwoAngles(a.getHeading(), 20) <= 30 / 2);
		System.err.println(a.differenceTwoAngles(350, 0));
		System.err.println(a.signedTwoAnglesDifference(0, 350));
		System.err.println(a.signedTwoAnglesDifference(350, 0));
		a.alignHeadingTo(50);
		System.err.println(a);
		a.alignHeadingTo(50);
		System.err.println(a);
		a.alignHeadingTo(50);
		System.err.println(a);
		a.alignHeadingTo(50);
		System.err.println(a);
	}

}