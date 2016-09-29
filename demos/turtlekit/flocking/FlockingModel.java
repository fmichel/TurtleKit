package turtlekit.flocking;

public class FlockingModel {

	/**
	 * The Field of View
	 */
	static int vision = 10;
	/**
	 * The Maximum Speed
	 * @see BirdFlockingCPU#adaptSpeed()
	 */
	static float maxSpeed = 2.0f;
	/**
	 * The Minimum Speed
	 * @see BirdFlockingCPU#adaptSpeed()
	 */
	static float minSpeed = 0.5f;
	/**
	 * The minimal distance between agents
	 * @see BirdFlockingCPU#flock()
	 */
	static int minSeparation = 1;
	/**
	 * The maximum rotation angle for the align behavior
	 * @see BirdFlockingCPU#align()
	 */
	static int maxAlignTurn = 20;//15
	/**
	 * The maximum rotation angle for the cohesion behavior
	 * @see BirdFlockingCPU#cohere()
	 */
	static int maxCohereTurn = 20;//15
	/**
	 * The maximum rotation angle for the separate behavior
	 * @see BirdFlockingCPU#separate()
	 */
	static int maxSeparateTurn = 60;//45

}
