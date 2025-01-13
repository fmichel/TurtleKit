package turtlekit.cpu_boids;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.TKModel;

public class FlockingModel extends TKModel{

	private static double MAX_TURN_DEFAULT_VALUE = 27;

	private static double MIN_SEPARATION_DEFAULT_VALUE = 8;

	private static double MAX_DISTANCE_DEFAULT_VALUE = 15;

	private static double MAX_NEIGHBORS_DEFAULT_VALUE = 9;

	private static double MAX_SPEED_DEFAULT_VALUE = 17;

	private static double BOID_VISION_CONE_DEFAULT_VALUE = 180;

	private static double BOID_FOV_DEFAULT_VALUE = 7;

	private static double CUDA_FOV_DEFAULT_VALUE = 5;

	private static double MAX_ACCELERATION_DEFAULT_VALUE = 18;

	private static double MAX_ALIGN_TURN_DEFAULT_VALUE = 15;

	private static double GROUP_SIZE_DEFAULT_VALUE = 5;

	/**
	 * The Minimum Speed
	 * 
	 * @see BirdFlockingCPU#adaptSpeed()
	 */
	static float MIN_SPEED = 0.5f;
	/**
	 * The Field of View
	 */


	@SliderProperty(minValue = 1, maxValue = 30, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "FOV")
	public static double BOID_FOV = BOID_FOV_DEFAULT_VALUE;

	/**
	 * The Field of View
	 */
	@SliderProperty(minValue = 1, maxValue = 100, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_ACCELERATION")
	static public double MAX_ACCELERATION = MAX_ACCELERATION_DEFAULT_VALUE;

	/**
	 * The vision cone
	 */
	@SliderProperty(minValue = 1, maxValue = 360, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "BOID_VISION_CONE")
	static public double BOID_VISION_CONE = BOID_VISION_CONE_DEFAULT_VALUE;

	@SliderProperty(minValue = 1, maxValue = 100, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_SPEED")
	static public double MAX_SPEED = MAX_SPEED_DEFAULT_VALUE;

	@SliderProperty(minValue = 1, maxValue = 30, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_NEIGHBORS")
	static public double MAX_NEIGHBORS = MAX_NEIGHBORS_DEFAULT_VALUE;

	/**
	 * have to test that for more coherence: less blue birds
	 */
	@SliderProperty(minValue = 0, maxValue = 100, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_DISTANCE")
	static public double MAX_DISTANCE = MAX_DISTANCE_DEFAULT_VALUE;

	/**
	 * The minimal distance between agents
	 * 
	 */
	@SliderProperty(minValue = 0, maxValue = 100, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MIN_DISTANCE")
	static public double MIN_DISTANCE = MIN_SEPARATION_DEFAULT_VALUE;

	/**
	 * The maximum rotation angle for the separate behavior
	 * 
	 */
	@SliderProperty(minValue = 1, maxValue = 300, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_TURN")
	static public double MAX_TURN = MAX_TURN_DEFAULT_VALUE;

	/**
	 * The maximum rotation angle for the align behavior
	 * 
	 */
	@SliderProperty(minValue = 1, maxValue = 300, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "MAX_ALIGN_TURN")
	static public double MAX_ALIGN_TURN = MAX_ALIGN_TURN_DEFAULT_VALUE;

	@SliderProperty(minValue = 2, maxValue = 15, scrollPrecision = 1)
	@UIProperty(category = "Agent", displayName = "GROUP_SIZE")
	static public double GROUP_SIZE = GROUP_SIZE_DEFAULT_VALUE;

	/**
	 * @return the mIN_SPEED
	 */
	public static float getMIN_SPEED() {
		return MIN_SPEED;
	}

	/**
	 * @param mIN_SPEED the mIN_SPEED to set
	 */
	public static void setMIN_SPEED(float mIN_SPEED) {
		MIN_SPEED = mIN_SPEED;
	}

	/**
	 * @return the bOID_FOV
	 */
	public static double getBOID_FOV() {
		return BOID_FOV;
	}

	/**
	 * @param bOID_FOV the bOID_FOV to set
	 */
	public static void setBOID_FOV(double bOID_FOV) {
		BOID_FOV = bOID_FOV;
	}

	/**
	 * @return the mAX_ACCELERATION
	 */
	public static double getMAX_ACCELERATION() {
		return MAX_ACCELERATION;
	}

	/**
	 * @param mAX_ACCELERATION the mAX_ACCELERATION to set
	 */
	public static void setMAX_ACCELERATION(double mAX_ACCELERATION) {
		MAX_ACCELERATION = mAX_ACCELERATION;
	}

	/**
	 * @return the bOID_VISION_CONE
	 */
	public static double getBOID_VISION_CONE() {
		return BOID_VISION_CONE;
	}

	/**
	 * @param bOID_VISION_CONE the bOID_VISION_CONE to set
	 */
	public static void setBOID_VISION_CONE(double bOID_VISION_CONE) {
		BOID_VISION_CONE = bOID_VISION_CONE;
	}

	/**
	 * @return the mAX_SPEED
	 */
	public static double getMAX_SPEED() {
		return MAX_SPEED;
	}

	/**
	 * @param mAX_SPEED the mAX_SPEED to set
	 */
	public static void setMAX_SPEED(double mAX_SPEED) {
		MAX_SPEED = mAX_SPEED;
	}

	/**
	 * @return the mAX_NEIGHBORS
	 */
	public static double getMAX_NEIGHBORS() {
		return MAX_NEIGHBORS;
	}

	/**
	 * @param mAX_NEIGHBORS the mAX_NEIGHBORS to set
	 */
	public static void setMAX_NEIGHBORS(double mAX_NEIGHBORS) {
		MAX_NEIGHBORS = mAX_NEIGHBORS;
	}

	/**
	 * @return the mAX_DISTANCE
	 */
	public static double getMAX_DISTANCE() {
		return MAX_DISTANCE;
	}

	/**
	 * @param mAX_DISTANCE the mAX_DISTANCE to set
	 */
	public static void setMAX_DISTANCE(double mAX_DISTANCE) {
		MAX_DISTANCE = mAX_DISTANCE;
	}

	/**
	 * @return the mIN_DISTANCE
	 */
	public static double getMIN_DISTANCE() {
		return MIN_DISTANCE;
	}

	/**
	 * @param mIN_DISTANCE the mIN_DISTANCE to set
	 */
	public static void setMIN_DISTANCE(double mIN_DISTANCE) {
		MIN_DISTANCE = mIN_DISTANCE;
	}

	/**
	 * @return the mAX_TURN
	 */
	public static double getMAX_TURN() {
		return MAX_TURN;
	}

	/**
	 * @param mAX_TURN the mAX_TURN to set
	 */
	public static void setMAX_TURN(double mAX_TURN) {
		MAX_TURN = mAX_TURN;
	}

	/**
	 * @return the mAX_ALIGN_TURN
	 */
	public static double getMAX_ALIGN_TURN() {
		return MAX_ALIGN_TURN;
	}

	/**
	 * @param mAX_ALIGN_TURN the mAX_ALIGN_TURN to set
	 */
	public static void setMAX_ALIGN_TURN(double mAX_ALIGN_TURN) {
		MAX_ALIGN_TURN = mAX_ALIGN_TURN;
	}

	/**
	 * @return the gROUP_SIZE
	 */
	public static double getGROUP_SIZE() {
		return GROUP_SIZE;
	}

	/**
	 * @param gROUP_SIZE the gROUP_SIZE to set
	 */
	public static void setGROUP_SIZE(double gROUP_SIZE) {
		GROUP_SIZE = gROUP_SIZE;
	}

}
