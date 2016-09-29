package turtlekit.flocking;

import java.awt.Color;
import java.util.List;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;

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
 * 
 * @version 0.1
 * 
 * @see turtlekit.kernel.Turtle
 * 
 */

public class InfluReacBird extends Turtle {

    /**
     * The Field of View
     */
    private static int vision = 20;
 
    /**
     * The Speed
     * @see BirdFlockingCPU#activate()
     * @see BirdFlockingCPU#move()
     */
    private float speed = 1.0f;
 
    /**
     * The Maximum Speed
     * @see BirdFlockingCPU#adaptSpeed()
     */
    private static float maxSpeed = 2.0f;
 
    /**
     * The Minimum Speed
     * @see BirdFlockingCPU#adaptSpeed()
     */
    private static float minSpeed = 0.5f;
 
    /**
     * An adaptative speed value
     * @see BirdFlockingCPU#align()
     * @see BirdFlockingCPU#cohere()
     */
    private float adaptiveSpeed = 0.1f;
 
    /**
     * The minimal distance between agents
     * @see BirdFlockingCPU#flock()
     */
    private static int minSeparation = 1;
 
    /**
     * The maximum rotation angle for the align behavior
     * @see BirdFlockingCPU#align()
     */
    private static int maxAlignTurn = 20;//15
 
    /**
     * The maximum rotation angle for the cohesion behavior
     * @see BirdFlockingCPU#cohere()
     */
    private static int maxCohereTurn = 20;//15
 
    /**
     * The maximum rotation angle for the separate behavior
     * @see BirdFlockingCPU#separate()
     */
    private static int maxSeparateTurn = 60;//45
 
    /**
     * The nearest neighbor
     * @see BirdFlockingCPU#flock()
     * @see BirdFlockingCPU#separate()
     * @see BirdFlockingCPU#align()
     */
    private InfluReacBird nearestBird;
 
    /**
     * The neighbor list
     * @see BirdFlockingCPU#flock()
     * @see BirdFlockingCPU#cohere()
     */
    private List<InfluReacBird> birdList;
     
    /**
     * Return the Field of View
     * @return vision
     */
    public static int getVision() {
        return vision;
    }
     
    /**
     * Return the Speed
     * @return speed
     */
    public float getSpeed() {
        return speed;
    }
 
    /**
     * Set the Speed of the agent
     * @param speed
     *            The new Speed
     * 
     * @see BirdFlockingCPU#align()
     * @see BirdFlockingCPU#cohere()
     */
    public void setSpeed(float speed) {
        this.speed = speed;
    }
 
    /**
     * Activate the agent
     * @see Turtle#activate()
     */
    protected void activate() {
        super.activate();
        setColor(Color.YELLOW);
        moveTo(generator.nextInt(getWorldHeight()),
                generator.nextInt(getWorldWidth()));
        home();
        randomHeading();
        fd(0.1);
        speed = generator.nextFloat() + 0.5f;
        adaptiveSpeed = speed / 5.0f;
        setNextAction("flock");
    }
 
    /**
     * Agent behavior : move
     */
    public void move() {
        fd(speed);
    }
 
    /**
     * Agent behavior : global behavior
     * According to the distance between the agents, the different Reynolds's rules will be activated
     */
    public String flock() {
    	
    	birdList = getNearestTurtles(vision, 6, InfluReacBird.class);
    	float clusterAverageSpeed = 0;
    	for (InfluReacBird influReacBird : birdList) {
			clusterAverageSpeed += influReacBird.getSpeed();
		}
    	
    	
    	//adapt speed
    	clusterAverageSpeed /= birdList.size();
    	adaptSpeed(clusterAverageSpeed);
    	
//    	float clusterAverageHeading = 0;
//    	for (InfluReacBird influReacBird : birdList) {
//			clusterAverageHeading += influReacBird.getHeading();
//		}
    	
    	nearestBird = null;
    	if(birdList.isEmpty()){
    		move();
    		return "flock";
    	}
		nearestBird = birdList.get(0);
        
		//separate
        if (this.distance(nearestBird) < minSeparation) {
            this.setColor(Color.RED);

            double headingInterseptionNearestBird = this.towards(nearestBird);
     
            if (amIInTheInterval(this.getHeading(), headingInterseptionNearestBird, maxSeparateTurn*2)) {
    			this.setHeading(changeHeading(this.getHeading(), maxSeparateTurn));
    		}
     
            adaptSpeed(nearestBird.getSpeed() + generator.nextFloat());
        }
        else {
            if (birdList.size() > 6) {
                return "cohere";
            } else {
                return "align";
            }
        }
        move();
        return "flock";
    }
 
    /**
     * Agent behavior : separate
     * If agents are too close, they separate
     */
    public String separate() {
        this.setColor(Color.RED);

        double headingInterseptionNearestBird = this.towards(nearestBird);
 
        if (amIInTheInterval(this.getHeading(), headingInterseptionNearestBird, maxSeparateTurn*2)) {
			this.setHeading(changeHeading(this.getHeading(), maxSeparateTurn));
		}
 
        adaptSpeed(nearestBird.getSpeed() + generator.nextFloat());
        move();
        fillHeadingEnvironment(this.getHeading());// Avant double myHeading = this.getHeading()
        return "flock";
    }
 
    /**
     * Agent behavior : align
     * The agent searches for align its movement to its neighbor
     */
    public String align() {
        this.setColor(Color.BLUE);

        double otherHeading = nearestBird.getHeading();
        
		if (!amIInTheInterval(this.getHeading(), otherHeading, 1)) {
			this.setHeading(changeHeadingReduceInterval(this.getHeading(),otherHeading,maxAlignTurn));
		}
		
        adaptSpeed(nearestBird.getSpeed());
        move();
        fillHeadingEnvironment(this.getHeading());// Avant double myHeading = this.getHeading()
        return "flock";
    }
 
    /**
     * Agent behavior : cohesion
     * The agents try to make a group during their movement
     */
    public String cohere() {
        
        float globalHeading = 0;
                 
        int size = getOtherTurtles(0, true).size();
		if(size > 0){
			this.setColor(Color.WHITE);
        }
        else{
        	this.setColor(Color.GREEN);
        }
        
        if(!Environment.isCUDA()){
        	float globalSpeed = 0;
        	
             for(InfluReacBird bird : birdList){
             globalHeading += bird.getHeading();
             globalSpeed += bird.getSpeed();
             }
             
             globalHeading = globalHeading / birdList.size();
             globalSpeed = globalSpeed / birdList.size();
             
             adaptSpeed(globalSpeed);
        }
        else{
            globalHeading = ((Environment) getEnvironment()).getCudaHeadingValue(this.xcor(), this.ycor());
        }
 
//        if (myHeading > globalHeading) {
//            myHeading = myHeading - generator.nextInt(maxCohereTurn);
//        } else if (myHeading < globalHeading) {
//            myHeading = myHeading + generator.nextInt(maxCohereTurn);
//        } else {
//            myHeading = globalHeading;
//        }
// 
//        this.setHeading(myHeading);
        
		if (!amIInTheInterval(this.getHeading(), globalHeading, 1)) {
			this.setHeading(changeHeadingReduceInterval(this.getHeading(),globalHeading,maxCohereTurn));
		}
         
        move();
        fillHeadingEnvironment(this.getHeading());
 
        int currentBehaviorCount = getCurrentBehaviorCount();
        if (currentBehaviorCount > 10) {
            return "flock";
        }
        
        return "cohere";
    }
 
    /**
     * Random heading for the agent
     */
    public void headingChange() {
        randomHeading();
    }
     
    /**
     * Fill its heading value in the environment
     */
    public void fillHeadingEnvironment(double heading){
//        ((Environment) getEnvironment()).setCudaHeadingValue(this.xcor(), this.ycor(), heading);
    }
 
	/**
	 * Random heading for the agent
	 */
	public double changeHeading(double heading, int turn) {
		if (generator.nextBoolean()) {
			return heading = heading + generator.nextInt(turn);
		} else {
			return heading = heading - generator.nextInt(turn);
		}
	}
	
	/**
	 * Test pour connaitre si je suis dans l'intervalle
	 */
	public boolean amIInTheInterval(double myHeading, double otherHeading, int interval){
		double diffTwoAngle = differenceTwoAngle(myHeading, otherHeading);
		return (diffTwoAngle <= interval);
	}
			
	/**
	 * Connaitre la différence entre deux angles (sans prise en compte des signes)
	 */
	public double differenceTwoAngle(double targetA, double targetB){
		double d = Math.abs(targetA - targetB) % 360;
		return d > 180 ? 360 - d : d;
	}
	
	/**
	 * Adapter la direction en fonction de la différence d'angle entre l'agent et la cible
	 */
	public double changeHeadingReduceInterval(double myHeading, double otherHeading, int turn){
//		//GOOD for GPU
		double differenceAngle = differenceTwoAngle(myHeading, otherHeading);
		int turnAngle = generator.nextInt(turn);
		double temp = differenceTwoAngle((myHeading + turnAngle), otherHeading);
		
		if(temp > differenceAngle){
			return myHeading = myHeading - turnAngle;
		}
		else{
			return myHeading = myHeading + turnAngle;
		}
		
//		Good for CPU
//		double differenceAngle = differenceTwoAngle(myHeading, otherHeading);
//		int turnAngle = generator.nextInt(turn);
//		while (turnAngle > differenceAngle){
//			turnAngle = generator.nextInt(turn);
//		}
//		double tempP = differenceTwoAngle((myHeading + turnAngle), otherHeading);
//		double tempM = differenceTwoAngle((myHeading - turnAngle), otherHeading);
//
//		if(differenceAngle == 0){
//			return myHeading;
//		}
//		else {
//			if(tempP > tempM){
//				return myHeading = myHeading - turnAngle;
//			}
//			else{
//				return myHeading = myHeading + turnAngle;
//			}
//		}
	}
	
	/**
	 * Connaitre la différence entre deux angles (avec prise en compte des signes)
	 */
	public double differenceTwoAngleV2(double myHeading, double otherHeading){
		double a = (otherHeading - myHeading) % 360;
		if(a < -180) a += 360;
		if(a >  180) a -= 360;
		return a;
	}
	
	/**
	 * Adapter la direction en fonction de la différence d'angle entre l'agent et la cible (en fonction du signe de la diff d'angle)
	 */
	public double changeHeadingReduceIntervalV2(double myHeading, double otherHeading, int turn){
		double differenceAngle = differenceTwoAngleV2(myHeading, otherHeading);
		int turnAngle = generator.nextInt(turn);
		
		if(differenceAngle > 0){
			return myHeading = myHeading - turnAngle;
		}
		else{
			return myHeading = myHeading + turnAngle;
		}
	}
    
    /**
     * Adapt the speed
     */
    public void adaptSpeed(float comparativeSpeed) {
        float currentSpeed = this.getSpeed();
        if (currentSpeed > maxSpeed) {
            currentSpeed = currentSpeed - adaptiveSpeed;
        } else if (currentSpeed < minSpeed) {
            currentSpeed = currentSpeed + adaptiveSpeed;
        } else {
            if (currentSpeed > comparativeSpeed) {
                currentSpeed = currentSpeed - adaptiveSpeed;
            } else {
                currentSpeed = currentSpeed + adaptiveSpeed;
            }
        }
        this.setSpeed(currentSpeed);
    }
    
    public static void main(String[] args) {
		executeThisTurtle(100,
				Option.envDimension.toString(),"500,500");
	}
}