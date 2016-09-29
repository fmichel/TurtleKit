package turtlekit.flocking;

import java.awt.Color;
import java.util.List;

import turtlekit.kernel.Turtle;

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

public class BirdFlockingUnify extends Turtle {

    /**
     * The Speed
     * @see BirdFlockingCPU#activate()
     * @see BirdFlockingCPU#move()
     */
    private float speed = 1.0f;
 
    /**
     * An adaptative speed value
     * @see BirdFlockingCPU#align()
     * @see BirdFlockingCPU#cohere()
     */
    private float adapatativeSpeed = 0.1f;
 
    /**
     * The nearest neighbor
     * @see BirdFlockingCPU#flock()
     * @see BirdFlockingCPU#separate()
     * @see BirdFlockingCPU#align()
     */
    private BirdFlockingUnify nearestBird;
 
    /**
     * The neighbor list
     * @see BirdFlockingCPU#flock()
     * @see BirdFlockingCPU#cohere()
     */
    private List<BirdFlockingUnify> birdList;
     
    /**
     * Return the Field of View
     * @return vision
     */
    public static int getVision() {
        return FlockingModel.vision;
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
        speed = generator.nextFloat() + 0.5f;
        adapatativeSpeed = speed / 5.0f;
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
 
        birdList = getPatch().getTurtles(FlockingModel.vision, false, BirdFlockingUnify.class);
        nearestBird = null;
         
        if (! birdList.isEmpty()) {
            nearestBird = birdList.get(0);
        }
        if (nearestBird == null || birdList.isEmpty()) {
            move();
            fillHeadingEnvironment(this.getHeading());
            return "flock";
        }
        if (this.distance(nearestBird) < FlockingModel.minSeparation) {
            return "separate";
        }
        else {
            if (birdList.size() > 5) {
                return "cohere";
            } else {
                return "align";
            }
        }
    }
 
    /**
     * Agent behavior : separate
     * If agents are too close, they separate
     */
    public String separate() {
        this.setColor(Color.RED);

        double headingInterseptionNearestBird = this.towards(nearestBird);
 
        if (amIInTheInterval(this.getHeading(), headingInterseptionNearestBird, FlockingModel.maxSeparateTurn*2)) {
			this.setHeading(changeHeading(this.getHeading(), FlockingModel.maxSeparateTurn));
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
			this.setHeading(changeHeadingReduceInterval(this.getHeading(),otherHeading,FlockingModel.maxAlignTurn));
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
        	
             for(BirdFlockingUnify bird : birdList){
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
			this.setHeading(changeHeadingReduceInterval(this.getHeading(),globalHeading,FlockingModel.maxCohereTurn));
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
        ((Environment) getEnvironment()).setCudaHeadingValue(this.xcor(), this.ycor(), heading);
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
        if (currentSpeed > FlockingModel.maxSpeed) {
            currentSpeed = currentSpeed - adapatativeSpeed;
        } else if (currentSpeed < FlockingModel.minSpeed) {
            currentSpeed = currentSpeed + adapatativeSpeed;
        } else {
            if (currentSpeed > comparativeSpeed) {
                currentSpeed = currentSpeed - adapatativeSpeed;
            } else {
                currentSpeed = currentSpeed + adapatativeSpeed;
            }
        }
        this.setSpeed(currentSpeed);
    }
}