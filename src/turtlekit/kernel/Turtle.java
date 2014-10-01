/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package turtlekit.kernel;

import static java.lang.Math.abs;
import static java.lang.Math.atan;
import static java.lang.Math.cos;
import static java.lang.Math.hypot;
import static java.lang.Math.sin;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;

import java.awt.Color;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import madkit.kernel.AbstractAgent;
import madkit.kernel.Activator;
import madkit.kernel.Madkit;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.pheromone.Pheromone;

public class Turtle extends AbstractAgent {
	
	/**
	 * Used in {@link #toString()}
	 */
	private static DecimalFormat df = new DecimalFormat("0.#");

	/**
	 * turtle's heading
	 */
	double angle; 
	
	/**
	 * package visibility : used by TK environment... Ugly in OOP but faster //TODO is this true ?
	 * 
	 * Max value is used to detect that default value is used @
	 */
	double x = Double.MAX_VALUE;
	/**
	 * package visibility : used by TK environment... Ugly in OOP but faster //TODO is this true ?
	 */
	double y = Double.MAX_VALUE, 
	
			
			angleCos = 1, 
	angleSin = 0;//TODO float !!
	private TKEnvironment environment;
	private Patch position;
	protected Color color = Color.RED;
	private int currentBehaviorCount = 0;
	private boolean changePatch = false;
	private Method nextAction;
	private int id;
	protected static Random generator = new Random();//TODO move to model
	private static String community;

	@Override
	protected void activate() {
//		setLogLevel(Level.ALL);
		if (community == null) {
			community = getMadkitProperty(TurtleKit.Option.community);
		}
		requestRole(community, TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE, null);
		setup();//TODO
	}
	
	/**one way to identify a kind of turtle: give them a Role in the simulation.*/
	public  void playRole(String role){
		bucketModeRequestRole(community,TKOrganization.TURTLES_GROUP,role,null);
	}

	/**the turtle will no longer play the specified role*/
	public  void giveUpRole(String role){
		leaveRole(community,TKOrganization.TURTLES_GROUP,role);
	}
	
	public int getMinDirection(Pheromone p){
		return p.getMinDirection(xcor(), ycor());
	}
	
	public int getMaxDirection(Pheromone p){
		return p.getMaxDirection(xcor(), ycor());
	}
	
	public void towardsMinGradientField(Pheromone p) {
		setHeading(getMinDirection(p));
	}

	public void towardsMaxGradientField(Pheromone p) {
		setHeading(getMaxDirection(p));
	}

	/**
	 * Tells if the turtle is currently playing this role
	 * 
	 * @param role
	 * @return <code>true</code> if the turtle is playing this role
	 */
	public  boolean isPlayingRole(String role){
		return hasRole(community,TKOrganization.TURTLES_GROUP,role);
	}

	/**
	 * @return the absolute index of the turtle's location
	 * in a 1D data grid
	 * representing a 2D grid of width * height size.
	 * Its purpose is to be used on a Pheromone to optimize several calls 
	 * on its data structure.
	 */
	public int get1DIndex(){//useful because there is no check 
		 return position.x + position.y * environment.getWidth();
	 }

	/**
	 * @param x absolute 
	 * @param y absolute 
	 * @return the absolute index in a 1D data grid representing a 2D grid of
	 *         width,height size. This should be used with {@link #xcor()} and
	 *         {@link #ycor()} to ensure that this x,y point in inside the
	 *         environment's boundaries. Its purpose is to be used on a
	 *         Pheromone
	 */
	public int get1DIndex(int x, int y){
		 return environment.get1DIndex(x,y);
	 }

	
//	/** return the number of turtles in the patch situated at (a,b) units away */
//	public final int countTurtlesAt(final int a, final int b) {
//		return env.turtlesCountAt(normeValue(a + xcor(), env.getWidth()),
//				normeValue(b + ycor(), env.getHeight()));
//	}
	
//	/**
//	 * @return <code>true</code> if the next patch (dx dy) is a different patch
//	 *         and is occupied by a turtle
//	 */
//	public boolean isNextPatchOccupied() {
//		return false;
//	}

	public int getWorldWidth() {return environment.getWidth();}
	public int getWorldHeight(){return environment.getHeight();}

	/** Teleports the turtle to the center patch */
	public final void home() {
		if (position != null) {
			moveTo(environment.getWidth() / 2,environment.getHeight() / 2);
		}
	}

	public void setPatchColor(Color c) {
		position.setColor(c);
	}

	public Color getPatchColor() {
		return position.getColor();
	}
	
	/**
	 * @return the next patch if fd(1) leads to a different patch, <code>null</code> otherwise
	 */
	public Patch getNextPatch(){
		final int dx = dx();
		final int dy = dy();
		if (dx != 0 || dy != 0)
			return environment.getPatch(dx + xcor(), dy + ycor());
		return null;
	}
	
	public float getFieldValue(Pheromone p){
		return p.get(xcor(), ycor());
	}

	/** Teleports the turtle to a random location */
	public void randomLocation() {
		moveTo(generator.nextDouble()*getWorldWidth(), generator.nextDouble()*getWorldHeight());
	}

	/** Get the patch situated at (a,b) units away */
	public Patch getPatchAt(final int a, final int b) {
		return environment.getPatch(a + xcor(), b + ycor());
	}
	
	/**
	 * Gets the corresponding pheromone or create a new one using defaults
	 * parameters : 50% for both the evaporation rate and 
	 * the diffusion rate.
	 * @param name the pheromone's name
	 * @return the pheromone
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public Pheromone getPheromone(String name){
//		System.exit(0);
		return environment.getPheromone(name, 50, 50);
	}

	/**
	 * Gets the direction where there is a minimum of this pheromone
	 * @param name the pheromone's name
	 * @return the direction to take
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public double getPheroMinDirection(Pheromone phero){
		return phero.getMinDirection(xcor(),ycor());
	}

	/**
	 * Gets the direction where there is a maximum of this pheromone
	 * @param name the pheromone's name
	 * @return the direction to take
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public double getPheroMaxDirection(Pheromone phero){
		return phero.getMaxDirection(xcor(),ycor());
	}

	//For the Python Mode
	public Turtle(String initMethod)
	{
		setNextAction(initMethod);
	}
	
	public void setNextAction(String methodName){
		try {
			setNextMethod(Activator.findMethodOn(getClass(),methodName));
		} catch (NoSuchMethodException | SecurityException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * @deprecated replaced by {@link AbstractAgent#hashCode()}
	 * return the turtle ID
	 */
	public  int mySelf(){return hashCode();}

	public Turtle() {
		this("defaultBehavior");
	}
	
	public String defaultBehavior() {
		wiggle();
		return "defaultBehavior";
	}

	public void incrementBehaviorCount() {
		currentBehaviorCount++;
	}
	
	public  void setX(final double a){
		moveTo(a, y);
	}
	
	public  void setY(final double b){
		moveTo(x,b);
	}
	
	public void setXY(final double a,final double b) {
		moveTo(a,b);
	}
	
	/**
	 * @deprecated use {@link #launchAgent(AbstractAgent)} instead
	 * create a turtle at the creator position (xcor,ycor)
    returns the ID of the new turtle*/
	public  int createTurtle(Turtle t){
		launchAgent(t);
		return t.getID();
		}


	
	/**
	 * the id of this turtle
	 * @return the id of this turtle
	 */
	public int getID() {
		return id;
	}

	/**return the x-increment if the turtle were to take one
    step forward in its current heading.*/
	public  int dx()
	{
		return ((int) (x + angleCos)) - position.x;
//		return (int) (round(x+angleCos)-round(x));
	}

	/**return the int y-increment if the turtle were to take one
    step forward in its current heading.*/
	public  int dy()
	{
		return ((int) (y + angleSin)) - position.y;
//		return (int) (round(y+angleSin)-round(y));
	}

	public  void setColor(Color c){
		color=c;
		}

	final void setNextMethod(Method nextMethod)
	{
			nextAction= nextMethod;
	}

//	 public int getPositionCode(){
//		 return position.i + position.j * env.getWidth();
//	 }
//
//	 public int getPositionCode(int i, int j){
//		 int width = env.getWidth();
//		return normeValue(i, width) + normeValue(j, env.getHeight()) * width;
//	 }

	public Color getColor() {
		return color;
	}

	public float smell(String pheromone) {
		return environment.smell(pheromone, position.x, position.y);//TODO test perf vs xcor()
	}

	public String toString(){
		try {
			return getClass().getSimpleName()+"-"+hashCode()+"@("+xcor()+","+ycor()+") H="+df.format(angle);
		} catch (NullPointerException e) {
			return "turtle-"+hashCode()+" heading="+angle;
		}
	}
	
	

	/**
	 * Shortcut for <code>getPheromone(pheromone).incValue(xcor(), ycor(), value);</code>
	 * 
	 * @param pheromone the name of the targeted pheromone
	 * @param value how much to drop
	 */
	protected void emit(String pheromone, float value) {
		environment.getPheromone(pheromone).incValue(xcor(), ycor(), value);
			if (logger != null)
				logger.finest("emitting " + value + " of " + pheromone + " on "
						+ position);
	}

	/**
	 * @return ant int which is the x of the patch 
	 */
	public int xcor() {
//		return (int) x;
		return position.x;
	}

	/**
	 * @return ant int which is the y of the patch 
	 */
	public int ycor() {
//		return (int) y;
		return position.y;
	}

	/**
	 * shortcut for <code>wiggle(45)</code>
	 */
	public void wiggle() {
		wiggle(45);
	}

	public void wiggle(int range) {
		randomHeading(range);
		fd(1);
	}


	/**
	 * @param range
	 */
	public void randomHeading(int range) {
		turnRight(generator.nextFloat() * range);
		turnLeft(generator.nextFloat() * range);
	}

	public void turnRight(double a) {
		setHeading(angle - a);
	}

	public void turnLeft(double a) {
		setHeading(angle + a);
	}

	/** set the turtle heading to the value of direction */
	public void setHeading(final double direction) {
		angle = direction % 360;
		if (angle < 0)
			angle += 360;
		final double radian = toRadians(angle);
		angleSin = sin(radian);
		angleCos = cos(radian);
	}

	/**
	 * Makes turtle moves forward of nb space units
	 * @param nb number of space units
	 */
	public void fd(final double nb) {
		moveTo(x + angleCos * nb, y + angleSin * nb);
	}
	
	/**
	 * Shortcut for fd(1)
	 */
	public void step(){
		fd(1);
	}

	/**
	 * teleport the turtle to patch (a,b). Can be used as a jump primitive:
	 * MoveTo(xcor()+10,ycor())
	 */
	public void moveTo(final double a, final double b) {
		x = normalizeX(a);
		y = normalizeY(b);
		position.removeAgent(this);
		environment.getPatch((int) x, (int) y).addAgent(this);
//		if (environment != null) {//TODO test vs no test : add then remove
//			final int lastXcor = xcor();
//			final int lastYcor = ycor();
//			x = normeValue(a, environment.getWidth());
//			y = normeValue(b, environment.getHeight());
//			final int newXcor = (int) x;
//			final int newYcor = (int) y;
//			changePatch = lastXcor != newXcor || lastYcor != newYcor;
//			if (changePatch) {
//				position.removeAgent(this);
//				environment.getPatch(newXcor, newYcor).addAgent(this);
//			}
//		}
//		else{
//			x = a;
//			y = b;
//		}
	}
	
	/**
	 * Check if this is true and reset the property.
	 * @return <code>true</code> if the last move has changed 
	 * the patch the turtle is on
	 */
	public boolean isPatchChanged(){
		final boolean value = changePatch;
		changePatch = false;
		return value;
	}
	
	public List<Turtle> getPatchOtherTurtles() {
		List<Turtle> l = getPatch().getTurtles();
		l.remove(this);
		return l;
	}

	public List<Turtle> getOtherTurtles(int inRadius, boolean includeThisPatch) {
		List<Turtle> l = getPatch().getTurtles(inRadius, includeThisPatch);
		l.remove(this);
		return l;
	}
	
	public List<Turtle> getOtherTurtlesWithRole(int inRadius, boolean includeThisPatch, final String role) {
		List<Turtle> l = getPatch().getTurtlesWithRole(inRadius, includeThisPatch, role);
		l.remove(this);
		return l;
	}
	
	public <T extends Turtle> List<T> getOtherTurtlesWithRole(int inRadius, boolean includeThisPatch, final String role, Class<T> turtleType) {
		List<T> l = getPatch().getTurtlesWithRole(inRadius, includeThisPatch, role, turtleType);
		l.remove(this);
		return l;
	}
	
	public <T extends Turtle> List<T> getOtherTurtles(int inRadius, boolean includeThisPatch, Class<T> turtleType) {
		List<T> l = getPatch().getTurtles(inRadius, includeThisPatch, turtleType);
		l.remove(this);
		return l;
	}
	
	/**
	 * Returns the nearest turtle from this one 
	 * within the inRadius-vicinity as a patch distance.
	 * 
	 * @param inRadius
	 * @return the corresponding turtle or <code>null</code> 
	 * there is no turtle around. In case of equality between 
	 * turtles (same patch distance or same patch), the first found is returned.
	 */
	public Turtle getNearestTurtle(int inRadius) {
		for (final Patch p : getPatch().getNeighbors(inRadius, true)) {
			for (final Turtle t : p.getTurtles()) {
					if (t != this) {
						return t;
					}
				}
			}
		return null;
	}
	
	/**
	 * Returns the nearest turtle of type T from this one
	 * according to its patch distance.
	 * 
	 * @param patchRadius the side length of the square of patches 
	 * to look into
	 * @return the corresponding turtle or <code>null</code> if
	 * there is no turtle of this type around. In case of equality between 
	 * turtles (same patch distance or same patch), the first found is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> T getNearestTurtle(int patchRadius, Class<T> turtleType) {
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			for (final Turtle t : p.getTurtles(turtleType)) {//TODO could be optimized
					if (t != this) {
						return (T) t;
					}
				}
			}
		return null;
	}
	
	/**
	 * Returns the nearest turtle of type T from this one
	 * according to its patch distance.
	 * 
	 * @param patchRadius the side length of the square of patches 
	 * to look into
	 * @return the corresponding turtle or <code>null</code> if
	 * there is no turtle of this type around. In case of equality between 
	 * turtles (same patch distance or same patch), the first found is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> T getNearestTurtleWithRole(int patchRadius, String role, Class<T> turtleType) {
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			for (final Turtle t : p.getTurtlesWithRole(role, turtleType)) {//TODO could be optimized
					if (t != this) {
						return (T) t;
					}
				}
			}
		return null;
	}
	
	/**
	 * Returns the nearest turtle of type T from this one
	 * according to its patch distance.
	 * 
	 * @param patchRadius the side length of the square of patches 
	 * to look into
	 * @return the corresponding turtle or <code>null</code> if
	 * there is no turtle of this type around. In case of equality between 
	 * turtles (same patch distance or same patch), the first found is returned.
	 */
	public Turtle getNearestTurtleWithRole(int patchRadius, String role) {
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			for (final Turtle t : p.getTurtlesWithRole(role)) {//TODO could be optimized
					if (t != this) {
						return t;
					}
				}
			}
		return null;
	}
	
	
	/**
	 * Returns the nearest turtle of type T from this one
	 * according to its real distance.
	 * 
	 * @return the corresponding turtle or <code>null</code> if
	 * there is no turtle of this type around. In case of equality between 
	 * turtles (same patch distance or same patch), the first found is returned.
	 * 
	 * @param realRadius the maximum distance of the returned turtle
	 * @param turtleType the type of turtle to seek
	 * @return a turtle or <code>null</code> if there no turtle is found
	 * 
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> T getNearestTurtle(double realRadius, Class<T> turtleType) {
		for (final Turtle t : getPatch().getTurtles((int) Math.ceil(realRadius),true,turtleType)) {//TODO could be optimized && no wrap
			if (t != this && distance(t) <= realRadius) {
				return (T) t;
			}
		}
		return null;
	}
	
	public void setHeadingTowardsNoWrap(final Turtle t){
		setHeading(towardsNoWrap(t));
	}

	/**
	 * Set the heading toward the turtle t.
	 * This has no effect if this turtle and t have exactly
	 * the same coordinates.
	 * 
	 * @param t a turtle
	 */
	public void setHeadingTowards(final Turtle t){
		try {
			setHeading(towards(t));
		} catch (ArithmeticException e) {
		}
	}

	/**
	 * Set the heading toward the turtle t.
	 * This has no effect if this turtle and t have exactly
	 * the same coordinates.
	 * 
	 * @param t a turtle
	 * @param offset the angle to add to the corresponding direction
	 */
	public void setHeadingTowards(final Turtle t, final double offset){
		try {
			setHeading(towards(t) + offset);
		} catch (ArithmeticException e) {
		}
	}


	public double towards(final Turtle t) {
		try {
			return towards(t.getX(), t.getY());
		} catch (ArithmeticException e) {
			throw new ArithmeticException("direction to self makes no sense");
		}
	}

	public double towardsNoWrap(final Turtle t) {
		return towardsNoWrap(t.getX(), t.getY());
	}

	public  double towardsNoWrap(final double a, final double b){
		return angleToPoint(normalizeX(a) - x, normalizeY(b) - y);
	}

	/**returns direction to the patch (a,b).
       If the "wrapped distance", when wrap mode is on, (around the edges of the screen)
       is shorter than the "onscreen distance," towards will report
       the direction of the wrapped path,
       otherwise it while will report the direction of the onscreen path*/
	public  double towards(final double a, final double b){
		return angleToPoint(relativeX(a), relativeY(b));
//		if (! environment.wrapMode || distanceNoWrap(a,b) <= distance(a,b)) 
//			return towardsNoWrap(a,b);
//		a = normalizeX(a);
//		b = normalizeY(b);
//		double relativeX = a-x;
//		if (abs(relativeX) > environment.getWidthRadius())
//			if (a < x) 
//				relativeX += environment.getWidth();
//			else 
//				relativeX -= environment.getWidth();
//		double relativeY = b-y;
//		if (abs(relativeY) > environment.getHeightRadius())
//			if (b < y) 
//				relativeY += environment.getHeight();
//			else 
//				relativeY -= environment.getHeight();
//		return angleToPoint(relativeX, relativeY);
	}

	/**
	 * return the direction to a location which is i,j units
	 * away 
	 * 
	 * @param u the x-coordinate of the direction vector
	 * @param v the y-coordinate of the direction vector
	 * @return the heading towards a relative location
	 */
	private double angleToPoint(final double u, final double v) {
		if (u == 0 && v == 0)
			throw new ArithmeticException("directionAngleToPoint(0,0) makes no sense");
		if(u >= 0)
			if(v > 0)
				return toDegrees(atan(v/u));
			else
				return 360.0 + toDegrees(atan(v/u));
		else
			return 180.0 + toDegrees(atan(v/u));
	}
	

	/**
	 * return the "on screen distance" between the turtle 
	 * and the coordinates (a,b).
	 * 
	 * @return a distance as a double
	 */
	 public double distanceNoWrap(final double xCoordinate, final double yCoordinate) {
		return hypot(normalizeX(xCoordinate) - x, normalizeY(yCoordinate) - y);
	}
	
	/**
	 * returns the distance from the patch (a,b).
	 * The "wrapped distance", when wrap mode is on, (around the edges of the screen)
	 * if that distance is shorter than the "onscreen distance."
	 * 
	 * @param a the a
	 * @param b the b
	 * 
	 * @return the distance as a double
	 */
		public  double distance(final double a, final double b){
			return hypot(relativeX(a),relativeY(b));
		}
		
		final private double relativeX(double a){//TODO facto and bench
			a = normalizeX(a);
			double relativeX = a - x;
			if (environment.wrapMode && abs(relativeX) > environment.getWidthRadius())
				if (a < x) 
					relativeX += environment.getWidth();
				else 
					relativeX -= environment.getWidth();
			return relativeX;
		}

		final private double relativeY(double b){
			b = normalizeX(b);
			double relativeY = b - y;
			if (environment.wrapMode && abs(relativeY) > environment.getHeightRadius())
				if (b < y) 
					relativeY += environment.getHeight();
				else 
					relativeY -= environment.getHeight();
			return relativeY;
		}

		public  double distance(final Turtle t){
			return distance(t.getX(), t.getY());
		}

		/**
		 * Returns the normalized value of x, so that 
		 * it is inside the environment's boundaries
		 * 
		 * @param x x-coordinate
		 * @return the normalized value
		 */
		final public double normalizeX(final double x){
			return environment.normalizeX(x);
		}
		
		/**
		 * Returns the normalized value of y, so that 
		 * it is inside the environment's boundaries
		 * 
		 * @param y y-coordinate
		 * @return the normalized value
		 */
		final public double normalizeY(final double y){
			return environment.normalizeY(y);
		}
		
		/**
		 * Returns the normalized value of x, so that 
		 * it is inside the environment's boundaries
		 * 
		 * @param x x-coordinate
		 * @return the normalized value
		 */
		final public double normalizeX(final int x){
			return environment.normalizeX(x);
		}
		
		/**
		 * Returns the normalized value of y, so that 
		 * it is inside the environment's boundaries
		 * 
		 * @param y y-coordinate
		 * @return the normalized value
		 */
		final public double normalizeY(final int y){
			return environment.normalizeY(y);
		}

	private void doIt() {
		try {
			wiggle();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/** return the current heading of the turtle */
	public double getHeading() {
		return angle;
	}

	public float smellAt(final String pheromoneName, final int xOffset, final int yOffset) {
		return environment.getPheromone(pheromoneName).get(xcor() + xOffset, ycor() + yOffset);
	}

	public float smellAt(Pheromone pheromone, int xOffset, int yOffset) {
		return pheromone.get(xcor() + xOffset, ycor() + yOffset);
	}

	/**
	 * Returns the heading toward the patch which has the lowest value for
	 * <code>patchVariable</code> in the immediate vicinity of the turtle
	 * 
	 * @param patchVariableName
	 * @return the heading toward the patch which has the lowest value of the
	 *         neighbors
	 */
//	public double getDirectionOfMin(String patchVariableName) {
//		return new PerceptMap(this, patchVariableName).getMinIndex() * 45;
//	}
//
//	public double getDirectionOfMaxInMyDirection(String patchVariableName) {
//		PerceptMap map = new PerceptMap(this, patchVariableName);
//		int index = map.getMaxAround(getHeading(), 45);
//		return index * 45;
//	}
//
//	public double getDirectionOfMaxInMyDirection(PerceptMap pm) {
//		int index = pm.getMaxAround(getHeading(), 45);
//		return index * 45;
//	}

	/**
	 * Returns the heading toward the patch which has the highest value for
	 * <code>patchVariable</code> considering a radius in the vicinity of the
	 * agent.
	 * 
	 * @param patchVariable
	 * @param inRadius
	 *            the highest distance from the agent which should be considered
	 * @param wrap
	 *            is the heading should be given considering the torus mode ?
	 * @return
	 */
//	public double getHeadingToMaxOf(String patchVariable, int inRadius,
//			boolean wrap) {
//		final Patch p = getPatchWithMaxOf(patchVariable, inRadius);
//		if (wrap) {
//			return towards(p.i - xcor(), p.j - ycor());
//		} else {
//			return towardsNowrap(p.i - xcor(), p.j - ycor());
//		}
//	}

	/**
	 * Returns the patch which has the highest value for
	 * <code>patchVariable</code> considering a radius in the vicinity of the
	 * agent.
	 * 
	 * @param patchVariable
	 * @param inRadius
	 *            the highest distance from the agent which should be considered
	 * @return the patch which has the highest value for
	 *         <code>patchVariable</code>
	 */
//	public Patch getPatchWithMaxOf(String patchVariable, int inRadius) {
//		double max = -Double.MAX_VALUE;
//		Patch p = null;
//		for (int i = -inRadius; i <= inRadius; i++) {
//			for (int j = -inRadius; j <= inRadius; j++) {
//				if (!(i == 0 && j == 0)) {
//					final double tmp = smell(patchVariable);
//					if (tmp > max) {
//						max = tmp;
//						p = env.getPatch(i, j);
//					}
//				}
//			}
//		}
//		return p;
//	}

//	public double getDirectionOfMinInMyDirection(String patchVariableName) {
//		return new PerceptMap(this, patchVariableName).getMinAround(getHeading(), 45) * 45;
//	}
//
//	public double getDirectionOfMinInMyDirection(PerceptMap perceptMap) {
//		return perceptMap.getMinAround(getHeading(), 45) * 45;
//	}

	/**
	 * 
	 * @return the patch the turtle is on
	 */
	public Patch getPatch() {
		return position;
	}
	
	void setPatch(Patch p){
		position = p;
	}

	public void randomHeading() {
		setHeading(generator.nextFloat()*360);
	}

	/**
	 * When the turtle switches its behavior the value of this counter is 0
	 * @return returns the number of time the current behavior has been successively activated previously
	 */
	public int getCurrentBehaviorCount() {
		return currentBehaviorCount;
	}
	
	public void setCurrentBehaviorCount(int i) {
		this.currentBehaviorCount = i;
	}
	
	/**
	 * This offers a convenient way to create a main method 
	 * that launches a simulation containing the turtle
	 * class under development. 
	 * This call only works in the main method of the turtle.
	 * 
	 * @param args
	 *           MaDKit or TurtleKit options
	 * @see #executeThisAgent(int, boolean, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisTurtle(int nbOfInstances, String... args) {
		StackTraceElement element = null;
		for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
			if(stackTraceElement.getMethodName().equals("main")){
				element  = stackTraceElement;
				break;
			}
		}
		final ArrayList<String> arguments = new ArrayList<>(Arrays.asList(
				Madkit.BooleanOption.desktop.toString(),"false"
//				Madkit.Option.configFile.toString(), "turtlekit/kernel/turtlekit.properties",
				));
		String turtles = element.getClassName()+","+nbOfInstances;
		if (args != null) {
			final List<String> asList = new ArrayList<String>(Arrays.asList(args));
			for (Iterator<String> iterator = asList.iterator(); iterator.hasNext();) {
				if(iterator.next().equals(Option.turtles.toString())){
					iterator.remove();
					turtles+=";"+iterator.next();
					iterator.remove();
				}
			}
			arguments.addAll(asList);
		}
		arguments.add(TurtleKit.Option.turtles.toString());
		arguments.add(turtles);
		new TurtleKit(arguments.toArray(new String[0]));
	}

	/**
	 * @return the environment
	 */
	public TKEnvironment getEnvironment() {
		return environment;
	}


	void setID(int id) {
		this.id = id;
	}


	/**
	 * @return the x
	 */
	public double getX() {
		return x;
	}


	/**
	 * @return the y
	 */
	public double getY() {
		return y;
	}


	void setPosition(Patch patch) {
		position = patch;
	}


	/**
	 * @return the nextAction
	 */
	Method getNextAction() {
		return nextAction;
	}


	/**
	 * @return the community
	 */
	public String getCommunity() {
		return community;
	}

	/**
	 * @deprecated {@link #activate()} should be overridden instead, beginning by super.activate();
	 */
	public void setup() {
	}

}
