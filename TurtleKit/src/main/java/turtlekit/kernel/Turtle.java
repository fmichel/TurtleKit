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

import java.lang.invoke.MethodHandle;
import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.temporal.Temporal;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javafx.scene.paint.Color;
import madkit.kernel.Activator;
import madkit.reflection.MethodHandleFinder;
import madkit.simulation.SimuAgent;
import madkit.simulation.SimuLauncher;
import madkit.simulation.scheduler.SimuTimer;
import turtlekit.agr.TKOrganization;
import turtlekit.pheromone.Pheromone;

public class Turtle<P extends Patch> extends SimuAgent {

	protected static TKGridModel grid;

	public TKGridModel<?> getGrid() {
		return getEnvironment().getGridModel();
	}

	@SuppressWarnings("unchecked")
	@Override
	public TKEnvironment<P> getEnvironment() {
		return getLauncher().getEnvironment();
	}

	/**
	 * Used in {@link #toString()}
	 */
	private static DecimalFormat df = new DecimalFormat("0.#");

	/**
	 *
	 * Max value is used to detect that default value is used
	 */
	double x = Double.MAX_VALUE;
	/**
	 * Max value is used to detect that default value is used
	 */
	double y = Double.MAX_VALUE;

	/**
	 * turtle's heading
	 */
	double heading;

	/**
	 * heading converted into cosinus
	 */
	double cosinus = 1;

	/**
	 * heading converted into sinus
	 */
	double sinus = 0; // TODO float !!

	private double angleInRadians = toRadians(0);

	/**
	 * The patch the turtle is in
	 */
	private P position;

	/**
	 * The turtle's color. Default is red.
	 */
	protected Color color = Color.RED;

	/**
	 * Count how much time the same behavior has been called in a row
	 *
	 * @see TurtleActivator
	 */
	private int currentBehaviorCount = 0;

	private LocalDateTime behaviorActivationStamp;
//	/**
//	 * Count how much time the same behavior has been called in a row
//	 *
//	 * @see TurtleActivator
//	 */
//	private Duration currentBehaviorDuration = Duration.ofDays(0);

	/**
	 * The next behavior's method to trigger
	 */
	private MethodHandle nextAction;
	/**
	 * The turtle's ID
	 */
	private int id;
	private boolean visible = true;

	/**
	 * Changes the next behavior that will be triggered and reset behavior counter
	 *
	 * @param methodName
	 */
	public void changeNextBehavior(String methodName) {
		if (methodName != null) {
			setCurrentBehaviorCount(0);
			SimuTimer<?> simuTimer = getSimuTimer();// TODO to optimize
			if (simuTimer != null && simuTimer.getCurrentTime() instanceof LocalDateTime ldt) {
				behaviorActivationStamp = LocalDateTime.from(ldt);
			}
			try {
//				setNextMethod(MethodFinder.getMethodFromTypes(getClass(), methodName));
				setNextMethod2(MethodHandleFinder.findMethodHandle(getClass(), methodName));
			} catch (NoSuchMethodException e) {
				e.printStackTrace();
			}
			getLogger().finest(() -> this.toString() + " change behavior -> " + methodName);
		} else {
			if (isAlive()) {
				killAgent(this);
			}
		}
	}

	public Duration getCurrentBehaviorDuration() {
		return Duration.between(behaviorActivationStamp, (Temporal) getSimuTimer().getCurrentTime());
	}

	public final void setNextMethod2(MethodHandle nextMethod) {
		nextAction = nextMethod;
	}

//	/**
//	 * Creates a turtle with {@link #wiggle()} as initial behavior
//	 */
//	public Turtle() {
//	}
//
//	/**
//	 * Creates a turtle with the specified initial behavior, That is a method of the
//	 * turtle's API
//	 *
//	 * @param initialBehavior the name of the method that will be used as initial
//	 *                        behavior
//	 */
//	public Turtle(String initialBehavior) {
//		changeNextBehavior(initialBehavior);
//	}

	/**
	 * Execute the behavior which has been set using
	 * {@link #changeNextBehavior(String)}. This method should be used by subclasses
	 * of {@link Activator} such as {@link TurtleActivator}
	 *
	 */
	public void executeBehavior() {
		if (getPatch() == null) // killed by another before its turn
			return;
		incrementBehaviorCount();
		try {
			getLogger().finest(() -> this.toString() + " -> " + nextAction);
//			nextAction.invoke(this);
			nextAction.invoke(this);
		} catch (InvocationTargetException e) {
			Throwable cause = e.getCause();
			cause.printStackTrace();
		} catch (Exception e) {
			getLogger().severe(() -> "Can't invoke: " + nextAction + "\n");// let the others running
			e.printStackTrace();
		} catch (AssertionError e) {
			throw e;
		} catch (Throwable e) {
			e.printStackTrace();
		}
	}

	@Override
	protected void onActivation() {
		requestRole(getCommunity(), getModelGroup(), TKOrganization.TURTLE_ROLE, null);
		getEnvironment().addTurtleToEnvironment(this);
	}

	protected void onStart() {

	}

	/**
	 * one way to identify a kind of turtle: give them a Role in the simulation.
	 * 
	 * @param role the role to give to the turtle
	 */
	public void playRole(String role) {
		requestRole(getCommunity(), getModelGroup(), role);
	}

	/**
	 * the turtle will no longer play the specified role
	 * 
	 * @param role the role to give up
	 */
	public void giveUpRole(String role) {
		leaveRole(getCommunity(), getModelGroup(), role);
	}

	public int getMinDirection(Pheromone<?> p) {
		return p.getMinDirection(xcor(), ycor());
	}

	public double getMaxDirectionInVisionCone(Pheromone phero, double visionConeAngle) {
		Patch selected = null;
		double max = -2;
		for (Patch p : getPatchesInVisionCone(visionConeAngle)) {
			final float d = (float) phero.get(p.x, p.y);
			if (d > max) {
				max = d;
				selected = p;
			}
		}
		return towards(selected);
	}

	public int getMaxDirection(Pheromone p) {
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
	public boolean isPlayingRole(String role) {
		return getOrganization().getRole(getCommunity(), getModelGroup(), role).contains(this);
	}

	/**
	 * @return the absolute index of the turtle's location in a 1D data grid
	 *         representing a 2D grid of width * height size. Its purpose is to be
	 *         used on a Pheromone to optimize several calls on its data structure.
	 */
	public int get1DIndex() {// useful because there is no check
		return position.x + position.y * getEnvironment().getWidth();
	}

	/**
	 * @param x absolute
	 * @param y absolute
	 * @return the absolute index in a 1D data grid representing a 2D grid of
	 *         width,height size. This should be used with {@link #xcor()} and
	 *         {@link #ycor()} to ensure that this x,y point in inside the
	 *         environment's boundaries. Its purpose is to be used on a Pheromone
	 */
	public int get1DIndex(int x, int y) {
		return getEnvironment().get1DIndex(x, y);
	}

	// /** return the number of turtles in the patch situated at (a,b) units away */
	// public final int countTurtlesAt(final int a, final int b) {
	// return env.turtlesCountAt(normeValue(a + xcor(), env.getWidth()),
	// normeValue(b + ycor(), env.getHeight()));
	// }

	// /**
	// * @return <code>true</code> if the next patch (dx dy) is a different patch
	// * and is occupied by a turtle
	// */
	// public boolean isNextPatchOccupied() {
	// return false;
	// }

	public int getWorldWidth() {
		return getEnvironment().getWidth();
	}

	public int getWorldHeight() {
		return getEnvironment().getHeight();
	}

	public List<P> getPatchNeighborhood(int radius, boolean includeMyPatch) {
		return (List<P>) getPatch().getNeighbors(radius, includeMyPatch);
	}

	/** Teleports the turtle to the center patch */
	public final void home() {
		if (position != null) {
			moveTo(getEnvironment().getWidth() / 2, getEnvironment().getHeight() / 2);
		}
	}

	public boolean isInVisionCone(Turtle<?> t, double visionConeAngle) {
		return differenceTwoAngles(getHeading(), towards(t)) <= visionConeAngle / 2;
	}

	public boolean isInVisionCone(Patch p, double visionConeAngle) {
		return differenceTwoAngles(getHeading(), towards(p)) <= visionConeAngle / 2;
	}

	public List<Patch> getPatchesInVisionCone(double visionConeAngle) {
		List<Patch> patches = new ArrayList<>();
		for (Patch p : getPatch().getNeighbors(1, false)) {
			if (isInVisionCone(p, visionConeAngle))
				patches.add(p);
		}
		return patches;
	}

	/**
	 * Returns the angle between the turtle and a location
	 * 
	 * @param targetA x coordinate
	 * @param targetB y coordinate
	 * @return the angle between the turtle and a location
	 */
	public double differenceTwoAngles(double targetA, double targetB) {
		double d = Math.abs(targetA - targetB) % 360;
		return d > 180 ? 360 - d : d;
	}

	public Patch getCenterPatch() {
		return getEnvironment().getPatch(getEnvironment().getWidth() / 2, getEnvironment().getHeight() / 2);
	}

	public void setPatchColor(Color c) {
		position.setColor(c);
	}

	public Color getPatchColor() {
		return position.getColor();
	}

	/**
	 * @return the next patch if fd(1) leads to a different patch, <code>null</code>
	 *         otherwise
	 */
	public Patch getNextPatch() {
		final int dx = dx();
		final int dy = dy();
		if (dx != 0 || dy != 0)
			return getEnvironment().getPatch(dx + xcor(), dy + ycor());
		return null;
	}

	public double getFieldValue(Pheromone<Float> p) {
		return p.get(xcor(), ycor());
	}

	/** Teleports the turtle to a random location */
	public void randomLocation() {
		moveTo(prng().nextFloat() * getWorldWidth(), prng().nextFloat() * getWorldHeight());
	}

	/**
	 * Get the patch situated at (a,b) units away
	 * 
	 * @param a the x offset
	 * @param b the y offset
	 * @return the patch situated at (a,b) units away
	 */
	public Patch getPatchAt(final int a, final int b) {
		return getEnvironment().getPatch(a + xcor(), b + ycor());
	}

	/**
	 * Gets the corresponding pheromone or create a new one using defaults
	 * parameters : 50% for both the evaporation rate and the diffusion rate.
	 *
	 * @param name the pheromone's name
	 * @return the pheromone
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public Pheromone<Float> getPheromone(String name) {
		// System.exit(0);
		return getEnvironment().getPheromone(name);
	}

	public Collection<Pheromone<Float>> getPheromones() {
		return getEnvironment().getPheromones();
	}

	/**
	 * Gets the direction where there is a minimum of this pheromone
	 *
	 * @param pheromone the pheromone's name
	 * @return the direction to take
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public double getPheroMinDirection(Pheromone<?> pheromone) {
		return pheromone.getMinDirection(xcor(), ycor());
	}

	/**
	 * Gets the direction where there is a maximum of this pheromone
	 *
	 * @param pheromone the pheromone's name
	 * @return the direction to take
	 * @see TKEnvironment#getPheromone(String, int, int)
	 */
	public double getPheroMaxDirection(Pheromone<?> pheromone) {
		return pheromone.getMaxDirection(xcor(), ycor());
	}

	public String defaultBehavior() {
		wiggle();
		return "defaultBehavior";
	}

	public void incrementBehaviorCount() {
		currentBehaviorCount++;
	}

	public void setX(final double a) {
		moveTo(a, y);
	}

	public void setY(final double b) {
		moveTo(x, b);
	}

	public void setXY(final double a, final double b) {
		moveTo(a, b);
	}

	/**
	 * launch a turtle at the creator position (xcor,ycor) and returns the given ID
	 * to the new turtle
	 *
	 * @param t
	 * @return the ID given to the turtle
	 */
	public int createTurtleHere(Turtle<?> t) {
		return getEnvironment().createTurtle(t, x, y);
	}

	/**
	 * Launch a turtle with predefined coordinates
	 *
	 * @param t
	 * @param x
	 * @param y
	 * @return the ID given to the turtle
	 */
	public int createTurtle(Turtle t, double x, double y) {
		return getEnvironment().createTurtle(t, x, y);
	}

	public void printBehaviorInFinerLoggingLevel() {
		final String string = nextAction.toString();
		getLogger().finer(() -> toString() + " " + string.substring(string.lastIndexOf('.') + 1, string.length() - 2)
				+ "(" + getCurrentBehaviorCount() + ")");
	}

	/**
	 * Launch a turtle with a random location.
	 *
	 * @param t
	 * @return the ID given to the turtle
	 */
	public int createTurtle(Turtle t) {
		return getEnvironment().createTurtle(t);
	}

	/**
	 * the id of this turtle
	 *
	 * @return the id of this turtle
	 */
	public int getID() {
		return id;
	}

	/**
	 * return the x-increment if the turtle were to take one step forward in its
	 * current heading.
	 * 
	 * @return the x-increment if the turtle were to take one step forward in its
	 */
	public int dx() {
		return ((int) (x + cosinus)) - position.x;
		// return (int) (round(x+angleCos)-round(x));
	}

	/**
	 * return the int y-increment if the turtle were to take one step forward in its
	 * current heading.
	 * 
	 * @return the int y-increment if the turtle were to take one step forward in
	 *         its current heading.
	 */
	public int dy() {
		return ((int) (y + sinus)) - position.y;
		// return (int) (round(y+angleSin)-round(y));
	}

	public void setColor(Color c) {
		color = c;
	}

	// public int getPositionCode(){
	// return position.i + position.j * env.getWidth();
	// }
	//
	// public int getPositionCode(int i, int j){
	// int width = env.getWidth();
	// return normeValue(i, width) + normeValue(j, env.getHeight()) * width;
	// }

	public Color getColor() {
		return color;
	}

	public double smell(String pheromone) {
		return getEnvironment().smell(pheromone, position.x, position.y);// TODO test perf vs xcor()
	}

	public String toString() {
		try {
			return getClass().getSimpleName() + "-" + hashCode() + "@(" + xcor() + "," + ycor() + ") H="
					+ df.format(heading);
		} catch (NullPointerException e) {
			return "turtle-" + hashCode() + " heading=" + heading;
		}
	}

	/**
	 * Shortcut for
	 * <code>getPheromone(pheromone).incValue(xcor(), ycor(), value);</code>
	 *
	 * @param pheromone the name of the targeted pheromone
	 * @param value     how much to drop
	 */
	protected void emit(String pheromone, float value) {
		getEnvironment().getPheromone(pheromone).incValue(xcor(), ycor(), value);
		// if (logger != null)
		// logger.finest("emitting " + value + " of " + pheromone + " on "
		// + position);
	}

	/**
	 * @return ant int which is the x of the patch
	 */
	public int xcor() {
		// return (int) x;
		return position.x;
	}

	/**
	 * @return ant int which is the y of the patch
	 */
	public int ycor() {
		// return (int) y;
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

	public void wiggle(double range) {
		randomHeading(range);
		fd(1);
	}

	/**
	 * @param range
	 */
	public void randomHeading(double range) {
		turnRight(prng().nextFloat() * range);
		turnLeft(prng().nextFloat() * range);
	}

	public void turnRight(double a) {
		setHeading(heading - a);
	}

	public void turnLeft(double a) {
		setHeading(heading + a);
	}

	/**
	 * set the turtle heading to the value of direction
	 * 
	 * @param direction the new heading
	 */
	public void setHeading(double direction) {
		heading = direction % 360;
		if (heading < 0)
			heading += 360;
		angleInRadians = toRadians(heading);
		sinus = sin(angleInRadians);
		cosinus = cos(angleInRadians);
	}

	/**
	 * Makes turtle moves forward of nb space units
	 *
	 * @param nb number of space units
	 */
	public void fd(final double nb) {
		moveTo(x + cosinus * nb, y + sinus * nb);
	}

	/**
	 * Shortcut for fd(1)
	 */
	public void step() {
		fd(1);
	}

	public void die() {
		changeNextBehavior(null);
	}

	@Override
	protected void onEnd() {
		position.removeAgent(this);
		setPatch(null);// FIXME is it necessary ? GUI pb involved
	}

	/**
	 * teleport the turtle to patch (a,b). Can be used as a jump primitive:
	 * MoveTo(xcor()+10,ycor())
	 */
	public void moveTo(final double a, final double b) {
		if (getEnvironment() != null) {
			x = normalizeX(a);
			y = normalizeY(b);
			position.removeAgent(this);
			getEnvironment().getPatch((int) x, (int) y).addAgent(this);
			// if (environment != null) {//TODO test vs no test : add then remove
			// final int lastXcor = xcor();
			// final int lastYcor = ycor();
			// x = normeValue(a, environment.getWidth());
			// y = normeValue(b, environment.getHeight());
			// final int newXcor = (int) x;
			// final int newYcor = (int) y;
			// changePatch = lastXcor != newXcor || lastYcor != newYcor;
			// if (changePatch) {
			// position.removeAgent(this);
			// environment.getPatch(newXcor, newYcor).addAgent(this);
			// }
		} else {
			x = a;
			y = b;
		}
	}

	public <T extends Turtle<?>> List<T> getPatchOtherTurtles() {
		List<T> l = (List<T>) getPatch().getTurtles();
		l.remove(this);
		return l;
	}

	public List<Turtle<?>> getOtherTurtles(int inRadius, boolean includeThisPatch) {
		List<Turtle<?>> l = getPatch().getTurtles(inRadius, includeThisPatch);
		if (includeThisPatch) {
			l.remove(this);
		}
		return l;
	}

	public List<Turtle<?>> getOtherTurtlesWithRole(int inRadius, boolean includeThisPatch, final String role) {
		List<Turtle<?>> l = getPatch().getTurtlesWithRole(inRadius, includeThisPatch, role);
		if (includeThisPatch) {
			l.remove(this);
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getOtherTurtlesWithRole(int inRadius, boolean includeThisPatch,
			final String role, Class<T> turtleType) {
		List<T> l = getPatch().getTurtlesWithRole(inRadius, includeThisPatch, role, turtleType);
		if (includeThisPatch) {
			l.remove(this);
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getOtherTurtles(int inRadius, boolean includeThisPatch,
			Class<T> turtleType) {
		List<T> l = getPatch().getTurtles(inRadius, includeThisPatch, turtleType);
		if (includeThisPatch) {
			l.remove(this);
		}
		return l;
	}

	/**
	 * Returns the nearest turtle from this one within the inRadius-vicinity as a
	 * patch distance.
	 *
	 * @param inRadius
	 * @return the corresponding turtle or <code>null</code> there is no turtle
	 *         around. In case of equality between turtles (same patch distance or
	 *         same patch), the first found is returned.
	 */
	public Turtle<?> getNearestTurtle(int inRadius) {
		for (final Patch p : getPatch().getNeighbors(inRadius, true)) {
			for (final Turtle<?> t : p.getTurtles()) {
				if (t != this) {
					return t;
				}
			}
		}
		return null;
	}

	/**
	 * Returns the nearest turtle of type T from this one according to its patch
	 * distance.
	 *
	 * @param patchRadius the side length of the square of patches to look into
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         of this type around. In case of equality between turtles (same patch
	 *         distance or same patch), the first found is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> T getNearestTurtle(int patchRadius, Class<T> turtleType) {
		// getPatch().getNeighbors(patchRadius, true).parallelStream().forEach(p ->
		// p.getTurtles(turtleType).parallelStream().anyMatch(t -> {return (t!=
		// DefaultTurtle.this);}));//cannot use that
		// because we need the first
		// getPatch().getNeighbors(patchRadius, true).forEach(p ->
		// p.getTurtles(turtleType).parallelStream().anyMatch(t
		// -> {return (t!= DefaultTurtle.this);}));//cannot use that because we need the first
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			final List<Turtle<?>> turtles = new ArrayList<Turtle<?>>(p.getTurtles(turtleType));
			for (final Turtle<?> t : turtles) {// TODO could be optimized
				if (t != this) {
					return (T) t;
				}
			}
		}
		return null;
	}

	/**
	 * Returns the <code>howMany</code> nearest turtles of type T from this one
	 * according to a patch distance.
	 *
	 * @param patchRadius the side length of the square of patches to look into
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         of this type around. In case of equality between turtles (same patch
	 *         distance or same patch), the first found is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getNearestTurtles(int patchRadius, int howMany, Class<T> turtleType) {
		List<T> l = getPatch().getTurtles(turtleType);
		l.remove(this);
		if (l.size() >= howMany) {
			return l.subList(0, howMany);
		}
		howMany -= l.size();
		for (Patch p : getPatch().getNeighbors(patchRadius, false)) {
			List<T> tmp = p.getTurtles(turtleType);
			for (T t : tmp) {
				l.add(t);
				if (--howMany == 0) {
					return l;
				}
			}
		}
		return l;
	}

	// /**
	// * Returns the <code>howMany</code> nearest turtles of type T from this one
	// * according to a patch distance.
	// *
	// * @param inRadius the side length of the square of patches
	// * to look into
	// * @return the corresponding turtle or <code>null</code> if
	// * there is no turtle of this type around. In case of equality between
	// * turtles (same patch distance or same patch), the first found is returned.
	// */
	// @SuppressWarnings("unchecked")
	// public List<DefaultTurtle> getNearestTurtles(int inRadius, int howMany) {
	// List<DefaultTurtle> l = getPatch().getTurtles();
	// l.remove(this);
	// if (l.size() >= howMany) {
	// return l.subList(0, howMany);
	// }
	// howMany -= l.size();
	// for (Patch p : getPatch().getNeighbors(inRadius, false)) {
	// List<T> tmp = p.getTurtles();
	// for (T t : tmp) {
	// l.add(t);
	// if(--howMany == 0){
	// return l;
	// }
	// }
	// }
	// return l;
	// }

	/**
	 * Returns the nearest turtle of type T from this one according to its patch
	 * distance.
	 *
	 * @param patchRadius the side length of the square of patches to look into
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         of this type around. In case of equality between turtles (same patch
	 *         distance or same patch), the first found is returned.
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> T getNearestTurtleWithRole(int patchRadius, String role, Class<T> turtleType) {
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			for (final Turtle<?> t : p.getTurtlesWithRole(role, turtleType)) {// TODO could be optimized
				if (t != this) {
					return (T) t;
				}
			}
		}
		return null;
	}

	/**
	 * Returns the nearest turtle of type T from this one according to its patch
	 * distance.
	 *
	 * @param patchRadius the side length of the square of patches to look into
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         of this type around. In case of equality between turtles (same patch
	 *         distance or same patch), the first found is returned.
	 */
	public Turtle<?> getNearestTurtleWithRole(int patchRadius, String role) {
		for (Patch p : getPatch().getNeighbors(patchRadius, true)) {
			for (final Turtle<?> t : p.getTurtlesWithRole(role)) {// TODO could be optimized
				if (t != this) {
					return t;
				}
			}
		}
		return null;
	}

	/**
	 * Returns the nearest turtle of type T from this one according to its real
	 * distance.
	 *
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         of this type around. In case of equality between turtles (same patch
	 *         distance or same patch), the first found is returned.
	 *
	 * @param realRadius the maximum distance of the returned turtle
	 * @param turtleType the type of turtle to seek
	 * @return a turtle or <code>null</code> if there no turtle is found
	 *
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> T getNearestTurtle(double realRadius, Class<T> turtleType) {
		for (final Turtle<?> t : getPatch().getTurtles((int) Math.ceil(realRadius), true, turtleType)) {// TODO
			// could be
			// optimized && no
			// wrap
			if (t != this && distance(t) <= realRadius) {
				return (T) t;
			}
		}
		return null;
	}

	public void setHeadingTowardsNoWrap(final Turtle<?> t) {
		setHeading(towardsNoWrap(t));
	}

	/**
	 * Set the heading toward the turtle t. This has no effect if this turtle and t
	 * have exactly the same coordinates.
	 *
	 * @param t a turtle
	 */
	public void setHeadingTowards(final Turtle<?> t) {
		try {
			setHeading(towards(t));
		} catch (ArithmeticException e) {
		}
	}

	/**
	 * Set the heading toward the turtle t. This has no effect if this turtle and t
	 * have exactly the same coordinates.
	 *
	 * @param t      a turtle
	 * @param offset the heading to add to the corresponding direction
	 */
	public void setHeadingTowards(final Turtle<?> t, final double offset) {
		try {
			setHeading(towards(t) + offset);
		} catch (ArithmeticException e) {
		}
	}

	public double towards(final Turtle<?> t) {
		try {
			return towards(t.getX(), t.getY());
		} catch (ArithmeticException e) {
			throw new ArithmeticException("direction to self makes no sense");
		}
	}

	public double towards(Patch p) {
		try {
			return towards(p.x + .5f, p.y + .5f);
		} catch (ArithmeticException e) {
			throw new ArithmeticException("direction to self makes no sense");
		}
	}

	public double towardsNoWrap(final Turtle<?> t) {
		return towardsNoWrap(t.getX(), t.getY());
	}

	public double towardsNoWrap(final double a, final double b) {
		return angleToPoint(normalizeX(a) - x, normalizeY(b) - y);
	}

	/**
	 * returns direction to the patch (a,b). If the "wrapped distance", when wrap
	 * mode is on, (around the edges of the screen) is shorter than the "onscreen
	 * distance," towards will report the direction of the wrapped path, otherwise
	 * it while will report the direction of the onscreen path
	 */
	public double towards(final double a, final double b) {
		return angleToPoint(relativeX(a), relativeY(b));
		// if (! environment.wrapMode || distanceNoWrap(a,b) <= distance(a,b))
		// return towardsNoWrap(a,b);
		// a = normalizeX(a);
		// b = normalizeY(b);
		// float relativeX = a-x;
		// if (abs(relativeX) > environment.getWidthRadius())
		// if (a < x)
		// relativeX += environment.getWidth();
		// else
		// relativeX -= environment.getWidth();
		// float relativeY = b-y;
		// if (abs(relativeY) > environment.getHeightRadius())
		// if (b < y)
		// relativeY += environment.getHeight();
		// else
		// relativeY -= environment.getHeight();
		// return angleToPoint(relativeX, relativeY);
	}

	/**
	 * return the direction to a location which is i,j units away
	 *
	 * @param u the x-coordinate of the direction vector
	 * @param v the y-coordinate of the direction vector
	 * @return the heading towards a relative location
	 */
	private double angleToPoint(final double u, final double v) {// FIXME
		if (u == 0 && v == 0)
			throw new ArithmeticException("directionAngleToPoint(0,0) makes no sense");
		if (u >= 0)
			if (v > 0)
				return toDegrees(atan(v / u));
			else {
				return (360.0 + toDegrees(atan(v / u)));
			}
		else
			return (180.0 + toDegrees(atan(v / u)));
	}

	/**
	 * return the "on screen distance" between the turtle and the coordinates (a,b).
	 *
	 * @return a distance as a float
	 */
	public double distanceNoWrap(final double xCoordinate, final double yCoordinate) {
		return (double) hypot(normalizeX(xCoordinate) - x, normalizeY(yCoordinate) - y);
	}

	/**
	 * returns the distance from the patch (a,b). The "wrapped distance", when wrap
	 * mode is on, (around the edges of the screen) if that distance is shorter than
	 * the "onscreen distance."
	 *
	 * @param a the a
	 * @param b the b
	 *
	 * @return the distance as a double
	 */
	public double distance(final double a, final double b) {
		return (double) hypot(relativeX(a), relativeY(b));
	}

	private final double relativeX(double a) {// TODO facto and bench
		a = normalizeX(a);
		double relativeX = a - x;
		if (grid.isTorusModeOn() && abs(relativeX) > getEnvironment().getWidthRadius())
			if (a < x)
				relativeX += getEnvironment().getWidth();
			else
				relativeX -= getEnvironment().getWidth();
		return relativeX;
	}

	private final double relativeY(double b) {
		b = normalizeX(b);
		double relativeY = b - y;
		if (grid.isTorusModeOn() && abs(relativeY) > getEnvironment().getHeightRadius())
			if (b < y)
				relativeY += getEnvironment().getHeight();
			else
				relativeY -= getEnvironment().getHeight();
		return relativeY;
	}

	public double distance(final Turtle<?> t) {
		return distance(t.getX(), t.getY());
	}

	/**
	 * Returns the normalized value of x, so that it is inside the environment's
	 * boundaries
	 *
	 * @param x x-coordinate
	 * @return the normalized value
	 */
	public final double normalizeX(final double x) {
		return getEnvironment().normalizeX(x);
	}

	/**
	 * Returns the normalized value of y, so that it is inside the environment's
	 * boundaries
	 *
	 * @param y y-coordinate
	 * @return the normalized value
	 */
	public final double normalizeY(final double y) {
		return getEnvironment().normalizeY(y);
	}

	/**
	 * Returns the normalized value of x, so that it is inside the environment's
	 * boundaries
	 *
	 * @param x x-coordinate
	 * @return the normalized value
	 */
	public final double normalizeX(final int x) {
		return getEnvironment().normalizeX(x);
	}

	/**
	 * Returns the normalized value of y, so that it is inside the environment's
	 * boundaries
	 *
	 * @param y y-coordinate
	 * @return the normalized value
	 */
	public final double normalizeY(final int y) {
		return getEnvironment().normalizeY(y);
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
		return heading;
	}

	/**
	 * return the current heading of the turtle
	 * 
	 * @return the current heading of the turtle
	 */
	public double getHeadingInRadians() {
		return angleInRadians;
	}

	public double smellAt(final String pheromoneName, final int xOffset, final int yOffset) {
		return (double) getEnvironment().getPheromone(pheromoneName).get(xcor() + xOffset, ycor() + yOffset);
	}

	public double smellAt(Pheromone<Float> pheromone, int xOffset, int yOffset) {
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
	// public float getDirectionOfMin(String patchVariableName) {
	// return new PerceptMap(this, patchVariableName).getMinIndex() * 45;
	// }
	//
	// public float getDirectionOfMaxInMyDirection(String patchVariableName) {
	// PerceptMap map = new PerceptMap(this, patchVariableName);
	// int index = map.getMaxAround(getHeading(), 45);
	// return index * 45;
	// }
	//
	// public float getDirectionOfMaxInMyDirection(PerceptMap pm) {
	// int index = pm.getMaxAround(getHeading(), 45);
	// return index * 45;
	// }

	/**
	 * Returns the heading toward the patch which has the highest value for
	 * <code>patchVariable</code> considering a radius in the vicinity of the agent.
	 *
	 * @param patchVariable
	 * @param inRadius      the highest distance from the agent which should be
	 *                      considered
	 * @param wrap          is the heading should be given considering the torus
	 *                      mode ?
	 * @return
	 */
	// public float getHeadingToMaxOf(String patchVariable, int inRadius,
	// boolean wrap) {
	// final Patch p = getPatchWithMaxOf(patchVariable, inRadius);
	// if (wrap) {
	// return towards(p.i - xcor(), p.j - ycor());
	// } else {
	// return towardsNowrap(p.i - xcor(), p.j - ycor());
	// }
	// }

	/**
	 * Returns the patch which has the highest value for <code>patchVariable</code>
	 * considering a radius in the vicinity of the agent.
	 *
	 * @param patchVariable
	 * @param inRadius      the highest distance from the agent which should be
	 *                      considered
	 * @return the patch which has the highest value for <code>patchVariable</code>
	 */
	// public Patch getPatchWithMaxOf(String patchVariable, int inRadius) {
	// float max = -float.MAX_VALUE;
	// Patch p = null;
	// for (int i = -inRadius; i <= inRadius; i++) {
	// for (int j = -inRadius; j <= inRadius; j++) {
	// if (!(i == 0 && j == 0)) {
	// final float tmp = smell(patchVariable);
	// if (tmp > max) {
	// max = tmp;
	// p = env.getPatch(i, j);
	// }
	// }
	// }
	// }
	// return p;
	// }

	// public float getDirectionOfMinInMyDirection(String patchVariableName) {
	// return new PerceptMap(this, patchVariableName).getMinAround(getHeading(), 45)
	// * 45;
	// }
	//
	// public float getDirectionOfMinInMyDirection(PerceptMap perceptMap) {
	// return perceptMap.getMinAround(getHeading(), 45) * 45;
	// }

	/**
	 *
	 * @return the patch the turtle is on
	 */
	public P getPatch() {
		return position;
	}

	void setPatch(P p) {
		position = p;
	}

	public void randomHeading() {
		setHeading(prng().nextFloat() * 360);
	}

	/**
	 * When the turtle switches its behavior the value of this counter is reset to 1
	 *
	 * @return returns the number of time the current behavior has been successively
	 *         activated previously
	 */
	public int getCurrentBehaviorCount() {
		return currentBehaviorCount;
	}

	public void setCurrentBehaviorCount(int i) {
		this.currentBehaviorCount = i;
	}

	/**
	 * This offers a convenient way to create a main method that launches a
	 * simulation containing the turtle class under development. This call only
	 * works in the main method of the turtle.
	 *
	 * @param args MaDKit or TurtleKit options
	 * @see #executeThisTurtle(int, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisTurtle(int nbOfInstances, String... args) {
		final List<String> arguments = new ArrayList<>(List.of(args));
		String className = Utils.getClassFromMainStackTrace();
		try {
			arguments.addAll(SimuLauncher.getEngineAgentsArgsFrom(Class.forName(className)));
		} catch (ClassNotFoundException | IllegalAccessException | InvocationTargetException | NoSuchMethodException
				| SecurityException e) {
			e.printStackTrace();
		}
//		arguments.add("--uiParameterClass");
//		arguments.add(className);
		arguments.addAll(List.of("-t", className + "," + nbOfInstances));
		arguments.addAll(Utils.getPatchClassArgsFromClasses(Turtle.class, className));
		TutleKit4.main(arguments.toArray(new String[arguments.size()]));
	}

	// /**
	// * @return the environment
	// */
	// @Override
	// public TKEnvironment getEnvironment() {
	// return super.getEnvironment();
	// }

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
		position = (P) patch;
	}

	/**
	 * Used by default viewers.
	 *
	 * @return <code>true</code> if the turtle should be displayed by viewers.
	 */
	public boolean isVisible() {
		return visible;
	}

	/**
	 * Can be used to tell viewers that this turtle should not be rendered when
	 * displayed
	 *
	 * @param visible the turtle will not be visible in default viewers if set to
	 *                <code>true</code>
	 */
	public void setVisible(boolean visible) {
		this.visible = visible;
	}

	/**
	 * @return the cosinus
	 */
	public double getCosinus() {
		return cosinus;
	}

	/**
	 * @return the sinus
	 */
	public double getSinus() {
		return sinus;
	}

}
