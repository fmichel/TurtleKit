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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import madkit.kernel.MadkitClassLoader;
import madkit.kernel.Probe;
import madkit.simulation.environment.Environment2D;
import turtlekit.agr.TKOrganization;
import turtlekit.cuda.CudaEngine;
import turtlekit.cuda.CudaGPUGradientsPhero;
import turtlekit.cuda.CudaPheromone;
import turtlekit.cuda.CudaPheromoneWithBlock;
import turtlekit.cuda.CudaPlatform;
import turtlekit.cuda.GPUSobelGradientsPhero;
import turtlekit.pheromone.CPUStreamPheromone;
import turtlekit.pheromone.CPU_SobelPheromone;
import turtlekit.pheromone.Pheromone;

public class TKEnvironment<P extends Patch> extends Environment2D {

	private Consumer<Patch> initFunction;
	private Map<String, Pheromone<Float>> pheromones;
	private Map<String, Probe> turtleProbes;
	private int heightRadius;
	private int widthRadius;
	private final static transient AtomicInteger turtleCounter = new AtomicInteger(0);
	private int[] neighborsIndexes;// TODO or remove
	protected boolean GPU_GRADIENTS = false;
	private boolean synchronizeGPU = true;
	private TKGridModel<P> gridModel;
	
	@Override
	protected void onActivation() {
		super.onActivation();
		setWidth(getKernelConfig().getInt("width"));
		setHeight(getKernelConfig().getInt("height"));
		initGridModel();
//		onInit();
	}

	public void askPatchesOnStartup(Consumer<Patch> c) {
		initFunction = c;
	}

	public void askTurtles(Consumer<Turtle<?>> turtleFunction) {
		if (turtleFunction != null) {
			getTurtlesWithRoles(TKOrganization.TURTLE_ROLE).forEach(turtleFunction);
		}
	}

	@Override
	public void onSimulationStart() {
		super.onSimulationStart();
		askPatches((Consumer<P>) initFunction);
	}
	
	public void askPatches(Consumer<P> c) {
		if (c != null) {
			getGridModel().stream().forEach(c);
		}
	}

	public void askTurtes(Consumer<P> c) {//FIXME
		getGridModel().stream().forEach(c);
	}

	public boolean isSynchronizeGPU() {
		return synchronizeGPU;
	}

	public void setSynchronizeGPU(boolean synchronizeGPU) {
		this.synchronizeGPU = synchronizeGPU;
	}

	public TKEnvironment() {
		pheromones = new TreeMap<>();
		// patchClass = (Class<P>) TypeResolver.resolveRawArguments(TKEnvironment.class,
		// this.getClass())[0];
		// System.err.println(patchClass);
	}

	public Collection<Pheromone<Float>> getPheromones() {
		return pheromones.values();
	}

	public Map<String, Pheromone<Float>> getPheromonesMap() {
		return pheromones;
	}

	@Override
	protected void onEnd() {
		super.onEnd();
		for (Pheromone<Float> i : pheromones.values()) {
			if (i instanceof CudaPheromone) {
				((CudaPheromone) i).freeMemory();
			}
		}
		pheromones = null;
		turtleProbes = null;
	}

	private void initGridModel() {
		Class<? extends Patch> patchClass;
		try {
			String patch = getKernelConfig().getString("patchClass", "turtlekit.kernel.Patch");
			patchClass = (Class<? extends Patch>) MadkitClassLoader.getLoader().loadClass(patch);
			// String patch =
			// getKernelConfiguration().getString("patchClass","turtlekit.kernel.Patch");
			// patchClass = (Class<? extends Patch>)
			// MadkitClassLoader.getLoader().loadClass(patch);
		}
		catch(ClassNotFoundException e) {
			patchClass = Patch.class;
			getLogger().severe(e.toString() + " -> using " + patchClass.getName());
		}
		var wrapMode = getKernelConfig().getBoolean("wrapMode", true);
		gridModel = new TKGridModel<>(this, wrapMode, getWidth(), getHeight(), patchClass);
		initNeighborsIndexes();
	}

	/**
	 *
	 */
	private void initNeighborsIndexes() {
		neighborsIndexes = new int[getWidth() * getHeight() * 8];
		for (int i = 0; i < getWidth(); i++) {
			for (int j = 0; j < getHeight(); j++) {
				final int retrieveIndex = retrieveIndex(i, j);
				int k = retrieveIndex * 8;
				neighborsIndexes[k] = get1DIndex(i, normalizeCoordinate(j - 1, getHeight()));// TODO not used and thus
				// useless (check that)
				neighborsIndexes[++k] = get1DIndex(i, normalizeCoordinate(j + 1, getHeight()));
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i - 1, getWidth()),
						normalizeCoordinate(j - 1, getHeight()));
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i - 1, getWidth()), j);
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i - 1, getWidth()),
						normalizeCoordinate(j + 1, getHeight()));
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i + 1, getWidth()),
						normalizeCoordinate(j - 1, getHeight()));
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i + 1, getWidth()), j);
				neighborsIndexes[++k] = get1DIndex(normalizeCoordinate(i + 1, getWidth()),
						normalizeCoordinate(j + 1, getHeight()));
			}
		}
	}

	// private void initPatchGrid() {
	// patchGrid = new Patch[width * height];
	// Class<? extends Patch> patchClass;
	// try {
	// patchClass = (Class<? extends Patch>)
	// MadkitClassLoader.getLoader().loadClass(getMadkitProperty(patch));
	// } catch (ClassNotFoundException e) {
	// patchClass = Patch.class;
	// getLogger().severe(e.toString()+" -> using "+patchClass.getName());
	// }
	// for (int i = 0; i < width; i++) {
	// for (int j = 0; j < height; j++) {
	// final int retrieveIndex = retrieveIndex(i, j);
	// try {
	// final Patch patch = patchClass.newInstance();
	// patchGrid[retrieveIndex] = patch;
	// patch.setCoordinates(i, j);
	// patch.setEnvironment(this);
	// } catch (InstantiationException | IllegalAccessException e) {
	// e.printStackTrace();
	// }
	// int k = retrieveIndex*8;
	// neighborsIndexes[k]=get1DIndex(i, normalizeCoordinate(j-1,height));//TODO not
	// used and thus useless (check that)
	// neighborsIndexes[++k]=get1DIndex(i, normalizeCoordinate(j+1,height));
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width),
	// normalizeCoordinate(j-1,height));
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width), j);
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width),
	// normalizeCoordinate(j+1,height));
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width),
	// normalizeCoordinate(j-1,height));
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width), j);
	// neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width),
	// normalizeCoordinate(j+1,height));
	// }
	// }
	// }

	private int retrieveIndex(int u, int v) {
		return v * getWidth() + u;
	}

	protected Patch getPatch(int x, int y) {
		// return patchGrid[retrieveIndex(normalizeCoordinate(x, getWidth()),
		// normalizeCoordinate(y, getHeight()))];
		return gridModel.getPatch(x, y);
	}

	//
	final int normalizeCoordinate(int a, final int dimensionThickness) {
		if (gridModel.isTorusModeOn()) {
			a %= dimensionThickness;
			return a < 0 ? a + dimensionThickness : a;
		}
		if (a >= dimensionThickness)
			return dimensionThickness - 1;
		else
			return a < 0 ? 0 : a;
	}

	final double normalizeCoordinate(double a, final int dimensionThickness) {
		if (gridModel.isTorusModeOn()) {
			a %= dimensionThickness;
			return a < 0 ? a + dimensionThickness : a;
		}
		if (a >= dimensionThickness)
			return dimensionThickness - .01;
		else
			return a < 0 ? 0 : a;
	}

	/**
	 * Returns the normalized value of x, so that it is inside the environment's boundaries
	 *
	 * @param x
	 *           x-coordinate
	 * @return the normalized value
	 */
	public final double normalizeX(double x) {
		return normalizeCoordinate(x, getWidth());
	}

	/**
	 * Returns the normalized value of y, so that it is inside the environment's boundaries
	 *
	 * @param y
	 *           y-coordinate
	 * @return the normalized value
	 */
	public final double normalizeY(double y) {
		return normalizeCoordinate(y, getHeight());
	}

	/**
	 * Returns the normalized value of x, so that it is inside the environment's boundaries
	 *
	 * @param x
	 *           x-coordinate
	 * @return the normalized value
	 */
	public final double normalizeX(int x) {
		return normalizeCoordinate(x, getWidth());
	}

	/**
	 * Returns the normalized value of y, so that it is inside the environment's boundaries
	 *
	 * @param y
	 *           y-coordinate
	 * @return the normalized value
	 */
	public final double normalizeY(int y) {
		return normalizeCoordinate(y, getHeight());
	}

	/**
	 * @param xcor
	 *           absolute
	 * @param ycor
	 *           absolute
	 * @return the absolute index in a 1D data grid representing a 2D grid of width,height size. This should be used with
	 *         {@link DefaultTurtle#xcor()} and {@link DefaultTurtle#ycor()}. Its purpose is to be used on a Pheromone
	 */
	public int get1DIndex(int xcor, int ycor) {
		return normalizeCoordinate(xcor, getWidth()) + normalizeCoordinate(ycor, getHeight()) * getWidth();
	}

	protected void update() {
		if (! pheromones.isEmpty()) {
			// executePheromonesSequentialy();
			executePheromonesInParallel();
			// myDynamic
		}
	}

	/**
	 *
	 */
	public void executePheromonesSequentialy() {
		getPheromones().stream().forEach(Pheromone::diffusionAndEvaporation);
		if (CudaPlatform.isCudaAvailable()) {
			CudaEngine.cuCtxSynchronizeAll();
		}
	}

	/**
	 * Launch a turtle with predefined coordinates
	 *
	 * @param t
	 * @param x
	 * @param y
	 * @return the ID given to the turtle
	 */
	public int createTurtle(Turtle<?> t, double x, double y) {
		t.x = x;
		t.y = y;
		launchAgent(t);
		return t.getID();
	}

	protected void setTurtleCoordinate(Turtle<?> t, double x, double y) {
		t.x = x;
		t.y = y;
	}

	/**
	 * Launch a turtle with a random location
	 *
	 * @param t
	 * @return the ID given to the turtle
	 */
	public int createTurtle(Turtle<?> t) {
		return createTurtle(t, Float.MAX_VALUE, Float.MAX_VALUE);
	}

	public void executePheromonesInParallel() {
		getPheromones().parallelStream().forEach(Pheromone::diffusionAndEvaporation);
		// final Collection<Pheromone<Float>> pheromonesList = getPheromones();
		// if (!pheromonesList.isEmpty()) {
		// final ArrayList<Callable<Void>> workers = new
		// ArrayList<>(pheromonesList.size());
		// for (final Pheromone<Float> pheromone : pheromonesList) {
		// workers.add(new Callable<Void>() {
		//
		// @Override
		// public Void call() throws Exception {
		// pheromone.diffusionAndEvaporation();
		// return null;
		// }
		// });
		// }
		// try {
		// Activator.getMadkitServiceExecutor().invokeAll(workers);
		 if (CudaPlatform.isCudaAvailable() && synchronizeGPU) {
			 CudaEngine.cuCtxSynchronizeAll();
		 }
		// } catch (InterruptedException e) {
		// e.printStackTrace();
		// }
		// }
	}

	/**
	 * reset max values for rendering purposes
	 */
	protected void resetPheroMaxValues() {
		for (Pheromone<Float> phero : pheromones.values()) {
			phero.setMaximum(0f);
		}
	}

	float smell(String pheromone, int x, int y) {
		return getPheromone(pheromone).get(x, y);
	}

	void emit(String pheromone, int x, int y, float value) {
		final Pheromone<Float> p = getPheromone(pheromone);
		p.set(x, y, p.get(x, y) + value);
	}

	/**
	 * Gets the corresponding pheromone or create a new one using defaults parameters : 50% for both the evaporation rate
	 * and the diffusion rate.
	 *
	 * @param name
	 *           the pheromone's name
	 * @return the pheromone
	 */
	public Pheromone<Float> getPheromone(String name) {
		return getPheromone(name, 50, 50);
	}

	/**
	 * Gets the corresponding pheromone or create a new one using the parameters if
	 * available: The first float is the evaporation rate and the second is the
	 * diffusion rate.
	 *
	 * @param name                  the pheromone's name the first float is the
	 *                              evaporation rate and the second is the diffusion
	 *                              rate.
	 * @param evaporationPercentage the evaporation rate
	 * @param diffusionPercentage   the diffusion rate
	 * @return the pheromone created
	 */
	public Pheromone<Float> getPheromone(String name, int evaporationPercentage, int diffusionPercentage) {
		return getPheromone(name, evaporationPercentage / 100f, diffusionPercentage / 100f);
	}

	public Pheromone<Float> getCudaPheromoneWithBlock(String name, int evaporationPercentage, int diffusionPercentage) {
		Pheromone<Float> phero = pheromones.get(name);
		if (phero == null) {
			synchronized (pheromones) {
				phero = pheromones.get(name);
				if (phero == null) {
					phero = new CudaPheromoneWithBlock(name, getWidth(), getHeight(), evaporationPercentage / 100f,
							diffusionPercentage / 100f);
				}
				pheromones.put(name, phero);
			}
		}
		return phero;
	}

	/**
	 * Gets the corresponding pheromone or create a new one using the parameters if
	 * available: The first float is the evaporation rate and the second is the
	 * diffusion rate.
	 *
	 * @param name                  the pheromone's name the first float is the
	 *                              evaporation rate and the second is the diffusion
	 *                              rate.
	 * @param evaporationPercentage the evaporation rate
	 * @param diffusionPercentage   the diffusion rate
	 * @return the pheromone created
	 */
	public Pheromone<Float> getPheromone(String name, float evaporationPercentage, float diffusionPercentage) {
		return pheromones.computeIfAbsent(name, k ->{
			if (CudaPlatform.isCudaAvailable()) {
				return createCudaPheromone(name, evaporationPercentage, diffusionPercentage);
			}
			else {
				return new CPUStreamPheromone(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage,
						neighborsIndexes);
			}
		});
	}

	public Pheromone<Float> getSobelPheromone(String name, float evaporationPercentage, float diffusionPercentage) {
		Pheromone<Float> phero = pheromones.get(name);
		if (phero == null) {
			synchronized (pheromones) {
				phero = pheromones.get(name);
				if (phero == null) {
					if (CudaPlatform.isCudaAvailable()) {
						phero = new GPUSobelGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage,
								diffusionPercentage);
					}
					else {
						phero = new CPU_SobelPheromone(name, getWidth(), getHeight(), evaporationPercentage,
								diffusionPercentage, neighborsIndexes);
					}
					pheromones.put(name, phero);
				}
			}
		}
		return phero;
	}

	protected Pheromone<Float> createCudaPheromone(String name, float evaporationPercentage, float diffusionPercentage) {
		if (GPU_GRADIENTS)
			return new CudaGPUGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
		return new CudaPheromone(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
	}

	/**
	 * @param community
	 *           the group's community.
	 * @param group
	 *           the targeted group.
	 * @param role
	 *           the desired role.
	 * @return a list of Turtles currently having this role
	 */
	public <A extends Turtle<?>> List<A> getTurtlesWithRoles(final String community, final String group, final String role) {
		if (turtleProbes == null) {
			turtleProbes = new HashMap<>();
		}
		final String key = community + ";" + group + ";" + role;
		turtleProbes.computeIfAbsent(key, k -> {
			Probe p = new Probe(community, group, role);
			addProbe(p);
			return p;
		});
		return turtleProbes.get(key).getAgents();
	}

	/**
	 * Gets the turtles with this role in the default community and group
	 *
	 * @param role
	 * @return a list of turtles
	 */
	public <A extends Turtle<?>> List<A> getTurtlesWithRoles(final String role) {
		return getTurtlesWithRoles(getCommunity(), getModelGroup(), role);
	}

	class TurtlesProbe extends Probe {

		public TurtlesProbe() {
			super(TKEnvironment.this.getCommunity(), getModelGroup(), TKOrganization.TURTLE_ROLE);
		}

		protected void adding(Turtle<? extends Patch> agent) {
			addTurtleToEnvironment(agent);
		}

		protected void removing(Turtle<? extends Patch> agent) {
			removeTurtleFromEnvironment(agent);
		}
	}

	/**
	 * @return the patchGrid
	 */
	protected final Patch[] getPatchGrid() {
		return gridModel.getPatchGrid();
	}

	public TKGridModel<P> getGridModel() {
		return gridModel;
	}

	/**
	 * This offers a convenient way to create a main method that launches a simulation using the environment class under
	 * development. This call only works in the main method of the environment.
	 *
	 * @param args
	 *           MaDKit or TurtleKit options
	 * @see #executeThisAgent(int, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisEnvironment(String... args) {
		final List<String> arguments = new ArrayList<>(List.of(args));
		String className = Utils.getClassFromMainStackTrace();
		arguments.addAll(List.of("--environment", className));
		arguments.addAll(Utils.getPatchClassArgsFromClasses(TKEnvironment.class, className));
		TutleKit4.main(arguments.toArray(new String[arguments.size()]));
	}

	/**
	 * @param width
	 *           the width to set
	 */
	public void setWidth(int width) {
		super.setWidth(width);
		widthRadius = width / 2;
	}

	/**
	 * @param height
	 *           the height to set
	 */
	public void setHeight(int height) {
		super.setHeight(height);
		heightRadius = height / 2;
	}

	/**
	 * @return the envHeightRadius
	 */
	final int getHeightRadius() {
		return heightRadius;
	}

	/**
	 * @return the envWidthRadius
	 */
	final int getWidthRadius() {
		return widthRadius;
	}

	/**
	 * Keep the agents synchronized with the environment dynamics
	 *
	 * @param synchronizedEnvironment
	 */
	public void synchronizeEnvironment(boolean synchronizedEnvironment) {
		this.synchronizeGPU = synchronizedEnvironment;
	}

	protected void addTurtleToEnvironment(Turtle<?> agent) {
		agent.setID(turtleCounter.incrementAndGet());
		Patch p;
		if (agent.x == Double.MAX_VALUE || agent.y == Double.MAX_VALUE) {
			p = getPatch(prng().nextInt(getWidth()), prng().nextInt(getHeight()));
		}
		else {
			p = getPatch((int) agent.x, (int) agent.y);
		}
		agent.x = p.x;
		agent.y = p.y;
		p.addAgent(agent);
	}

	/**
	 * @param agent
	 */
	protected void removeTurtleFromEnvironment(Turtle<?> agent) {
		agent.getPatch().removeAgent(agent);
	}

	
}
