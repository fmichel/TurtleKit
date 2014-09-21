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

import static turtlekit.kernel.TurtleKit.Option.patch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import madkit.kernel.Activator;
import madkit.kernel.Madkit;
import madkit.kernel.MadkitClassLoader;
import madkit.kernel.Probe;
import madkit.kernel.Watcher;
import madkit.simulation.probe.PropertyProbe;
import turtlekit.agr.TKOrganization;
import turtlekit.cuda.CudaEngine;
import turtlekit.cuda.CudaGPUGradientsPhero;
import turtlekit.cuda.CudaPheromone;
import turtlekit.pheromone.JavaPheromone;
import turtlekit.pheromone.Pheromone;

public class TKEnvironment extends Watcher {

	private String community;
	private int width;
	private int height;
	private Map<String, Pheromone> pheromones;
	private Map<String,Probe<Turtle>> turtleProbes;
	private Patch[] patchGrid;
	boolean wrapMode;
	private boolean cudaOn;
	private int heightRadius;
	private int widthRadius;
	private final static transient AtomicInteger	turtleCounter = new AtomicInteger(0);
	private int[] neighborsIndexes;//TODO or remove
	protected boolean GPU_GRADIENTS = false;
	private boolean synchronizeGPU = true;

	public TKEnvironment(){
		pheromones = new TreeMap<String, Pheromone>();
	}
	
	public Collection<Pheromone> getPheromones(){
		return pheromones.values();
	}

	@Override
	protected void activate() {
		GPU_GRADIENTS = Boolean.parseBoolean(getMadkitProperty("GPU_gradients"));
//		setLogLevel(Level.ALL);
		community = getMadkitProperty(TurtleKit.Option.community);
		setWidth(Integer.parseInt(getMadkitProperty(TurtleKit.Option.envWidth)));
		setHeight(Integer.parseInt(getMadkitProperty(TurtleKit.Option.envHeight)));
		wrapMode = ! isMadkitPropertyTrue(TurtleKit.Option.noWrap);
		cudaOn = isMadkitPropertyTrue(TurtleKit.Option.cuda);
		if(logger != null){
			logger.info("----------------------CUDA ON "+isCudaOn());
			logger.info("----------------------GPU_GRADIENTS "+GPU_GRADIENTS);
		}
		if(cudaOn){
//			Turtle.generator = new GPU_PRNG(12134); //TODO
		}
		initPatchGrid();
		// 1 : request my role so that the viewer can probe me 
		requestRole(community, TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE);
		
		// 2 : this probe is used to initialize the agents' environment field
		addProbe(new AgentsProbe());
	}

	private void initPatchGrid() {
		patchGrid = new Patch[width * height];
		neighborsIndexes = new int[width * height * 8];
		Class<? extends Patch> patchClass;
		try {
			patchClass = (Class<? extends Patch>) MadkitClassLoader.getLoader().loadClass(getMadkitProperty(patch));
		} catch (ClassNotFoundException e) {
			patchClass = Patch.class;
			getLogger().severe(e.toString()+" -> using "+patchClass.getName());
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				final int retrieveIndex = retrieveIndex(i, j);
				try {
					final Patch patch = patchClass.newInstance();
					patchGrid[retrieveIndex] = patch;
					patch.setCoordinates(i, j);
					patch.setEnvironment(this);
				} catch (InstantiationException | IllegalAccessException e) {
					e.printStackTrace();
				}
				int k = retrieveIndex*8;
				neighborsIndexes[k]=get1DIndex(i, normalizeCoordinate(j-1,height));//TODO not used and thus useless (check that)
				neighborsIndexes[++k]=get1DIndex(i, normalizeCoordinate(j+1,height));
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width), normalizeCoordinate(j-1,height));
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width), j);
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i-1,width), normalizeCoordinate(j+1,height));
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width), normalizeCoordinate(j-1,height));
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width), j);
				neighborsIndexes[++k]=get1DIndex(normalizeCoordinate(i+1,width), normalizeCoordinate(j+1,height));
			}
		}
	}

	private int retrieveIndex(int u, int v) {
		return v * width + u;
	}
	
	protected Patch getPatch(int i, int j) {
		return patchGrid[retrieveIndex(normalizeCoordinate(i, width), normalizeCoordinate(j, height))];
	}
	

	final int normalizeCoordinate(int a, final int dimensionThickness) {
		if (wrapMode) {
			a %= dimensionThickness;
			return a < 0 ? a + dimensionThickness : a;
		} 
		if (a >= dimensionThickness)
			return dimensionThickness - 1;
		else
			return a < 0 ? 0 : a;
	}
	
	final double normalizeCoordinate(double a, final int dimensionThickness) {
		if (wrapMode) {
			a %= dimensionThickness;
			return a < 0 ? a + dimensionThickness : a;
		}
		if (a >= dimensionThickness)
			return dimensionThickness - .01;
		else
			return a < 0 ? 0 : a;
	}
	
	/**
	 * Returns the normalized value of x, so that 
	 * it is inside the environment's boundaries
	 * 
	 * @param x x-coordinate
	 * @return the normalized value
	 */
	final public double normalizeX(double x){
		return normalizeCoordinate(x, width);
	}
	
	/**
	 * Returns the normalized value of y, so that 
	 * it is inside the environment's boundaries
	 * 
	 * @param y y-coordinate
	 * @return the normalized value
	 */
	final public double normalizeY(double y){
		return normalizeCoordinate(y, height);
	}
	
	/**
	 * Returns the normalized value of x, so that 
	 * it is inside the environment's boundaries
	 * 
	 * @param x x-coordinate
	 * @return the normalized value
	 */
	final public double normalizeX(int x){
		return normalizeCoordinate(x, width);
	}
	
	/**
	 * Returns the normalized value of y, so that 
	 * it is inside the environment's boundaries
	 * 
	 * @param y y-coordinate
	 * @return the normalized value
	 */
	final public double normalizeY(int y){
		return normalizeCoordinate(y, height);
	}
	
	/**
	 * @param x absolute 
	 * @param y absolute 
	 * @return the absolute index in a 1D data grid representing a 2D grid of
	 *         width,height size. This should be used with {@link Turtle#xcor()} and
	 *         {@link Turtle#ycor()}. 
	 *         Its purpose is to be used on a
	 *         Pheromone
	 */
	public int get1DIndex(int xcor, int ycor){
		return normalizeCoordinate(xcor, width) + normalizeCoordinate(ycor, height) * width;
	}
	
	protected void update(){
//			executePheromonesSequentialy();
			executePheromonesInParallel();
			//myDynamic
	}

	/**
	 * 
	 */
	private void executePheromonesSequentialy() {
		for (Pheromone pheromone : pheromones.values()) {
//			pheromone.set((int) (Math.random()*200), (int) (Math.random()*200), 1E38f); //TODO nice art
			pheromone.diffusionAndEvaporation();
		}
		if (cudaOn) {
			CudaEngine.cuCtxSynchronizeAll();
		}
	}
	
	private void executePheromonesInParallel() {
		final Collection<Pheromone> pheromonesList = getPheromones();
		if (! pheromonesList.isEmpty()) {
			final ArrayList<Callable<Void>> workers = new ArrayList<>(
					pheromonesList.size());
			for (final Pheromone pheromone : pheromonesList) {
				workers.add(new Callable<Void>() {
					@Override
					public Void call() throws Exception {
						pheromone.diffusionAndEvaporation();
						return null;
					}
				});
			}
			try {
				Activator.getMadkitServiceExecutor().invokeAll(workers);
				if (isCudaOn() && synchronizeGPU) {
					CudaEngine.cuCtxSynchronizeAll();
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	
	/**
	 * reset max values for rendering purposes
	 */
	protected void resetPheroMaxValues(){
		for (Pheromone phero : pheromones.values()) {
			phero.setMaximum(0);
		}
	}
	

	float smell(String pheromone, int x, int y) {
		return getPheromone(pheromone).get(x, y);
	}

	void emit(String pheromone, int x, int y, float value) {
		getPheromone(pheromone).incValue(x, y, value);
	}
	
	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}


	/**
	 * Gets the corresponding pheromone or create a new one using defaults
	 * parameters : 50% for both the evaporation rate and 
	 * the diffusion rate.
	 * @param name the pheromone's name
	 * @return the pheromone
	 */
	public Pheromone getPheromone(String name) {
		return getPheromone(name,50,50);
	}
	
	/**
	 * Gets the corresponding pheromone or create a new one using the 
	 * parameters if available: The first float is the evaporation rate and the second
	 * is the diffusion rate.
	 * 
	 * @param name the pheromone's name
	 * @param parameters the first float is the evaporation rate and the second
	 * is the diffusion rate.
	 * @return the pheromone
	 */
	public Pheromone getPheromone(String name, int evaporationPercentage, int diffusionPercentage) {
		return getPheromone(name, evaporationPercentage / 100f, diffusionPercentage / 100f);
	}
	
	/**
	 * Gets the corresponding pheromone or create a new one using the 
	 * parameters if available: The first float is the evaporation rate and the second
	 * is the diffusion rate.
	 * 
	 * @param name the pheromone's name
	 * @param parameters the first float is the evaporation rate and the second
	 * is the diffusion rate.
	 * @return the pheromone
	 */
	public Pheromone getPheromone(String name, float evaporationPercentage, float diffusionPercentage) {
		Pheromone phero = pheromones.get(name);
		if (phero == null) {
			synchronized (pheromones) {
				phero = pheromones.get(name);
				if (phero == null) {
					if (cudaOn && CudaEngine.isCudaAvailable() ) {
						phero = createCudaPheromone(name, evaporationPercentage, diffusionPercentage);
					}
					else{
						phero = new JavaPheromone(name, getWidth(),	getHeight(), evaporationPercentage, diffusionPercentage, neighborsIndexes);
					}
					pheromones.put(name, phero);
				}
			}
		}
		return phero;
	}
	
	protected Pheromone createCudaPheromone(String name, float evaporationPercentage, float diffusionPercentage){
		if(GPU_GRADIENTS)
			return new CudaGPUGradientsPhero(name, getWidth(), getHeight(), evaporationPercentage, diffusionPercentage);
		return new CudaPheromone(name, getWidth(),	getHeight(), evaporationPercentage, diffusionPercentage);
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
	public List<Turtle> getTurtlesWithRoles(final String community, final String group, final String role){
		if(turtleProbes == null){
			turtleProbes = new HashMap<>();
		}
		final String key = community+";"+group+";"+role;
		Probe<Turtle> probe = turtleProbes.get(key);
		if(probe == null){
			turtleProbes.put(key, probe = new Probe<Turtle>(community,group,role));
			addProbe(probe);
		}
		return probe.getCurrentAgentsList();
	}

	/**
	 * Gets the turtles with this role in the default community and group
	 * 
	 * @param role
	 * @return a list of turtles
	 */
	public List<Turtle> getTurtlesWithRoles(final String role){
		return getTurtlesWithRoles(community, TKOrganization.TURTLES_GROUP, role);
	}


	class AgentsProbe extends PropertyProbe<Turtle, TKEnvironment>{
		
		public AgentsProbe() {
			super(community, TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE, "environment");
		}
		
//		@Override
//		protected void adding(List<Turtle> agents) {
//			Turtle agentTest = agents.get(0);
//			Patch p;
//			if(agentTest.x == Double.MAX_VALUE || agentTest.y == Double.MAX_VALUE){
//				p = getPatch(width /2, height / 2);
//				for (Turtle turtle : agents) {
//					turtle.x = p.i;
//					turtle.y = p.j;
//				}
//			}
//			else{
//				p = getPatch((int) agentTest.x, (int) agentTest.y);
//			}
//			p.installTurtles(agents);
//			for (Turtle turtle : agents) {
//				turtle.setID(turtleCounter.incrementAndGet());
//				setPropertyValue(turtle, Environment.this);
//			}
//		}
//		
		protected void adding(Turtle agent) {
			agent.setID(turtleCounter.incrementAndGet());
			setPropertyValue(agent, TKEnvironment.this);
			Patch p;
			if(agent.x == Double.MAX_VALUE || agent.y == Double.MAX_VALUE){
//				int xcor = Turtle.generator.nextInt(getWidth());
//				int ycor = Turtle.generator.nextInt(getHeight());
				p = getPatch(Turtle.generator.nextInt(getWidth()), Turtle.generator.nextInt(getHeight()));
				agent.x = p.x;
				agent.y = p.y;
			}
			else{
				p = getPatch((int) agent.x, (int) agent.y);
			}
			p.addAgent(agent);
//			patchGrid[0].addAgent(agent);
		}
		protected void removing(Turtle agent) {
			agent.getPatch().removeAgent(agent);
//			agent.setPatch(null);
		}
	}



	/**
	 * @return the patchGrid
	 */
	protected final Patch[] getPatchGrid() {
		return patchGrid;
	}

	/**
	 * This offers a convenient way to create a main method 
	 * that launches a simulation using the environment
	 * class under development. 
	 * This call only works in the main method of the environment.
	 * 
	 * @param args
	 *           MaDKit or TurtleKit options
	 * @see #executeThisAgent(int, boolean, String...)
	 * @since TurtleKit 3.0.0.1
	 */
	protected static void executeThisEnvironment(String... args) {
		StackTraceElement element = null;
		for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
			if(stackTraceElement.getMethodName().equals("main")){
				element  = stackTraceElement;
				break;
			}
		}
		final ArrayList<String> arguments = new ArrayList<>(Arrays.asList(
				Madkit.BooleanOption.desktop.toString(),"false",
				Madkit.Option.configFile.toString(), "turtlekit/kernel/turtlekit.properties",
				TurtleKit.Option.environment.toString(), element.getClassName()));
		if (args != null) {
			arguments.addAll(Arrays.asList(args));
		}
		new Madkit(arguments.toArray(new String[0]));
	}

	/**
	 * @return the cudaOn
	 */
	public boolean isCudaOn() {
		return cudaOn;
	}

	/**
	 * @param width the width to set
	 */
	final void setWidth(int width) {
		this.width = width;
		widthRadius = width / 2;
	}

	/**
	 * @param height the height to set
	 */
	final void setHeight(int height) {
		this.height = height;
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
		this.synchronizeGPU= synchronizedEnvironment;
	}


}
