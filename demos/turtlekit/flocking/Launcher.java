package turtlekit.flocking;

import static turtlekit.kernel.TurtleKit.Option.cuda;
import static turtlekit.kernel.TurtleKit.Option.scheduler;

import turtlekit.kernel.TKLauncher;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.TKDefaultViewer;

/**
 * SimuLauncher manages the launch of the simulation
 *  
 * @author Emmanuel Hermellin
 * 
 * @version 0.1
 * 
 * @see turtlekit.kernel.TKLauncher;
 * 
 */

public class Launcher extends TKLauncher{
    
	/**
     * Activate the Launcher
     * @see TKLauncher#activate()
     */
	@Override
	protected void activate() {
		setMadkitProperty(cuda, "true");
		setMadkitProperty(Option.envDimension, String.valueOf(Environment.getEnvDimension())+","+String.valueOf(Environment.getEnvDimension()));
		setMadkitProperty(scheduler, SchedulerFlocking.class.getName());
		super.activate();
	}
	
	/**
     * Create the simulation instance
     */
	@Override
	protected void createSimulationInstance() {
		setMadkitProperty(Option.environment, Environment.class.getName());
		setMadkitProperty(Option.viewers, TKDefaultViewer.class.getName()
//				+";"+HeadingChecker.class.getName()
				);
//		setMadkitProperty(Option.viewers, FlockViewerColorPatch.class.getName());
		setMadkitProperty(Option.turtles, BirdFlockingUnify.class.getName()+",4000");
		setMadkitProperty(Option.fastRendering, "true");
		setMadkitProperty(Option.startSimu, "false");
		super.createSimulationInstance();
	}
	
	/**
     * Main
     */
	public static void main(String[] args) {
		executeThisLauncher();
	}

}
