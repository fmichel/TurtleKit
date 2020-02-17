package turtlekit.kernel;

import madkit.kernel.Probe;
import madkit.kernel.Scheduler;
import madkit.kernel.Watcher;
import turtlekit.agr.TKOrganization;

public class TKSimulationEngine extends Watcher {

	final private TKLauncher myLauncher;
	private TKEnvironment environment;
	private Scheduler myScheduler;

	public TKSimulationEngine(TKLauncher launcher) {
		myLauncher = launcher;
	}
	
	@Override
	protected void activate() {
		Probe<TKEnvironment> envProbe = new Probe<>(myLauncher.getCommunity(), TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE);
		addProbe(envProbe);
		environment = envProbe.getCurrentAgentsList().get(0);
		Probe<Scheduler> schedulerProbe = new Probe<>(myLauncher.getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE);
		addProbe(schedulerProbe);
		myScheduler = schedulerProbe.getCurrentAgentsList().get(0);
		removeAllProbes();
	}

	/**
	 * @return the myLauncher
	 */
	public TKLauncher getLauncher() {
		return myLauncher;
	}

	/**
	 * @return the environment
	 */
	public TKEnvironment getEnvironment() {
		return environment;
	}

	/**
	 * @return the myScheduler
	 */
	public Scheduler getScheduler() {
		return myScheduler;
	}

}
