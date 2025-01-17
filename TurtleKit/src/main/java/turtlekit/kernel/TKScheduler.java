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

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;

import jcuda.utils.Timer;
import madkit.kernel.Agent;
import madkit.kernel.Scheduler;
import madkit.simulation.SimuOrganization;
import madkit.simulation.scheduler.MethodActivator;
import turtlekit.agr.TKOrganization;

public class TKScheduler extends Scheduler<madkit.simulation.scheduler.TickBasedTimer> {

	private Map<String, TurtleActivator> turtleActivators = new HashMap<>();
	private MethodActivator environmentUpdateActivator;
	private MethodActivator pheroMaxReset;
	private MethodActivator viewerActivator;

	public TKScheduler() {
		getLogger().setLevel(Level.ALL);
//		setPause(1000);
	}

	@Override
	protected void onActivation() {
//		getSimuTimer().setEndTime(BigDecimal.valueOf(5000));
		setPause(0);
		super.onActivation();
		initializeActivators();
		viewerActivator.execute();
	}

	@Override
	public void doSimulationStep() {
		Timer.startTimer(allTurtlesActivator());
		allTurtlesActivator().execute();
		Timer.stopTimer(allTurtlesActivator());
//		executeAndLog(getTurtleActivator());
		allTurtlesActivator().execute();
		
		Timer.startTimer(getEnvironmentUpdateActivator());
		getEnvironmentUpdateActivator().execute();
		Timer.stopTimer(getEnvironmentUpdateActivator());
		getViewerActivator().execute();
		getSimuTimer().addOneTimeUnit();
	}
	
	@Override
	protected void onLive() {
		Timer.startTimer(this);
		super.onLive();
		Timer.stopTimer(this);
		System.err.println(Timer.createPrettyString());
	}
	
	@Override
	public void onSimulationStart() {
		super.onSimulationStart();
//		allTurtlesActivator().stream().forEach(Turtle::onStart);
//		allTurtlesActivator().doIt(Turtle::onStart);
		allTurtlesActivator().doIt(Turtle::onStart);
	}
	
//	@Override
//	protected void onStart() {
//		super.onStart();
//		getEnvironment().onStart();
//	}

//	* 2. adds a turtle activator on the {@link TKOrganization.TURTLE_ROLE}, that is
	/**
	 * Initialize default TurtleKit activators.<br/>
	 * 1. adds a environment update activator<br/>
	 * on all the turtles<br/>
	 * Should be overridden for setting up customized activators for a specific
	 * simulation dynamics
	 */
	protected void initializeActivators() {
		viewerActivator = addViewersActivator();
	}

	@Override
	protected void onEnd() {
		super.onEnd();
		// if (isMadkitPropertyTrue(TurtleKit.Option.cuda)) {
		// CudaEngine.stop();
		// if (logger != null)
		// logger.fine("cuda freed");
		// }
		// killAgent(environmentUpdateActivator.getCurrentAgentsList().get(0));
		for (Agent a : getViewerActivator().getAgents()) {
			System.err.println("KILLING "+a);
			killAgent(a);
		}
//		System.gc();
//		Platform.exit();//FIXME
		// pause(10000);

		// killAgent(model.getEnvironment());
		// CudaEngine.stop();
		// sendMessage(
		// LocalCommunity.NAME,
		// Groups.SYSTEM,
		// Organization.GROUP_MANAGER_ROLE,
		// new KernelMessage(KernelAction.EXIT));//TODO work with AA but this is
		// probably worthless
	}

	/**
	 * @return the turtleActivator
	 */
	public TurtleActivator allTurtlesActivator() {
		return getTurtleActivator(TKOrganization.TURTLE_ROLE);
	}

	/**
	 * Returns a {@link TurtleActivator} for this role or creates and adds one to simulation
	 * if it not already exists.
	 * 
	 * @return the corresponding turtleActivator
	 */
	public TurtleActivator getTurtleActivator(String targetedRole) {
		return turtleActivators.computeIfAbsent(targetedRole, _ -> {
			TurtleActivator ta = new TurtleActivator(getModelGroup(), targetedRole);
			addActivator(ta);
			return ta;
		});
	}

	/**
	 * @return the environmentUpdateActivator
	 */
	public MethodActivator getEnvironmentUpdateActivator() {
		if (environmentUpdateActivator == null) {
			environmentUpdateActivator = new MethodActivator(getModelGroup(),
					SimuOrganization.ENVIRONMENT_ROLE, "update");
			addActivator(environmentUpdateActivator);
		}
		return environmentUpdateActivator;
	}

	/**
	 * @return the viewerActivator
	 */
	public MethodActivator getViewerActivator() {
//		if (viewerActivator == null) {
//			viewerActivator = new MethodActivator<AbstractGridViewer>(community, ENGINE_GROUP,
//					VIEWER_ROLE, "observe");
//			addActivator(viewerActivator);
//		}
		return viewerActivator;
	}

	/**
	 * @return the pheroMaxReset
	 */
	public MethodActivator getPheroMaxReset() {
		if (pheroMaxReset == null) {
			pheroMaxReset = new MethodActivator(SimuOrganization.MODEL_GROUP, SimuOrganization.ENVIRONMENT_ROLE,
					"resetPheroMaxValues");
			addActivator(pheroMaxReset);
		}
		return pheroMaxReset;
	}

}
