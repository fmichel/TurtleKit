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

import static turtlekit.agr.TKOrganization.TURTLE_ROLE;

import java.util.HashMap;
import java.util.Map;

import jcuda.utils.Timer;
import madkit.kernel.AbstractAgent;
import madkit.kernel.Scheduler;
import madkit.simulation.activator.GenericBehaviorActivator;
import turtlekit.agr.TKOrganization;
import turtlekit.viewer.AbstractGridViewer;

public class TKScheduler extends Scheduler {

    private Map<String, TurtleActivator> turtleActivators = new HashMap<>();
    protected String community;
    private GenericBehaviorActivator<TKEnvironment> environmentUpdateActivator;
    private GenericBehaviorActivator<AbstractGridViewer> viewerActivator;
    private TurtleActivator turtleActivator;
    private GenericBehaviorActivator<TKEnvironment> pheroMaxReset;

    public TKScheduler() {
	// setLogLevel(Level.ALL);
    }

    @Override
    protected void activate() {
	community = getMadkitProperty(TurtleKit.Option.community);
	requestRole(community, TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE);
	initializeActivators();
	//
	// turtleActivator = new TurtleActivator(community);
	// addActivator(turtleActivator);
	//// turtles.setMulticore(4);
    }

    @Override
    public void doSimulationStep() {
	Timer.startTimer(getTurtleActivator());
	getTurtleActivator().execute();
	Timer.stopTimer(getTurtleActivator());

	Timer.startTimer(getEnvironmentUpdateActivator());
	getEnvironmentUpdateActivator().execute();
	Timer.stopTimer(getEnvironmentUpdateActivator());
	getViewerActivator().execute();
	setGVT(getGVT() + 1);
    }

    /**
     * Initialize default TurtleKit activators.</br>
     * 1. adds a environment update activator</br>
     * 2. adds a turtle activator on the {@link TKOrganization.TURTLE_ROLE}, that is on all the turtles</br>
     * Should be overridden for setting up customized activators for a specific simulation dynamics
     */
    protected void initializeActivators() {
	getTurtleActivator();
	viewerActivator = new GenericBehaviorActivator<AbstractGridViewer>(community, TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE, "observe");
	addActivator(viewerActivator);
    }

    @Override
    protected void end() {
	super.end();
	// if (isMadkitPropertyTrue(TurtleKit.Option.cuda)) {
	// CudaEngine.stop();
	// if (logger != null)
	// logger.fine("cuda freed");
	// }
	// killAgent(environmentUpdateActivator.getCurrentAgentsList().get(0));
	for (AbstractAgent a : getViewerActivator().getCurrentAgentsList()) {
	    killAgent(a);
	}
	System.gc();
	// pause(10000);

	// killAgent(model.getEnvironment());
	// CudaEngine.stop();
	// sendMessage(
	// LocalCommunity.NAME,
	// Groups.SYSTEM,
	// Organization.GROUP_MANAGER_ROLE,
	// new KernelMessage(KernelAction.EXIT));//TODO work with AA but this is probably worthless
    }

    /**
     * @return the turtleActivator
     */
    public TurtleActivator getTurtleActivator() {
	return getTurtleActivator(TURTLE_ROLE);
    }

    /**
     * Returns a turtleActivator for this role or creates and adds one to simulation if it not already exists.
     * 
     * @return the corresponding turtleActivator
     */
    public TurtleActivator getTurtleActivator(String targetedRole) {
	return turtleActivators.computeIfAbsent(targetedRole, k -> {
	    TurtleActivator ta = new TurtleActivator(community, TKOrganization.TURTLES_GROUP, targetedRole);
	    addActivator(ta);
	    return ta;
	});
    }

    /**
     * @return the environmentUpdateActivator
     */
    public GenericBehaviorActivator<TKEnvironment> getEnvironmentUpdateActivator() {
	if (environmentUpdateActivator == null) {
	    environmentUpdateActivator = new GenericBehaviorActivator<TKEnvironment>(community, TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "update");
	    addActivator(environmentUpdateActivator);
	}
	return environmentUpdateActivator;
    }

    /**
     * @return the viewerActivator
     */
    public GenericBehaviorActivator<AbstractGridViewer> getViewerActivator() {
	if (viewerActivator == null) {
	    viewerActivator = new GenericBehaviorActivator<AbstractGridViewer>(community, TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE, "observe");
	    addActivator(viewerActivator);
	}
	return viewerActivator;
    }

    /**
     * @return the pheroMaxReset
     */
    public GenericBehaviorActivator<TKEnvironment> getPheroMaxReset() {
	if (pheroMaxReset == null) {
	    pheroMaxReset = new GenericBehaviorActivator<TKEnvironment>(community, TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "resetPheroMaxValues");
	    addActivator(pheroMaxReset);
	}
	return pheroMaxReset;
    }
}
