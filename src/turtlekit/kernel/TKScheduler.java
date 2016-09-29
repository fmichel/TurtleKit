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

import java.util.logging.Level;

import madkit.kernel.Scheduler;
import madkit.simulation.activator.GenericBehaviorActivator;
import turtlekit.agr.TKOrganization;
import turtlekit.cuda.CudaEngine;
import turtlekit.viewer.AbstractGridViewer;

public class TKScheduler extends Scheduler {

	protected String community;
	private GenericBehaviorActivator<TKEnvironment> environmentUpdateActivator;
	GenericBehaviorActivator<AbstractGridViewer> viewerActivator;
	private TurtleActivator turtleActivator;
	private GenericBehaviorActivator<TKEnvironment> pheroMaxReset;
				
	public TKScheduler() {
//		setLogLevel(Level.ALL);
	}
	
	@Override
	protected void activate() {
		community = getMadkitProperty(TurtleKit.Option.community);
		requestRole(
				community, 
				TKOrganization.ENGINE_GROUP, 
				TKOrganization.SCHEDULER_ROLE);
//		
		pheroMaxReset = new GenericBehaviorActivator<TKEnvironment>(community, TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "resetPheroMaxValues");
		addActivator(pheroMaxReset);
		turtleActivator = new TurtleActivator(community);
		addActivator(turtleActivator);
////		turtles.setMulticore(4);
		environmentUpdateActivator = new GenericBehaviorActivator<TKEnvironment>(community, TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "update");
		addActivator(environmentUpdateActivator);
		viewerActivator = new GenericBehaviorActivator<AbstractGridViewer>(community, TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE, "observe");
		addActivator(viewerActivator);
	}
	
	@Override
	protected void end() {
		super.end();
//		if (isMadkitPropertyTrue(TurtleKit.Option.cuda)) {
//			CudaEngine.stop();
//			if (logger != null)
//				logger.fine("cuda freed");
//		}
		killAgent(environmentUpdateActivator.getCurrentAgentsList().get(0));
		System.gc();
//		pause(10000);

//		killAgent(model.getEnvironment());
//		CudaEngine.stop();
//		sendMessage(
//				LocalCommunity.NAME, 
//				Groups.SYSTEM, 
//				Organization.GROUP_MANAGER_ROLE, 
//				new KernelMessage(KernelAction.EXIT));//TODO work with AA but this is probably worthless	
	}
	
	/**
	 * @return the turtleActivator
	 */
	public TurtleActivator getTurtleActivator() {
		return turtleActivator;
	}

	/**
	 * @return the environmentUpdateActivator
	 */
	public GenericBehaviorActivator<TKEnvironment> getEnvironmentUpdateActivator() {
		return environmentUpdateActivator;
	}

	/**
	 * @return the viewerActivator
	 */
	public GenericBehaviorActivator<AbstractGridViewer> getViewerActivator() {
		return viewerActivator;
	}

	/**
	 * @return the pheroMaxReset
	 */
	public GenericBehaviorActivator<TKEnvironment> getPheroMaxReset() {
		return pheroMaxReset;
	}
}
