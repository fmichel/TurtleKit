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
package turtlekit.mle;

import java.math.BigDecimal;

import turtlekit.agr.TKOrganization;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.activator.DateBasedPheromoneActivator;

public class MLEScheduler extends TKScheduler {
	

	private DateBasedPheromoneActivator agents0;

	@Override
	public void onSimulationStart() {
		initializeActivators();
		super.onSimulationStart();
	}
	
	protected void initializeActivators() {
		agents0 = new DateBasedPheromoneActivator();
	}

	@Override
	public void doSimulationStep() {
		if (getSimuTimer().getCurrentTime() == BigDecimal.valueOf(100))
				Particule.MUTATION=true;
			
			getTurtleActivator(TKOrganization.TURTLE_ROLE).execute();

			getEnvironmentUpdateActivator().execute();
			
			getViewerActivator().execute();
			getSimuTimer().addOneTimeUnit();
	}

}
