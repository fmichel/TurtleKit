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
package turtlekit.viewer.jfx;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import madkit.agr.SystemRoles;
import madkit.simulation.SimuOrganization;
import madkit.simulation.viewer.RolesPopulationLineChartDrawer;
import turtlekit.agr.TKOrganization;

/**
 * 
 * Creates a chart tracing the population for each role taken by the turtles in the Simulation. The "turtle" role is
 * ignored by default.
 * 
 * @author Fabien Michel
 *
 */
public class TurtlePopulationChartDrawer extends RolesPopulationLineChartDrawer {

	private Set<String> handledRoles = new HashSet<>();
	private Set<String> ignoredRoles = new HashSet<>();

	@Override
	protected void onActivation() {
		super.onActivation();
		ignoredRoles.add(SystemRoles.GROUP_MANAGER_ROLE);
		ignoredRoles.add(SimuOrganization.ENVIRONMENT_ROLE);
		ignoredRoles.add(TKOrganization.TURTLE_ROLE);
		for (String role : handledRoles) {
			addRoleToMonitoring(getModelGroup(), role);
		}
	}

	@Override
	public void display() {
		refreshMonitoredRoles();
		super.display();
	}

	/**
	 * Refresh the monitored roles by adding new roles
	 */
	private void refreshMonitoredRoles() {
		List<String> roles = new ArrayList<>(getOrganization().getGroup(getCommunity(), getModelGroup()).getRoleNames());
		roles.removeAll(ignoredRoles);
		roles.removeAll(handledRoles);
		for (String role : roles) {// new roles
			if (handledRoles.add(role)) {
				addRoleToMonitoring(getModelGroup(), role);
			}
		}
	}

	public void render() {
		// rendering is done in display by modifying the data of the chart
	}
}
