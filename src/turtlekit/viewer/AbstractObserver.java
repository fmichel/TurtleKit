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
package turtlekit.viewer;

import madkit.kernel.Watcher;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.TurtleKit;

/**
 * An observer that plays the role of viewer in the simulation so that it is
 * automatically inserted in the simulation process.
 * 
 * @author Fabien Michel
 * @since TurtleKit 3.0.0.4
 * @version 0.1
 * 
 */
public abstract class AbstractObserver extends Watcher{
	
	@Override
	protected void activate() {
		requestRole(getCommunity(), TKOrganization.ENGINE_GROUP,TKOrganization.VIEWER_ROLE);
	}
	
	/**
	 * automatically invoked for each time step
	 */
	protected abstract void observe();

	/**
	 * shortcut for getMadkitProperty(TurtleKit.Option.community)
	 * 
	 * @return the community of the simulation this agent is in
	 */
	public final String getCommunity() {
		return getMadkitProperty(TurtleKit.Option.community);
	}
	
}
