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
import java.util.logging.Level;

import madkit.kernel.AbstractAgent;
import madkit.kernel.Probe;
import madkit.kernel.Watcher;
import madkit.simulation.probe.PropertyProbe;
import turtlekit.agr.TKOrganization;


/**
 * Obsolete : not used right now
 * 
 * @author fab
 *
 */
public class ModelProber extends Watcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3101174765431271425L;
	private String group;
	private PropertyProbe<AbstractAgent, Integer> nrj;
	private int level = 2;
	private int time = 0;

	public ModelProber(String group) {
		this(group,-1);
	}
	
	public ModelProber(String group, int level) {
		setLogLevel(Level.ALL);
		this.group = group;
		this.level = level;
	}
	
	@Override
	protected void activate() {
		for (int i = 0; i < level; i++) {
			final int lvl = i;
			addProbe(new PropertyProbe<AbstractAgent, Integer>(TKOrganization.TK_COMMUNITY, group, ""+lvl, "nrj"){
				@Override
				public String toString() {
					return "LEVEL "+lvl+" -- "+size()+" agents, max nrj = "+getMaxValue();
				}
			});
		}
		setLogLevel(Level.FINEST);
		super.activate();
		requestRole(TKOrganization.TK_COMMUNITY, group, TKOrganization.VIEWER_ROLE);
	}

	public void observe(){
		time ++;
		if (time % 50 == 0) {
			System.err.println("----------\n\n");
			for (Probe<? extends AbstractAgent> p : allProbes()) {
				if (logger != null)
					logger.info(p.toString());
			}
		}
	}

}
