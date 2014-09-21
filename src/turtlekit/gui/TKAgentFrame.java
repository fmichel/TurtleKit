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
package turtlekit.gui;

import javax.swing.Box;
import javax.swing.JMenuBar;

import madkit.gui.AgentFrame;
import madkit.gui.menu.AgentLogLevelMenu;
import madkit.gui.menu.HelpMenu;
import madkit.kernel.AbstractAgent;
import turtlekit.gui.menu.LaunchTKConfigurations;
import turtlekit.gui.menu.LaunchTKModels;
import turtlekit.gui.menu.LaunchTurtleMenu;
import turtlekit.gui.menu.LaunchViewerMenu;
import turtlekit.gui.menu.TKMenu;

public class TKAgentFrame extends AgentFrame {

	public TKAgentFrame(AbstractAgent agent) {
		super(agent);
	}
	
	
	
	@Override
	public JMenuBar createMenuBar() {
		JMenuBar menuBar = new JMenuBar();
		menuBar.add(new TKMenu(getAgent()));
		menuBar.add(new LaunchTKConfigurations("Config"));
		menuBar.add(new LaunchTKModels("Models"));

//		menuBar.add(new AgentMenu(agent));
		menuBar.add(new AgentLogLevelMenu(getAgent()));
		menuBar.add(new LaunchViewerMenu(getAgent()));
		menuBar.add(new LaunchTurtleMenu(getAgent()));
		menuBar.add(Box.createHorizontalGlue());
		menuBar.add(new HelpMenu());
//		menuBar.add(new AgentStatusPanel(agent));
		return menuBar;
	}
	
	
	public static AgentFrame createAgentFrame(AbstractAgent agent) {
		return new TKAgentFrame(agent);
	}

}
