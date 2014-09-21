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
package turtlekit.gui.menu;

import java.lang.reflect.Modifier;

import madkit.gui.menu.AgentClassFilter;
import madkit.gui.menu.LaunchAgentsMenu;
import madkit.kernel.AbstractAgent;
import madkit.kernel.MadkitClassLoader;
import turtlekit.kernel.Turtle;

public class LaunchTurtleMenu extends LaunchAgentsMenu {

	public LaunchTurtleMenu(final AbstractAgent agent) {
		super(agent, "Turtles", new AgentClassFilter() {
			@Override
			public boolean accept(String agentClass) {
				try {
					Class<?> c = MadkitClassLoader.getLoader().loadClass(agentClass);
					return ! Modifier.isAbstract(c.getModifiers()) && Turtle.class.isAssignableFrom(c);
				} catch (ClassNotFoundException e) {
					e.printStackTrace();
				}
				return false;

			}
		});
	}
}
