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
/*
 * Copyright 1997-2011 Fabien Michel, Olivier Gutknecht, Jacques Ferber
 *
 * This file is part of MadKit.
 *
 * MadKit is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MadKit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MadKit. If not, see <http://www.gnu.org/licenses/>.
 */
package turtlekit.kernel.activator;

import java.util.List;

import madkit.simulation.scheduler.DateBasedDiscreteEventActivator;
import turtlekit.kernel.Turtle;

/**
 * The turtle activator : allow to make the turtles work like finite state
 * automata.
 *
 * @author Fabien Michel
 * @version 0.9
 *
 */
public class TurtleDiscreteEventActivator extends DateBasedDiscreteEventActivator {
	public TurtleDiscreteEventActivator(String community, String group, String role) {
		super(group, role, null);
	}

	@Override
	public void execute(Object... args) {
		List<Turtle<?>> turtles = getAgents();
		for (Turtle<?> t : turtles) {
			t.executeBehavior();
		}
		setNextActivationDate(getNextActivationDate().plus(getDefaultInterval()));
	}
}