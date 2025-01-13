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
package turtlekit.kernel;

import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

import madkit.kernel.Activator;
import turtlekit.agr.TKOrganization;

/**
 * The default turtle activator: trigger the behavior of all turtles having a
 * specific role
 *
 * @author Fabien Michel
 * @version 4
 *
 */
public class TurtleActivator extends Activator {

	public TurtleActivator() {
		this(TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE);
	}

	public TurtleActivator(String role) {
		this(TKOrganization.TURTLES_GROUP, role);
	}

	public TurtleActivator(String group, String role) {
		super(group, role);
	}

	@Override
	public String toString() {
		return getClass().getSimpleName() + "<" + getRole() + "-> " + size() + " agents";
	}

	@Override
	public void execute(Object... args) {
		stream().forEach(Turtle::executeBehavior);
	}
	
	/**
	 * for automatic casting
	 */
	@SuppressWarnings("unchecked")
	@Override
	public List<Turtle<?>> getAgents() {
		return super.getAgents();
	}
	
	public Stream<Turtle<?>> stream(){
		return getAgents().stream();
	}
	
	public <T extends Turtle<?>> void doIt(Consumer<? super Turtle<?>> c) {
		stream().forEach(c);
	}

}
