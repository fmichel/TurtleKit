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

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import madkit.kernel.Activator;
import madkit.simulation.SimulationException;
import turtlekit.agr.TKOrganization;

/**
 * The turtle activator : allow to make the turtles work like finite state automata.
 * 
 * @author Fabien Michel
 * @version 1.1
 * 
 */
public class TurtleActivator extends Activator<Turtle>
{     
	private final static ConcurrentHashMap<Class<? extends Turtle>,Map<String,Method>> methodTable = new ConcurrentHashMap<>();

	public TurtleActivator(String community) {
		this(community, TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE);
	}

	public TurtleActivator(String community, String role) {
		this(community, TKOrganization.TURTLES_GROUP, role);
	}

	public TurtleActivator(String community, String group, String role) {
		super(community, group, role);
//		useMulticore(Runtime.getRuntime().availableProcessors()); //TODO add the options
	}

//	@SuppressWarnings("unchecked")
//	public void initialize() {
//		for (Turtle t : getCurrentAgentsList()) {
//			methodTable.put((Class<Turtle>) t.getClass(), new HashMap<String, Method>());
//		}
//	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void adding(Turtle agent) {
		final Class<? extends Turtle> c = agent.getClass();
		if(methodTable.get(c) == null){
			methodTable.put(c, new HashMap<String,Method>());
		}
	}
	
	public void execute(final List<Turtle> agents, Object... args) {
		Collections.shuffle(agents); 
		for (Turtle t : agents) // TODO shuffle or not !!
		{
			if (t.getPatch() == null) // killed by another before its turn
				return;
			String nextMethod = null;
			final Method nextAction = t.getNextAction();
			try {
				nextMethod = (String) nextAction.invoke(t);
			} catch (NullPointerException e) {
				throw new SimulationException(t.getClass()+"'s initial behavior not set", null);
			} catch (Exception e) {
				System.err.println("Can't invoke: " + nextAction + "\n");//let the others go on running : no global exception
				e.getCause().printStackTrace();
			} catch(AssertionError e){
				throw e;
			}
			if (nextMethod != null) {
				if (nextMethod.equals(nextAction.getName())) {
					t.incrementBehaviorCount();
					continue;
				} else {
					t.setCurrentBehaviorCount(0);
				}
				t.setNextMethod(getMethodOnTurtle(t.getClass(),nextMethod));
			} else {
				if (t.isAlive()) {
					// t.setNextAction(null);
					t.killAgent(t);
				}
			}
		}
	}
	
	private static <T extends Turtle> Method getMethodOnTurtle(Class<T> agentClass, final String methodName) {
		Method m = methodTable.get(agentClass).get(methodName);
		if (m == null) {
			try {
				m = Activator.findMethodOn(agentClass, methodName);
				methodTable.get(agentClass).put(methodName, m);
			} catch (NoSuchMethodException | SecurityException e) {
				System.err.println("Can't use method: " + methodName);
				e.printStackTrace();
			}
		}
		return m;
	}

	
}









