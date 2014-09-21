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

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Patch {

	final private Collection<Turtle> turtlesHere = new HashSet<>();
	private Color color;
	public int x;
	public int y;
	private TKEnvironment environment;
	private ArrayList<Patch> neighbors = new ArrayList<>();// TODO bench lazy
														// creation
	private HashMap<String, Object> marks;

	public Patch() {
		color = Color.BLACK;
		neighbors.add(this);// for including this when computing neighbors with true for inclusion
	}
	
	public void setCoordinates(int x, int y){
		this.x = x;
		this.y = y;
	}
	
	public void setEnvironment(TKEnvironment environment){
		this.environment = environment;
	}

	/**
	 * @return <code>true</code> if there is no turtle
	 */
	public boolean isEmpty() {
		return turtlesHere.isEmpty();
	}

	public List<Patch> getNeighbors(int inRadius, boolean includeThisPatch) {
		final int height = environment.getHeight();
		final int width = environment.getWidth();
		int max = Math.max(height,width);
		if(inRadius > max - 1)
			inRadius = max - 1;
		int length = 1;
		for (int i = 1; i <= inRadius; i++) {
			length += nbOfNeighborsAt(i, width, height);
		}
		if (neighbors.size() < length) {
			int startIndex = length - nbOfNeighborsAt(inRadius, width, height);
			int startRadius = inRadius;
			while (startIndex != neighbors.size()) {
				startRadius--;
				startIndex -= nbOfNeighborsAt(startRadius, width, height);
			}
			for (; startRadius <= inRadius && nbOfNeighborsAt(startRadius, width, height) > 0; startRadius++) {
				Collection<Patch> tmp = new HashSet<>();
				for (int u = -startRadius; u <= startRadius ; u++) {
					for (int v = -startRadius; v <= startRadius ; v++) {
						if (Math.abs(u) < width && Math.abs(v) < height && (Math.abs(u) == startRadius || Math.abs(v) == startRadius)) {
							tmp.add(environment.getPatch(x + u, y + v));
						}
					}
				}
				neighbors.addAll(tmp);
			}
			neighbors.trimToSize();
		}
		return neighbors.subList(includeThisPatch ? 0 : 1, length);
	}
	
    /** Drop a mark on the patch
	   
  @param markName  mark name
  @param value  mark itself, can be any java object*/
    final public void dropMark(String markName, Object value) {
        if (marks == null)
            marks = new HashMap<>(1);
        marks.put(markName,value);
    }
    
    /** get a mark deposed on the patch
         @return the corresponding java object which thus is removed
         from the patch, or <code>null</code> if not present*/
    final public Object getMark(String markName){
    	try {
			return marks.remove(markName);
		} catch (NullPointerException e) {
			return null;
		}
    }
    /** tests if the corresponding mark is present on the patch (true or false)*/
    final public boolean isMarkPresent(String markName) {
            try {
				return marks.containsKey(markName);
			} catch (NullPointerException e) {
				return false;
			}
    }
  
	
	final private int nbOfNeighborsAt(int radius, int width, int height){
		radius *= 2;//diameter
		final int aSide = radius + 1;
		int verticalNeighbors = 0;
		int offset = 0;
		if (radius <= width) {
			verticalNeighbors = Math.min(aSide, height);
			offset++;
			if (radius != width) {
				verticalNeighbors *= 2;
				offset++;
			}
		}
		int horizontalNeighbors = 0;
		if (radius <= height) {
			horizontalNeighbors = Math.min(aSide, width);
			if (radius != height) {
				horizontalNeighbors *= 2;
				offset *= 2;
			}
		}
		else {
			offset=0;
		}
		return verticalNeighbors + horizontalNeighbors - offset;
	}

	
	public <T extends Turtle> List<T> getTurtles(int inRadius, boolean includeThisPatch, Class<T> turtleType) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtles(turtleType));
		}
		return l;
	}

	public <T extends Turtle> List<T> getTurtlesWithRole(int inRadius, boolean includeThisPatch, String role, Class<T> turtleType) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtlesWithRole(role, turtleType));
		}
		return l;
	}

	public List<Turtle> getTurtlesWithRole(int inRadius, boolean includeThisPatch, String role) {
		List<Turtle> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtlesWithRole(role));
		}
		return l;
	}

	public List<Turtle> getTurtles(int inRadius, boolean includeThisPatch) {
		List<Turtle> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtles());
		}
		return l;
	}

	/**
	 * Gets the nearest turtle of type T in the vicinity of the patch.
	 * 
	 * @param inRadius the range of the search
	 * @param includeThisPatch for the search
	 * @param turtleType the type of the turtle as a {@link Class}
	 * @return the corresponding turtle or <code>null</code> if no such turtle is found
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> T getNearestTurtle(int inRadius, boolean includeThisPatch, Class<T> turtleType) {
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			for (final Turtle t : p.getTurtles()) {
				if (turtleType.isAssignableFrom(t.getClass())) {
					return (T) t;
				}
			}
		}
		return null;
	}

	/**
	 * Gets the nearest turtle of type T in the vicinity of the patch.
	 * 
	 * @param inRadius the range of the search
	 * @param includeThisPatch for the search
	 * @return the corresponding turtle or <code>null</code> if there is 
	 * no turtle around
	 * 
	 */
	public Turtle getNearestTurtle(int inRadius, boolean includeThisPatch) {
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			for (final Turtle t : p.getTurtles()) {
					return t;
				}
			}
		return null;
	}

	void installTurtles(final List<Turtle> l) {
		turtlesHere.addAll(l);
		for (final Turtle turtle : l) {
			turtle.setPatch(this);
		}
	}

	/**
	 * Get all the turtles on this patch 
	 * according to their type.
	 * 
	 * @param turtleType
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> List<T> getTurtles(Class<T> turtleType) {
		final ArrayList<T> turtles = new ArrayList<>();
		for (final Turtle t : turtlesHere) {
			if (turtleType.isAssignableFrom(t.getClass())) {
				turtles.add((T) t);
			}
		}
		return turtles;
	}

	/**
	 * Get all the turtles on this patch 
	 * according to their type and role.
	 * 
	 * @param role
	 * @param turtleType
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> List<T> getTurtlesWithRole(String role, Class<T> turtleType) {
		final List<T> turtles = new ArrayList<>();
		for (final Turtle t : turtlesHere) {
			if (turtleType.isAssignableFrom(t.getClass()) && t.isPlayingRole(role)) {
				turtles.add((T) t);
			}
		}
		return turtles;
	}

	/**
	 * Get all the turtles which are on this patch 
	 * and having this role.
	 * 
	 * @param role
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle> List<T> getTurtlesWithRole(String role) {
		final List<T> turtles = new ArrayList<>();
		for (final Turtle t : turtlesHere) {
			if (t.isPlayingRole(role)) {
				turtles.add((T) t);
			}
		}
		return turtles;
	}

	public List<Turtle> getTurtles() {
		synchronized (turtlesHere) {
			return new ArrayList<>(turtlesHere);
		}
	}

	public int countTurtles() {
		return turtlesHere.size();
	}

	public Color getColor() {
		return color;
	}

	public void setColor(Color c) {
		color = c;
	}

	final void removeAgent(Turtle a) {
		synchronized (turtlesHere) {
			turtlesHere.remove(a);
		}
	}

	final void addAgent(Turtle a) {
		synchronized (turtlesHere) {
			turtlesHere.add(a);
		}
		a.setPosition(this);
	}

	protected void update() {

	}

	@Override
	public String toString() {
		return "P(" + x + "," + y + ")";
	}

	public void dropPheromone(String name, float quantity, Float... parameters) {
		environment.getPheromone(name).incValue(x, y, quantity);
	}

}
