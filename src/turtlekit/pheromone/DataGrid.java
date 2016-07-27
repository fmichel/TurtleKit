/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2016 Fabien Michel
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

package turtlekit.pheromone;

/**
 * @author Fabien Michel
 *
 * @param <T> type of data
 */
public abstract class DataGrid<T> {

	final private int width;
	final private int height;
	final private String name;
	
	
	/**
	 * @param name
	 * @param width
	 * @param height
	 * 
	 */
	public DataGrid(String name, int width, int height) {
		this.name = name;
		this.width = width;
		this.height = height;
	}

	final public T get(int x, int y) {
		return get(get1DIndex(x, y));
	}


	public abstract T get(int get1dIndex);
	

	/**
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}


	/**
	 * @return the height
	 */
	public int getHeight() {
		return height;
	}


	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	public int get1DIndex(int x, int y) {
		return y * width + x;
	}
	
	public abstract void set(int index, T value);
	
	public void set(int x, int y, T value) {
		set(get1DIndex(x, y), value);
	}

}
