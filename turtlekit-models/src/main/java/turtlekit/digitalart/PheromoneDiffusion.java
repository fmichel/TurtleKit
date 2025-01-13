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
package turtlekit.digitalart;

import turtlekit.viewer.jfx.FXPheroViewer;

public class PheromoneDiffusion extends FXPheroViewer {

	@Override
	protected void onActivation() {
		super.onActivation();
		getEnvironment().getPheromone("red", 2, 80);
		setSelectedPheromone("red");
		getSelectedPheromone().incValue(getWidth()/2-30, getHeight()/2-30, 90000000000L);
		getSelectedPheromone().incValue(getWidth()/2+30, getHeight()/2+30, 90000000000L);
		getSelectedPheromone().incValue(getWidth()/2-100, getHeight()/2-100, 10000000000L);
		getSelectedPheromone().incValue(getWidth()/2+100, getHeight()/2+100, 90000000000L);
		getSelectedPheromone().incValue(getWidth()/2-100, getHeight()/2+100, 100000000000L);
		getSelectedPheromone().incValue(getWidth()/2+240, getHeight()/2-200, 100000000000L);
	}

	public static void main(String[] args) {
		executeThisViewer(
				"--width","500"
				,"--height","500"
				,"--cuda"
				);
	}
}
