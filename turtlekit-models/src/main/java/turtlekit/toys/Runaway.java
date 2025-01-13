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
package turtlekit.toys;

import static javafx.scene.paint.Color.BLUE;

import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.jfx.FXPheroViewer;

public class Runaway extends DefaultTurtle {
	
	private Pheromone<Float> pheromone;

	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("runaway");
		home();
		for (int i = 0; i < 2; i++) {
			launchAgent(new PheroEmmiter());
		}
		randomHeading();
		randomLocation();
		setColor(BLUE);
		pheromone = getEnvironment().getPheromone("test",30,30);
	}
	
	
	@SuppressWarnings("unused")
	private void runaway() {
		setHeading(getPheroMinDirection(pheromone));
		wiggle(30);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(10
				,"-v",FXPheroViewer.class.getName()
				,"--width","500"
				,"--height","500"
//				,Option.noCuda.toString()
				);
	}

}
