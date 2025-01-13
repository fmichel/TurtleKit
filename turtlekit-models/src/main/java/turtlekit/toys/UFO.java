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

import javafx.scene.paint.Color;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.jfx.FXPheroViewer;

public class UFO extends DefaultTurtle {

	private Pheromone<?> pheromone;
	private Pheromone<?> pheromone2;

	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("fly");
		home();
		setColor(new Color(prng().nextDouble(), (prng().nextDouble()), (prng().nextDouble()), 1));
		pheromone = getEnvironment().getPheromone("test", 30, 30);
		pheromone2 = getEnvironment().getPheromone("other", 30, 60);
	}

	private void fly() {
		turnRight(2);
		fd(1);
		if (prng().nextDouble() < .5) {
			pheromone.incValue(xcor(), ycor(), 1000);
		} else {
			pheromone2.incValue(xcor(), ycor(), 1000);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(1000 
				,"-v",FXPheroViewer.class.getName()
				,"--width","150"
				,"--height","150"
//				,"--cuda"
				);
	}

}
