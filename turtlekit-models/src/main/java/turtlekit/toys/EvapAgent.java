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

import madkit.gui.UIProperty;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;

public class EvapAgent extends DefaultTurtle {

	@UIProperty
	private static int nbOfAgents = 2;

	private Pheromone<Float> pheromone;

	@Override
	protected void onActivation() {
		super.onActivation();
		setColor(BLUE);
		pheromone = getEnvironment().getPheromone("test", .03f, 0.29f);
		changeNextBehavior("behavior");
	}
	
	@Override
	protected void onStart() {
	}
	
	protected void behavior() {
		pheromone.incValue(xcor(), ycor(), 10000);
		towardsMinGradientField(pheromone);
		fd(1);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(getNbOfAgents()
//				, "--cuda"
				);
	}

	public static int getNbOfAgents() {
		return nbOfAgents;
	}

	public static void setNbOfAgents(int nbOfAgents) {
		EvapAgent.nbOfAgents = nbOfAgents;
	}

}
