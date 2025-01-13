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
import madkit.simulation.EngineAgents;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.jfx.FXPheroViewer;

@EngineAgents(viewers = {FXPheroViewer.class})
public class PheroEmmiter extends DefaultTurtle {
	
	private Pheromone<?> pheromone;

	@Override
	protected void onActivation() {
		pheromone = getEnvironment().getPheromone("test",0.2f,0.6f);
		getLogger().info(() -> pheromone.toString());
		super.onActivation();
		changeNextBehavior("fly");
		setColor(Color.color((prng().nextDouble()),prng().nextDouble(), prng().nextDouble()));
	}
	
	
	private void fly() {
		wiggle();
		pheromone.incValue(xcor(), ycor(), 100000);
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(1
				,"--width","700"
				,"--height","700"
//				,"--noLog"
				, "--cuda"
				,"--start"
				,"--tkLogLevel","ALL"
				);
	}

}
