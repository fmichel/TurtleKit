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

import java.awt.Color;

import turtlekit.cuda.CudaPheromone;
import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.PheromoneViewer;

public class Homogeneization extends Turtle {
	
	private Pheromone<Float> pheromone;

	public Homogeneization() {
		super("fly");
	}
	
	@Override
	protected void activate() {
		super.activate();
		home();
		randomHeading();
		randomLocation();
		setColor(new Color((int) (Math.random() * 256),(int) (Math.random() * 256), (int) (Math.random() * 256)));
		pheromone = getEnvironment().getPheromone("test",1,7);
		System.err.println(pheromone instanceof CudaPheromone);
	}
	
	
	private String fly() {
		setHeading(pheromone.getMinDirection(xcor(), ycor()));
		wiggle();
		pheromone.incValue(xcor(), ycor(), Math.random() < .5 ? 100000 : -10000);
		return "fly";
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(10
				,Option.viewers.toString(),PheromoneViewer.class.getName()
				,Option.envHeight.toString(),"512"
				,Option.envWidth.toString(),"512"
				,Option.cuda.toString()
				);
	}

}
