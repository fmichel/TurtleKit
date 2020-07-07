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

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.jfx.JfxPheroViewer;

public class PheroEmmiter extends Turtle {
	
	private Pheromone<?> pheromone;

	@Override
	protected void activate() {
		super.activate();
		changeNextBehavior("fly");
		randomLocation();
		randomHeading();
		setColor(new Color((int) (Math.random() * 256),(int) (Math.random() * 256), (int) (Math.random() * 256)));
//		pheromone = getEnvironment().getPheromone("test",0.333f,0.3f);
		pheromone = getEnvironment().getPheromone("test",20,60);
//		System.err.println(pheromone instanceof CudaPheromone);
	}
	
	
	private String fly() {
		wiggle();
		pheromone.incValue(xcor(), ycor(), 100000);
		return "fly";
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(100
//				,Option.viewers.toString(),PheromoneViewer.class.getName()
				,Option.envHeight.toString(),"512"
				,Option.envWidth.toString(),"512"
				,Option.viewers.toString(),JfxPheroViewer.class.getName()
//				,Option.viewers.toString(),JFXViewer.class.getName(),BooleanOption.JavaFX.toString()
				,Option.cuda.toString()
				,Option.startSimu.toString()
				);
	}

}
