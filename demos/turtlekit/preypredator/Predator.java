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
package turtlekit.preypredator;

import java.awt.Color;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PopulationCharter;
import turtlekit.viewer.TKDefaultViewer;


@SuppressWarnings("serial")
public class Predator extends Turtle 
{

	private Prey target;
	private int visionRadius = 10;

	@Override
	protected void activate() {
		super.activate();
		playRole("predator");
		setNextAction("live");
		setColor(Color.red);
		randomLocation();
		randomHeading();
//		setLogLevel(Level.INFO);
	}
	
	public String live()
	{
		setTarget(towardsPrey());
		wiggle(20);
//		step();
		return "live";
	}

	/**
	 * @return 
	 * 
	 */
	public Prey towardsPrey() {
		Prey p = getNearestTurtle(getVisionRadius(), Prey.class);
		if (p != null) {
			setHeadingTowards(p);
		}
		return p;
	}

	public static void main(String[] args) {
		executeThisTurtle(500
				,Option.turtles.toString(),Prey.class.getName()+",10000"
//				,Option.cuda.toString()
				,Option.viewers.toString(),PopulationCharter.class.getName()+";"+TKDefaultViewer.class.getName()
				,Option.startSimu.toString()
//				,Option.scheduler.toString(),BenchScheduler.class.getName()
				);
	}

	public int getVisionRadius() {
		return visionRadius;
	}

	public void setVisionRadius(int visionRadius) {
		this.visionRadius = visionRadius;
	}

	public Prey getTarget() {
		if (target != null && ! target.isAlive()) {
			target = null;
		}
		return target;
	}

	public void setTarget(Prey target) {
		this.target = target;
	}
}










