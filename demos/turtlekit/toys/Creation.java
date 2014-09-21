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
import turtlekit.pvequalsnrt.Gas;
import turtlekit.viewer.PopulationCharter;
import turtlekit.viewer.TKDefaultViewer;

public class Creation extends Turtle {
	
	private Pheromone pheromone;
	private boolean launched = false;
	private Gas photon = null;

	public Creation() {
		super("create");
		setColor(Color.ORANGE);
	}
	
	@Override
	protected void activate() {
		super.activate();
		home();
	}
	
	protected String create(){
		launchAgent(new Creation());
		if(generator.nextFloat() < 0.5){
			launchAgent(new Gas(){
				@Override
				public String go() {
					if(xcor() == 0 || ycor() == 0){
						return null; //die
					}
					return super.go();
				}
				@Override
				protected void activate() {
					super.activate();
					home();
					setColor(Color.WHITE);
				}
			});
		}
		return "fly";
	}
	
	private String fly() {
		wiggle(360);
		if(getCurrentBehaviorCount() > 20000){
			return null;
		}
		return"fly";
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		executeThisTurtle(1
				,Option.viewers.toString(),MyPopulationCharter.class.getName()+";"+TKDefaultViewer.class.getName()
				,Option.envDimension.toString(),"800,800"
				,Option.startSimu.toString()
				);
	}

}

class MyPopulationCharter extends PopulationCharter{
	public MyPopulationCharter() {
		setTimeFrame(1000);
		setMonitorTurtleRole(true);
	}
}
