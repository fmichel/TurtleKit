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
/*
 * Prey.java -TurtleKit - A 'star logo' in MadKit
 * Copyright (C) 2000-2007 Fabien Michel
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package turtlekit.preypredator;

import java.awt.Color;
import java.util.List;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PopulationCharter;
import turtlekit.viewer.TKDefaultViewer;


@SuppressWarnings("serial")
public class Prey extends Turtle 
{

	private double life;
	private int visionRadius=1;

	@Override
	protected void activate() {
		super.activate();
//		setLogLevel(Level.ALL);
		playRole("prey");
		randomHeading();
		randomLocation();
		setColor(Color.white);
		setNextAction("live");
	}

	//a behavior
	public String live()
	{
		final List<Predator> predatorsHere = getPatch().getTurtles(Predator.class);
		if(predatorsHere.size() > 2){
			int targetedBy = 0;
			for (Predator predator : predatorsHere) {
				if(predator.getTarget() == this && ++targetedBy == 4){
					if(logger != null)
						logger.info("killed by "+predatorsHere);
					return null; //die
				}
			}
		}
		Predator turtle = getNearestTurtle(visionRadius,Predator.class);
		if(turtle != null){
			setHeading(towards(turtle)+180);//flee
		}
		
		wiggle(20);
		return "live";
	}

	public static void main(String[] args) {
		executeThisTurtle(10000
				,Option.turtles.toString(),Predator.class.getName()+",100"
//				,Option.envDimension.toString(),"20,20"
//				,Option.cuda.toString()
				,Option.startSimu.toString()
				,Option.viewers.toString(),PopulationCharter.class.getName()+";"+TKDefaultViewer.class.getName()
				);
	}

}




