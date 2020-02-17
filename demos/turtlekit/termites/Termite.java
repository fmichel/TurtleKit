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
package turtlekit.termites;

import java.awt.Color;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;

/**
 * The TK Termite version of the famous termites example.
 * @author Fabien MICHEL
 * @version
 */

@SuppressWarnings("serial")
public class Termite extends Turtle {
	/** the first time behavior of a turtle will be "searchForChip" */
	public Termite() {
		super("searchForChip");
	}

	@Override
	protected void activate() {
		super.activate();
		setColor(Color.red);
		randomHeading();
		randomLocation();
	}

	// ////////here begin the methods that are used to define the termite's
	// behavior

	/**
	 * findEmptyPatch is a one time step behavior corresponding to a list of
	 * actions. So this method will be entirely executed, sure that no other
	 * turtle of the simulation is activated. It returns a String as the behavior
	 * that the turtle will take for the next time step
	 */
	public String findEmptyPatch() {
		wiggle();
		if (getPatchColor() == Color.BLACK) {
			setPatchColor(Color.yellow);
			return ("getAway");
		}
		return SAME_BEHAVIOR;
	}

	/** another one step behavior */
	public String getAway() {
		if (getPatchColor() == Color.BLACK)
			return ("searchForChip");
		randomHeading();
		fd(20);
		return SAME_BEHAVIOR;
	}

	/** another one step behavior */
	public String searchForChip() {
		wiggle();
		if (getPatchColor() == Color.YELLOW) {
			setPatchColor(Color.BLACK);
			fd(20);
			return ("findNewPile");
		}
		return SAME_BEHAVIOR;
	}

	/** another one step behavior */
	public String findNewPile() {
		if (getPatchColor() == Color.YELLOW)
			return ("findEmptyPatch");
		wiggle();
		return SAME_BEHAVIOR;
	}

	public static void main(String[] args) {
		executeThisTurtle(1000
				,Option.envDimension.toString(),"512,512"
				,Option.viewers.toString(),	TermiteViewer.class.getName()
				,Option.startSimu.toString()

//				+";"+
//						JOGLViewer.class.getName()+";"+
//				GLViewer.class.getName()
				);
	}

}
