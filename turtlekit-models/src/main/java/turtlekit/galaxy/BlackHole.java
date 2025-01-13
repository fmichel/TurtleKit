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
package turtlekit.galaxy;

import javafx.scene.paint.Color;
import turtlekit.kernel.DefaultTurtle;

/**
 *       Galaxy center
 */

@SuppressWarnings("serial")
public class BlackHole extends DefaultTurtle {

	@Override
	protected void onStart() {
		super.onStart();
		playRole("black hole");
		setColor(Color.CYAN);
		changeNextBehavior("move");
	}
	
	/** the only behavior of what we shamelessly call a BlackHole */
	public void move() {
		if (getCurrentBehaviorCount() == 2) {
			wiggle(5);
			setCurrentBehaviorCount(0);
		}
	}
	
	public static void main(String[] args) {
		executeThisTurtle(5
//				,"--headless"
				,"--turtles",Star.class.getName()+",2000"
				,"--noLog"
				,"--width","600"
				,"--height","600"
				);
		// ,Option.startSimu.toString()
//				startSimu.toString());
//
	}

}
