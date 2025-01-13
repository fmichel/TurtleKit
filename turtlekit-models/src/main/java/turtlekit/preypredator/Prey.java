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

import static javafx.scene.paint.Color.WHITE;

import java.util.List;

import turtlekit.kernel.DefaultTurtle;
import turtlekit.viewer.TKViewer;
import turtlekit.viewer.jfx.TurtlePopulationChartDrawer;

public class Prey extends DefaultTurtle {

	private int visionRadius = 1;

	@Override
	protected void onActivation() {
//		 setLogLevel(Level.ALL);
		super.onActivation();
		playRole("prey");
	}
	
	@Override
	protected void onStart() {
		super.onStart();
		setColor(WHITE);
		changeNextBehavior("doIt");
	}

	// a behavior
	private void doIt() {
		final List<Predator> predatorsHere = getPatch().getTurtles(Predator.class);
		if (predatorsHere.size() > 2) {
			int targetedBy = 0;
			for (Predator predator : predatorsHere) {
				if (predator.getTarget() == this && ++targetedBy == 4) {
					return;
				}
			}
		}
		Predator turtle = getNearestTurtle(visionRadius, Predator.class);
		if (turtle != null) {
			setHeading(towards(turtle) + 180);// flee
		}

		wiggle(20);
	}

	public static void main(String[] args) {
		executeThisTurtle(1000
				,"--turtles",Predator.class.getName()+",20000"
				,"--width","500"
				,"--height","500"
				,"-v",TurtlePopulationChartDrawer.class.getName()
				,"-v",TKViewer.class.getName()
				);
	}

}
