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
package turtlekit.pvequalsnrt;

import static turtlekit.kernel.TurtleKit.Option.startSimu;

import java.awt.Color;

import turtlekit.kernel.Patch;
import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;

/**
 * The turtle "gas" just needs space !!!
 * 
 * @author Fabien Michel
 */

@SuppressWarnings("serial")
public class Gas extends Turtle {

    public Gas() {
	super("go");
    }

    /**
     * 
     */
    protected void activate() {
	super.activate();
	setColor(Color.cyan);
	randomHeading();
	final String wallValue = getMadkitProperty("wallX");
	if (wallValue != null) {
	    int wallX = Integer.parseInt(wallValue);
	    moveTo(generator.nextInt(wallX - 1) + 1, generator.nextInt(getWorldHeight() - 2) + 1);
	}
    }

    /**
     * The gas looks for free space (without an other turtle or a wall) but can't go through the wall (white color
     * patches) and rebounds against the sides of the world
     */
    public void go() {
	final Patch nextPatch = getNextPatch();
	if (nextPatch != null) {
	    if (!nextPatch.isEmpty()) {
		randomHeading(180);
	    }
	    else if (nextPatch.getColor() == Color.WHITE) {
		setHeading(getHeading() + 100);
		randomHeading(40);
	    }
	    else {
		step();
	    }
	}
	else {
	    step();
	}
    }

    public static void main(String[] args) {
	executeThisTurtle(5000, Option.envDimension.toString(), "200,50", Option.noWrap.toString(), "--wallX", "10", startSimu.toString(), Option.viewers.toString(),
		GasViewer.class.getName() + ";" + PhysicsChecker.class.getName());

    }

}
