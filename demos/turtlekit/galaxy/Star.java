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
 * Star.java -TurtleKit - A 'star logo' in MadKit
 * Copyright (C) 2000-2013 Fabien Michel
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
 *00
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package turtlekit.galaxy;

import java.awt.Color;
import java.util.List;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;

/**
 *          this turtle turns around the nearest BlackKole. If two black holes
 *          are near it just goes forward
 * @author Fabien MICHEL
 * @version 1.3 4/1/2013
 */

@SuppressWarnings("serial")
public class Star extends Turtle {
	private int rayon;
	private Turtle myBlackHole;

	public Star() {
		super("fall");
	}

	@Override
	protected void activate() {
		super.activate();
		setColor(Color.WHITE);
		randomLocation();
		randomHeading();
		final String maxR = getMadkitProperty("maxRadius");
		int parseInt = 13;
		if (maxR != null) {
			try {
				parseInt = Integer.parseInt(maxR);
			} catch (NumberFormatException e) {
				e.printStackTrace();
			}
		}
		this.rayon = ((int) (Math.random() * parseInt)) + 1;
	}

	public String fall() {
		double dist1 = 50, dist2 = 50;
		List<Turtle> holes = getEnvironment().getTurtlesWithRoles("black hole");
		for (Turtle hole : holes) {
			final double distance = distance(hole);
			if (distance < dist1) {
				myBlackHole = hole;
				dist2 = dist1;
				dist1 = distance;
			}
		}
		if (dist2 - dist1 > 10 && dist1 < 37) {
			final double distance = distance(myBlackHole);
			if (distance > rayon + 2)
				setHeadingTowards(myBlackHole);
			else
				setHeadingTowards(myBlackHole, 90);
			if (distance > rayon)
				turnLeft(15);
			fd(1);
		} else {
			fd(2);
		}
		return "fall";
	}

	public static void main(String[] args) {
		executeThisTurtle(1500, Option.turtles.toString(),
				BlackHole.class.getName() + ",4"
				, "--maxRadius", "22"
				,Option.startSimu.toString()
				, Option.envDimension.toString(),"400,400"
				);
	}

}
