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

import java.util.List;

import javafx.scene.paint.Color;
import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.DefaultTurtle;

/**
 * this turtle turns around the nearest BlackKole. If two black holes are near
 * it just goes forward
 * 
 * @author Fabien MICHEL
 * @version 1.3 4/1/2013
 */

@SuppressWarnings("serial")
public class Star extends DefaultTurtle {

	
	@SliderProperty(minValue = 2,maxValue = 100, scrollPrecision = 0.01)
	@UIProperty
	private static double maxRange = 30;
	private int rayon;
	private double maxRangeCache;
	private DefaultTurtle myBlackHole;

	@Override
	protected void onStart() {
		super.onStart();
		changeNextBehavior("fall");
		setColor(Color.WHITE);
	}
	
	public void fall() {
		updateRayon();
		double dist1 = 50;
		double  dist2 = 50;
		List<DefaultTurtle> holes = getEnvironment().getTurtlesWithRoles("black hole");
		for (DefaultTurtle hole : holes) {
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
	}

	private void updateRayon() {
		if(maxRangeCache != maxRange) {
			this.rayon = prng().nextInt(1, (int) getMaxRange());
			maxRangeCache = maxRange;
		}
	}
	
	/**
	 * @return the maxRange
	 */
	public static double getMaxRange() {
		return maxRange;
	}

	/**
	 * @param maxRange the maxRange to set
	 */
	public static void setMaxRange(double maxRange) {
		Star.maxRange = maxRange;
	}

}
