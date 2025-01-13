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

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.viewer.jfx.TurtlePopulationChartDrawer;

public class Predator extends DefaultTurtle {

	private Prey target;
	@SliderProperty(minValue = 1, maxValue = 11, scrollPrecision = 1.0)
	@UIProperty(category = "Agent", displayName = "vision radius")
	static double visionRadius = 10.0;

	@Override
	protected void onActivation() {
		super.onActivation();
		playRole("predator");
		changeNextBehavior("doIt");
	}

	private void doIt() {
		setTarget(towardsPrey());
		wiggle(20);
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

	public Prey getTarget() {
		if (target != null && !target.isAlive()) {
			target = null;
		}
		return target;
	}

	public void setTarget(Prey target) {
		this.target = target;
	}

	public static void main(String[] args) {
			executeThisTurtle(2000
					,"--turtles",Prey.class.getName()+",2000"
					,"-v",TurtlePopulationChartDrawer.class.getName()
					,"--width","500"
					,"--height","500"
			);
		}

	/**
	 * @return the visionRadius
	 */
	public static double getVisionRadius() {
		return visionRadius;
	}

	/**
	 * @param visionRadius the visionRadius to set
	 */
	public static void setVisionRadius(double visionRadius) {
		Predator.visionRadius = visionRadius;
	}
}
