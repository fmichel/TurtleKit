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

import javafx.scene.paint.Color;
import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.DefaultTurtle;

/**
 * The TK Termite version of the famous Resnick's termites example.
 *
 * @author Fabien MICHEL
 * @version 6.0
 */

public class Termite extends DefaultTurtle {

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = 0.01)
	@UIProperty(category = "Environment")
	static double chipsDensity = 0.5;

	@SliderProperty(minValue = 1, maxValue = 100, scrollPrecision = 1)
	@UIProperty
	static double jump = 20;

	@Override
	protected void onActivation() {
		super.onActivation();
		getEnvironment().askPatchesOnStartup(p -> {
			p.setColor(prng().nextDouble() > chipsDensity ? Color.BLACK : Color.YELLOW);
		});
		changeNextBehavior("searchForChip");
	}

	// ////////here begin the methods that are used to define the termite's
	// behavior

	/**
	 * findEmptyPatch is a one time step behavior corresponding to a list of
	 * actions. So this method will be entirely executed, sure that no other turtle
	 * of the simulation is activated. It returns a String as the behavior that the
	 * turtle will take for the next time step
	 */
	public void findEmptyPatch() {
		wiggle();
		if (getPatchColor() == Color.BLACK) {
			setPatchColor(Color.YELLOW);
			changeNextBehavior("getAway");
		}
	}

	/** another one step behavior */
	public void getAway() {
		if (getPatchColor() == Color.BLACK)
			changeNextBehavior("searchForChip");
		randomHeading();
		fd(getJump());
	}

	/** another one step behavior */
	public void searchForChip() {
		wiggle();
		if (getPatchColor() == Color.YELLOW) {
			setPatchColor(Color.BLACK);
			fd(getJump());
			changeNextBehavior("findNewPile");
		}
	}

	/** another one step behavior */
	public void findNewPile() {
		if (getPatchColor() == Color.YELLOW)
			changeNextBehavior("findEmptyPatch");
		wiggle();
	}

	public static void main(String[] args) {
		executeThisTurtle(1000, "--width", "600"
//				,"--endTime","500"
//				, "--madkitLogLevel", "ALL"
//				,"--headless"
//				,"--start"
				);
//		, "-v", TermiteViewer.class.getName());
	}

	/**
	 * @return the chipsDensity
	 */
	public static double getChipsDensity() {
		return chipsDensity;
	}

	/**
	 * @param chipsDensity the chipsDensity to set
	 */
	public static void setChipsDensity(double chipsDensity) {
		Termite.chipsDensity = chipsDensity;
	}

	/**
	 * @return the jump
	 */
	public static double getJump() {
		return jump;
	}

	/**
	 * @param jump the jump to set
	 */
	public static void setJump(double jump) {
		Termite.jump = jump;
	}

}
