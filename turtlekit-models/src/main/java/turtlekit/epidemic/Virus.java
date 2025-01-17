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
 * TurtleKit 3 - An Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
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
package turtlekit.epidemic;

import java.util.List;

import javafx.scene.paint.Color;
import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import madkit.simulation.EngineAgents;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.viewer.TKViewer;
import turtlekit.viewer.TurtlePopulationChartDrawer;

/**
 * An infected turtle transmits virus, but not by sending a message, just by
 * changing the color of the turtles who cross its way
 * 
 * @author Fabien MICHEL
 * @version 1.4
 */

@EngineAgents(viewers = { TKViewer.class, TurtlePopulationChartDrawer.class })
public class Virus extends DefaultTurtle {

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = 0.01)
	@UIProperty(displayName = "get infected probability")
	static double goInfectedProbability = 0.001;

	@UIProperty(displayName = "recovering probability")
	static double goRecoveredProbability = 0.1;

	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("go");
		playRole("healty");
	}

	public void go() {
		if (getColor() == Color.RED)
			changeNextBehavior("infected");
		if (prng().nextDouble() < goInfectedProbability) {
			getInfected();
			changeNextBehavior("infected");
		}
		wiggle();
	}

	public void infected() {
		// if(getCurrentBehaviorCount() == 30)
		// return null;
		wiggle();
		List<Virus> l = getPatchOtherTurtles();
		if (!l.isEmpty()) {
			final Virus virus = l.get(0);
			if (virus.getColor() == Color.GREEN) {
				virus.getInfected();
			}
		}
		if (prng().nextDouble() < getGoRecoveredProbability()) {
			getCured();
			changeNextBehavior("go");
		}
	}

	private void getInfected() {
		setColor(Color.RED);
		giveUpRole("healty");
		playRole("infected");
	}

	private void getCured() {
		setColor(Color.GREEN);
		giveUpRole("infected");
		playRole("healty");
	}

	public static double getGoInfectedProbability() {
		return goInfectedProbability;
	}

	public static void setGoInfectedProbability(double goInfectedProbability) {
		Virus.goInfectedProbability = goInfectedProbability;
	}

	public static double getGoRecoveredProbability() {
		return goRecoveredProbability;
	}

	public static void setGoRecoveredProbability(double goRecoveredProbability) {
		Virus.goRecoveredProbability = goRecoveredProbability;
	}

	public static void main(String[] args) {
		executeThisTurtle(10_000
//				,"--headless"
				, "--noLog", "--width", "500", "--height", "500");
		// ,Option.startSimu.toString()
//				startSimu.toString());
//
	}
}
