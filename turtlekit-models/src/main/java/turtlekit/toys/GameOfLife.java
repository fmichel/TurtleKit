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
package turtlekit.toys;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.WHITE;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.TKEnvironment;

public class GameOfLife extends TKEnvironment<GOLPatch> {
	
	@UIProperty
	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = 0.01)
	private double density = 0.1;

	@Override
	protected void onActivation() {
		super.onActivation();
		askPatches(p -> {
			if (prng().nextDouble() < getDensity()) {
				p.setColor(WHITE);
				p.setNextState(BLACK);
			}
		});
	}

	@Override
	protected void update() {
		askPatches(p -> {
			long aliveN = p.getNeighbors(1, false).stream().filter(a -> a.getColor() == WHITE).count();
			if (p.getNextState() == WHITE && (aliveN < 2 || aliveN > 3)) {
				p.setNextState(BLACK);
			} else if (aliveN == 3) {
				p.setNextState(WHITE);
			}
		});
		getGridModel().getGridParallelStream().forEach(p -> p.setColor(p.getNextState()));
	}

	/**
	 * Get the density of alive cells
	 * 
	 * @return the density
	 */
	public double getDensity() {
		return density;
	}

	/**
	 * Set the density of alive cells
	 * 
	 * @param density the density to set
	 */
	public void setDensity(double density) {
		this.density = density;
	}

	public static void main(String[] args) {
		executeThisEnvironment("--width", "300", "--height", "300");
	}

}