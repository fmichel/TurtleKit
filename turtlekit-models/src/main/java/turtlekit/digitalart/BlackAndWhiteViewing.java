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
package turtlekit.digitalart;


import java.util.logging.Level;

import turtlekit.kernel.Patch;
import turtlekit.viewer.jfx.FXPheroViewer;

public class BlackAndWhiteViewing extends FXPheroViewer {

	@Override
	protected void onActivation() {
		getEnvironment().getPheromone("blackNwhite", 0, 33);
		super.onActivation();
		setSelectedPheromone("blackNwhite");
		getSelectedPheromone().incValue(getWidth()/2, getHeight()/2, 100000000);
	}
	
	@Override
	public void paintPatch(Patch p, int x, int y, int index) {
		final int a = getSelectedPheromone().get(index).intValue()%256;
		getGraphics().setFill(javafx.scene.paint.Color.rgb(a, a, a));
		getGraphics().fillRect(x, y, getCellSize(), getCellSize());
	}

	public static void main(String[] args) {
		executeThisViewer(
				"--width","200"
				,"--height","200"
//				,"--tkLogLevel", Level.ALL.toString()
				,"--madkitLogLevel", Level.ALL.toString()
				,"--kernelLogLevel", Level.ALL.toString()
				,"--start"
				);
	}
}
