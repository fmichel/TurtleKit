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

import java.awt.Color;
import java.awt.Graphics;

import turtlekit.kernel.Patch;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PheromoneViewer;

public class BlackAndWhiteViewing extends PheromoneViewer {

	@Override
	protected void activate() {
		super.activate();
		setSynchronousPainting(true);
		getCurrentEnvironment().getPheromone("blackNwhite", 0, 33);
		setSelectedPheromone("blackNwhite");
		getSelectedPheromone().incValue(getWidth()/2, getHeight()/2, 100000000);
	}
	
	@Override
	public void paintPatch(Graphics g, Patch p, int x, int y, int index) {
		final int a = ((int) getSelectedPheromone().get(index))%256;
		g.setColor(new Color(a,a,a));
		g.fillRect(x,y,cellSize,cellSize);
	}

	public static void main(String[] args) {
		executeThisViewer(
				Option.envDimension.toString(),"100,100"
				,Option.startSimu.toString()
				);
	}
}
