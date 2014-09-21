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


import java.awt.Color;

import turtlekit.viewer.TKDefaultViewer;

public class GasViewer extends TKDefaultViewer {
	
	/**
	 * Just to do some initialization work
	 */
	@Override
	protected void activate() {
//		setSynchronousPainting(false);
		super.activate();
		int wallX = Integer.parseInt(getMadkitProperty("wallX"));
		for (int i = 0; i < getHeight(); i++) {
			getPatch(wallX, i).setColor(Color.WHITE);
			getPatch(0, i).setColor(Color.WHITE);
			getPatch(getWidth()-1, i).setColor(Color.WHITE);
		}
		for (int i = 0; i < getWidth(); i++) {
			getPatch(i, 0).setColor(Color.WHITE);
			getPatch(i, getHeight()-1).setColor(Color.WHITE);
		}
		getPatch(wallX, getHeight()/2).setColor(Color.BLACK);
//		getPatch(wallX, 1+getHeight()/2).setColor(Color.BLACK);
	}
	

}
