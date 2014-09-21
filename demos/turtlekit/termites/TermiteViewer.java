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


import java.awt.Color;

import turtlekit.kernel.Patch;
import turtlekit.viewer.TKDefaultViewer;

public class TermiteViewer extends TKDefaultViewer {
	
	/**
	 * Just to do some initialization work
	 */
	@Override
	protected void activate() {
		super.activate();
//		setSynchronousPainting(false);// fastest display mode
		double densityRate = 0.5;
		for (Patch patch : getPatchGrid()) {
			if (Math.random() < densityRate){
				patch.setColor(Color.yellow);}
			else
				patch.setColor(Color.black);
		}
	}

}
