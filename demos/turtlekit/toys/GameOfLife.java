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

import java.awt.Color;
import java.util.Arrays;

import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TurtleKit.Option;

public class GameOfLife extends TKEnvironment{
	
	boolean[] gridBuffer;
	
	@Override
	protected void activate() {
		super.activate();
		final Patch[] patchGrid = getPatchGrid();
		gridBuffer = new boolean[patchGrid.length];
		Arrays.fill(gridBuffer, false);
		int index = 0;
		for (Patch p : patchGrid) {
			if(Math.random() < .4 ){
				p.setColor(Color.RED);
				gridBuffer[index] = true;
			}
			index++;
		}
	}

	@Override
	protected void update() {
		int index = 0;
		for (Patch p : getPatchGrid()) {
			int lifeCounter = 0;
			for(Patch tmp : p.getNeighbors(1,false)){
				if(tmp.getColor() == Color.RED){
					lifeCounter++;
				}
			}
			if(gridBuffer[index] && (lifeCounter < 2 || lifeCounter > 3)){
				gridBuffer[index] = false;
			}
			else if(lifeCounter == 3)
				gridBuffer[index] = true;
			index++;
		}
		index = 0;
		for (Patch p : getPatchGrid()) {
			p.setColor(gridBuffer[index++] ? Color.RED : Color.BLACK);
		}
	}
	
	public static void main(String[] args) {
		executeThisEnvironment(Option.envDimension.toString(),"150,150"
				,Option.startSimu.toString()
				);
	}
}

