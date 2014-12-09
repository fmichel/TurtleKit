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
package turtlekit.viewer;

import java.awt.Color;
import java.awt.Graphics;
import java.util.ConcurrentModificationException;

import turtlekit.kernel.Patch;
import turtlekit.kernel.Turtle;

/**
 * The default viewer for TK 3
 * 
 * @author Fabien Michel
 *
 */
@GenericViewer
public class TKDefaultViewer extends AbstractGridViewer{
	
	@Override
	protected void render(Graphics g) {
		try {
			int index = 0;
			final Patch[] grid = getPatchGrid();
				final int w = getWidth();
				for (int j = getHeight() - 1; j >= 0; j--) {
					for (int i = 0; i < w; i++) {
						final Patch p = grid[index];
						if (p.isEmpty()) {
							paintPatch(g, p, i * cellSize, j * cellSize, index);
						} 
						else {
								try {
									paintTurtle(g, p.getTurtles().get(0), i * cellSize, j * cellSize);
								} catch (NullPointerException | IndexOutOfBoundsException e) {//for the asynchronous mode
								}
							}
						index++;
					}
			}
		} catch (ConcurrentModificationException e) {//FIXME
		}
		g.setColor(Color.RED);
		g.drawLine(getWidth()*cellSize, getHeight()*cellSize, 0, getHeight()*cellSize);
		g.drawLine(getWidth()*cellSize, getHeight()*cellSize, getWidth()*cellSize, 0);
	}

	public void paintTurtle(final Graphics g, final Turtle t, final int i, final int j) {
		g.setColor(t.getColor());
		g.fillRect(i , j , cellSize, cellSize);
	}

	public void paintPatch(final Graphics g, final Patch p, final int x, final int y, final int index) {
		final Color c = p.getColor();
		if (c != Color.BLACK) {
			g.setColor(c);
			g.fillRect(x , y , cellSize, cellSize);
		}
	}

}

