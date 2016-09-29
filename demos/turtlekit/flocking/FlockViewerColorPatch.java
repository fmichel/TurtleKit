package turtlekit.flocking;

import java.awt.Color;
import java.awt.Graphics;
import java.util.ConcurrentModificationException;

import turtlekit.kernel.Patch;
import turtlekit.kernel.Turtle;
import turtlekit.viewer.TKDefaultViewer;

public class FlockViewerColorPatch extends TKDefaultViewer {
	
	protected void render(Graphics g) {
		try {
			int index = 0;
			final Patch[] grid = getPatchGrid();
				final int w = getWidth();
				for (int j = getHeight() - 1; j >= 0; j--) {
					for (int i = 0; i < w; i++) {
						final Patch p = grid[index];
						if (!p.isEmpty()) {
							paintPatch(g, p, i * cellSize, j * cellSize, index);
						} 

						index++;
					}
			}
		} catch (ConcurrentModificationException e) {//FIXME
		}
		g.drawLine(getWidth()*cellSize, getHeight()*cellSize, 0, getHeight()*cellSize);
		g.drawLine(getWidth()*cellSize, getHeight()*cellSize, getWidth()*cellSize, 0);
	}
	
	
	public void paintTurtle(final Graphics g, final Turtle t, final int i,
			final int j) {
//		g.setColor(t.getColor());
//		g.fillRect(i, j, cellSize, cellSize);
	}
	
	@Override
	public void paintPatch(final Graphics g, final Patch p, final int x, final int y, final int index) {
			
	        int size = p.getTurtles().size();
			if(size > 0){
	          g.setColor(new Color(0, 255/(size*4), 0));
//				this.setColor(Color.WHITE);
	        }
	        else{
	            g.setColor(Color.white);	
//	        	this.setColor(Color.GREEN);
	        }
//			g.setColor(c);
			g.fillRect(x , y , cellSize, cellSize);
		
	}
}
