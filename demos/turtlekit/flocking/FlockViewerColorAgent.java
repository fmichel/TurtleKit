package turtlekit.flocking;

import java.awt.Color;
import java.awt.Graphics;

import madkit.gui.AgentFrame;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.Patch;
import turtlekit.kernel.Turtle;
import turtlekit.viewer.TKDefaultViewer;

public class FlockViewerColorAgent extends TKDefaultViewer {

	@Override
	public void setupFrame(AgentFrame frame) {
		super.setupFrame(frame);
		getDisplayPane().setBackground(Color.WHITE);
	}
	
	@Override
	protected void render(Graphics g) {
		for (final Turtle t : getGridProbe().getProbedAgent()
				.getTurtlesWithRoles(TKOrganization.TURTLE_ROLE)) {
			paintTurtle(g, t, t.xcor() * cellSize, t.ycor() * cellSize);
		}

		// paintTurtle(g, p.getTurtles().get(0), i * cellSize, j * cellSize);

		g.drawLine(getWidth() * cellSize, getHeight() * cellSize, 0,
				getHeight() * cellSize);
		g.drawLine(getWidth() * cellSize, getHeight() * cellSize, getWidth()
				* cellSize, 0);
	}
	@Override
	public void paintTurtle(final Graphics g, final Turtle t, final int i,
			final int j) {
		g.setColor(t.getColor());
		g.fillRect(i, j, cellSize, cellSize);
	}
	@Override
	public void paintPatch(final Graphics g, final Patch p, final int x, final int y, final int index) {
		
	}
}
