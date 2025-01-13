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


import java.util.logging.Level;

import javax.swing.JFrame;

import madkit.kernel.Probe;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import turtlekit.agr.TKOrganization;
import turtlekit.gui.util.ChartsUtil;
import turtlekit.kernel.Turtle;
import turtlekit.viewer.AbstractObserver;

public class PhysicsChecker extends AbstractObserver {
	
	private int wallX;
	private Probe<Turtle> p;
	private XYSeries rightSide;
	private int index = 0;
	private XYSeries total;
	
	public PhysicsChecker() {
		createGUIOnStartUp(); //prevent inappropriate launching and thus null pointer
	}

	/**
	 * Just to do some initialization work
	 */
	@Override
	protected void activate() {
		setLogLevel(Level.ALL);
		super.activate();
		wallX = Integer.parseInt(getMadkitProperty("wallX"));
		p = new Probe<>(getCommunity(), TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE);
		addProbe(p);
	}

	@Override
	public void setupFrame(JFrame frame) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		final ChartPanel chartPanel = ChartsUtil.createChartPanel(dataset, "PV = nRT", null, null);
		chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
		rightSide = new XYSeries("Gas on the right side");
		dataset.addSeries(rightSide);
		total = new XYSeries("Total");
		dataset.addSeries(total);
		frame.setContentPane(chartPanel);
		frame.setLocation(50, 0);
		XYSeries s = dataset.getSeries("Total");
	}
	
	@Override
	protected void observe() {
		if(index % 10000 == 0){
			total.clear();
			rightSide.clear();
		}
		int nb = 0;
		for (Turtle t : p.getCurrentAgentsList()) {
			if(t.getX() > wallX){
				nb++;
			}
		}
		total.add(index, p.size());
		rightSide.add(index, nb);
		index++;
	}
	

}
