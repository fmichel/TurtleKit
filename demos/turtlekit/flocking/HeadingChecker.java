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
package turtlekit.flocking;


import java.util.logging.Level;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import madkit.gui.AgentFrame;
import madkit.simulation.SimulationException;
import madkit.simulation.probe.PropertyProbe;
import turtlekit.agr.TKOrganization;
import turtlekit.gui.util.ChartsUtil;
import turtlekit.viewer.AbstractObserver;

public class HeadingChecker extends AbstractObserver {
	
	private PropertyProbe<BirdFlockingUnify,Double> probeHeading;
	private XYSeries heading;
	private int index = 0;
	
	public HeadingChecker() {
		createGUIOnStartUp(); //prevent inappropriate launching and thus null pointer
	}

	/**
	 * Just to do some initialization work
	 */
	@Override
	protected void activate() {
		setLogLevel(Level.ALL);
		super.activate();
		probeHeading = new PropertyProbe<BirdFlockingUnify,Double>(getCommunity(), TKOrganization.TURTLES_GROUP, TKOrganization.TURTLE_ROLE, "angle");
		addProbe(probeHeading);
	}

	@Override
	public void setupFrame(AgentFrame frame) {
		XYSeriesCollection dataset = new XYSeriesCollection();
		final ChartPanel chartPanel = ChartsUtil.createChartPanel(dataset, "Average heading", null, null);
		chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
		heading = new XYSeries("Average heading");
		dataset.addSeries(heading);
		frame.setContentPane(chartPanel);
		frame.setLocation(50, 0);
//		XYSeries s = dataset.getSeries("Total");
	}
	
	@Override
	protected void observe() {
		double averageHeading = 0;
		double averageSpeed = 0;
		for (BirdFlockingUnify a : probeHeading.getCurrentAgentsList()) {
			averageHeading += probeHeading.getPropertyValue(a);
		}
		averageHeading /= (double) probeHeading.size();
		
		if(index % 10000 == 0){
			heading.clear();
		}

		try {
			heading.add(index, averageHeading);
		} catch (SimulationException | NullPointerException e ) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		index++;
	}
	

}
