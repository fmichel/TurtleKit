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

import java.util.logging.Level;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import madkit.simulation.probe.SingleAgentProbe;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import turtlekit.agr.TKOrganization;
import turtlekit.kernel.TKScheduler;

@GenericViewer
public class StatesPerSecondCharter extends AbstractChartViewer {

	private XYSeriesCollection dataset = new XYSeriesCollection();
	private int timeFrame = 0;
	private XYSeries serie;
	private SingleAgentProbe<TKScheduler, Double> probe;

	@Override
	protected void activate() {
		setLogLevel(Level.ALL);
		super.activate();
		probe = new SingleAgentProbe<>(getCommunity(),
				TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,
				"statesPerSecond");
		addProbe(probe);
		serie = new XYSeries("States Per Second");
		dataset.addSeries(serie);
	}

	@Override
	public void setupFrame(JFrame frame) {
		final ChartPanel chartPanel = createChartPanel(dataset,
				"States Per Second", null, null);
		chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
		frame.setContentPane(chartPanel);
		frame.setLocation(50, 0);
	}

	@Override
	protected void observe() {
		SwingUtilities.invokeLater(new Runnable() {// avoiding null pointers on
													// the awt thread
					@Override
					public void run() {
						final double gvt = probe.getProbedAgent().getGVT();
						final Double propertyValue = probe.getPropertyValue();
						if (propertyValue > 0) {
							serie.add(gvt, propertyValue);
						}
						if (timeFrame > 0 && gvt % timeFrame == 0) {
							serie.clear();
						}
					}
				});
	}

	public void setTimeFrame(int interval) {
		timeFrame = interval;
	}

}
