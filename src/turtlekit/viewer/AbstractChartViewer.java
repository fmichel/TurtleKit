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

import madkit.kernel.Watcher;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;

import turtlekit.agr.TKOrganization;
import turtlekit.kernel.TurtleKit;

public abstract class AbstractChartViewer extends Watcher{
	
	private String community;
	
	public AbstractChartViewer() {
		createGUIOnStartUp();
	}
	
	@Override
	protected void activate() {
		community = getMadkitProperty(TurtleKit.Option.community);
		requestRole(community, TKOrganization.ENGINE_GROUP,TKOrganization.VIEWER_ROLE);
	}
	
	public ChartPanel createChartPanel(final XYDataset dataset, String title, String xName, String yName){
		final JFreeChart chart = createChart(dataset, title, null, null);
		return new ChartPanel(chart);
	}
	
	/**
	 * automatically invoked for each time step
	 */
	protected abstract void observe();

	private JFreeChart createChart(final XYDataset dataset, String title, String xName, String yName) {
		// create the chart...
		final JFreeChart chart = ChartFactory.createXYLineChart(
				title,      // chart title
				xName,                      // x axis label
				yName,                      // y axis label
				dataset,                  // data
				PlotOrientation.VERTICAL,
				true,                     // include legend
				true,                     // tooltips
				false                     // urls
		);

		// NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
		chart.setBackgroundPaint(Color.white);

		// get a reference to the plot for further customization...
		final XYPlot plot = chart.getXYPlot();
		plot.setBackgroundPaint(Color.lightGray);
		plot.setDomainGridlinePaint(Color.white);
		plot.setRangeGridlinePaint(Color.white);

		final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
//		final XYAreaRenderer2 renderer = new XYAreaRenderer2();
		renderer.setSeriesShapesVisible(0, false);
		renderer.setSeriesShapesVisible(1, false);
		renderer.setSeriesShapesVisible(2, false);
		plot.setRenderer(renderer);
		final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		return chart;
	}

	/**
	 * @return the community
	 */
	public String getCommunity() {
		return community;
	}

}
