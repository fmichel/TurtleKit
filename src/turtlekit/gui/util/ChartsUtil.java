package turtlekit.gui.util;

import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;

/**
 * Class containing static methods which are shortcuts for creating charts with jfreechart.
 * 
 * @author Fabien Michel
 * @since TurtleKit 3.0.0.4
 * @version 0.1
 * 
 */
public class ChartsUtil {
	
	/**
	 * Creates a panel containing a XYLineChart. See {@link ChartFactory#createXYLineChart(String, String, String, XYDataset, PlotOrientation, boolean, boolean, boolean)}
	 * 
	 * @param dataset the dataset to use
	 * @param title the chart's title
	 * @param xName the x axis's title
	 * @param yName the y axis's title
	 * 
	 * @return a ChartPanel
	 * @see ChartFactory
	 * @see ChartPanel
	 */
	public static ChartPanel createChartPanel(final XYDataset dataset, String title, String xName, String yName){
		final JFreeChart chart = createChart(dataset, title, null, null);
		return new ChartPanel(chart);
	}
	
	/**
	 * Creates a default XYLineChart JFreeChart 
	 * 
	 * @param dataset the dataset to use
	 * @param title the chart's title
	 * @param xName the x axis's title
	 * @param yName the y axis's title
	 * 
	 * @return a JFreeChart
	 */
	public  static JFreeChart createChart(final XYDataset dataset, String title, String xName, String yName) {
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

}
