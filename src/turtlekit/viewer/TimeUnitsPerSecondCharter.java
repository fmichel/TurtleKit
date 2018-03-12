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

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import madkit.gui.AgentFrame;
import madkit.kernel.Watcher;
import madkit.simulation.probe.SingleAgentProbe;
import turtlekit.agr.TKOrganization;
import turtlekit.gui.util.ChartsUtil;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.TurtleKit;

/**
 * A viewer displaying the simulation speed in terms of simulation time units per second
 * 
 * @author Fabien Michel
 * @since TurtleKit 3.0.0.3
 * @version 0.2
 */
@GenericViewer
public class TimeUnitsPerSecondCharter extends Watcher {

    private XYSeriesCollection dataset = new XYSeriesCollection();
    private XYSeries serie;
    private int refreshRate = 1000;
    private SingleAgentProbe<TKScheduler, Double> probe;
    private Timer timer;

    public TimeUnitsPerSecondCharter() {
	createGUIOnStartUp();
    }

    @Override
    protected void activate() {
	probe = new SingleAgentProbe<TKScheduler, Double>(getMadkitProperty(TurtleKit.Option.community), TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE, "GVT");
	addProbe(probe);
	serie = new XYSeries("States Per Second");
	dataset.addSeries(serie);
	initTimer();
    }

    @Override
    protected void end() {
	stopTimer();
	super.end();
    }

    /**
     * 
     */
    private void initTimer() {
	stopTimer();
	timer = new java.util.Timer(true);
	// clear after stabilization
	timer.scheduleAtFixedRate(new TimerTask() {

	    @Override
	    public void run() {
		serie.clear();
	    }
	}, 1000, 60000);
	timer.scheduleAtFixedRate(new TimerTask() {

	    private double last = 0;

	    @Override
	    public void run() {
		try {
		    final double gvt = probe.getPropertyValue();
		    final double statesPerSecond = (gvt - last);
			getLogger().fine(() -> "SimulatedTimeUnitPerSecond =" + statesPerSecond);
		    last = gvt;
		    SwingUtilities.invokeLater(new Runnable() {// avoiding null pointers on the awt thread

			@Override
			public void run() {
			    if (statesPerSecond > 0) {
				serie.add((int) gvt, statesPerSecond);
			    }
			}
		    });
		}
		catch(NullPointerException e) {// ugly but avoids e when quitting
		}
	    }
	}, 0, getRefreshRate());
    }

    /**
     * 
     */
    private void stopTimer() {
	if (timer != null) {
	    timer.cancel();
	    timer = null;
	}
    }

    @Override
    public void setupFrame(AgentFrame frame) {
	final ChartPanel chartPanel = ChartsUtil.createChartPanel(dataset, "Simulated time units per second", null, null);
	chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
	// frame.setContentPane(chartPanel);
	frame.add(chartPanel);
	frame.add(new JButton(new AbstractAction("clear") {

	    @Override
	    public void actionPerformed(ActionEvent e) {
		serie.clear();
	    }
	}), BorderLayout.SOUTH);
	frame.restoreUIPreferences();
    }

    /**
     * @return the refreshRate in ms
     */
    public int getRefreshRate() {
	return refreshRate;
    }

    /**
     * set the refresh frequency in ms.
     * 
     * @param refreshRate
     *            the refreshRate to set in ms
     */
    public void setRefreshRate(int refreshRate) {
	if (refreshRate != this.refreshRate && refreshRate > 0) {
	    this.refreshRate = refreshRate;
	    initTimer();
	}
    }

}
