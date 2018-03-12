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

import java.awt.Container;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import madkit.agr.Organization;
import madkit.gui.AgentFrame;
import madkit.kernel.AbstractAgent;
import madkit.kernel.Probe;
import turtlekit.agr.TKOrganization;
import turtlekit.gui.util.ChartsUtil;
import turtlekit.kernel.Turtle;

/**
 * 
 * Creates a chart tracing the population for each role taken by the turtles in the 
 * Simulation. The "turtle" role is ignored by default.
 * 
 * @author Fabien Michel
 *
 */
@GenericViewer
public class PopulationCharter extends AbstractObserver{
	
	private int index=0;
	private XYSeriesCollection dataset = new XYSeriesCollection();
	private Map<Probe<Turtle>, XYSeries> series = new HashMap<>();
	private int timeFrame = 0;
	private Set<String> handledRoles = new HashSet<>();
	private HashSet<String> ignoredRoles = new HashSet<>();
	
	public PopulationCharter() {
		createGUIOnStartUp(); //prevent inappropriate launching and thus null pointer
		ignoreRole(Organization.GROUP_MANAGER_ROLE);
		ignoreRole(TKOrganization.ENVIRONMENT_ROLE);
		ignoreRole(TKOrganization.TURTLE_ROLE);
	}

	@Override
	protected void activate() {
//		setLogLevel(Level.ALL);
		super.activate();
		for (String role : handledRoles) {
			addSerie(role);
		}
		observe();
	}
	
	@Override
	public void setupFrame(AgentFrame frame) {
		final ChartPanel chartPanel = ChartsUtil.createChartPanel(dataset, "Population", null, null);
		chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
		frame.setContentPane(chartPanel);
		frame.setLocation(50, 0);
	}
	
	@Override
	protected void observe() {
		refreshMonitoredRoles();
		SwingUtilities.invokeLater(new Runnable() {//avoiding null pointers on the awt thread
			@Override
			public void run() {
				for(Entry<Probe<Turtle>, XYSeries> entry : series.entrySet()) {
					entry.getValue().add(index, entry.getKey().size());
				}
				index++;
				if(timeFrame > 0 && index % timeFrame == 0){
					for (XYSeries serie : series.values()) {
						serie.clear();
					}
				}
			}
		});
	}
	
	public void setTimeFrame(int interval){
		timeFrame = interval;
	}
	
	/**
	 * Starts ignoring this role.
	 * 
	 * @param role
	 */
	public void ignoreRole(String role){
		if (handledRoles.remove(role)) {
			for (Probe<? extends AbstractAgent> p : getProbes()) {
				if(p.getRole().equals(role)){
					dataset.removeSeries(series.remove(p));
				}
			}
		}
		ignoredRoles.add(role);
	}
	
	/**
	 * Starts monitoring this role.
	 * 
	 * @param role
	 */
	public void monitorRole(String role){
		if(handledRoles.add(role)){
			if (isAlive()) {
				addSerie(role);
			}
		}
		ignoredRoles.remove(role);
	}

	/**
	 * @param role
	 */
	private void addSerie(String role) {
		final Probe<Turtle> probe = new Probe<Turtle>(getCommunity(), TKOrganization.TURTLES_GROUP, role);
		addProbe(probe);
		XYSeries serie = new XYSeries(role);
		series.put(probe, serie);
		dataset.addSeries(serie);
	}

	/**
	 * @return
	 */
	private void refreshMonitoredRoles() {
		final Set<String> roles = getExistingRoles(getCommunity(), TKOrganization.TURTLES_GROUP);
		if (roles != null) {
			roles.removeAll(ignoredRoles);
			roles.removeAll(handledRoles);
			for (String role : roles) {//new roles
				monitorRole(role);
			}
		}
	}
	
}
