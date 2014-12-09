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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import madkit.agr.Organization;
import madkit.kernel.Probe;

import org.jfree.chart.ChartPanel;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import turtlekit.agr.TKOrganization;
import turtlekit.gui.util.ChartsUtil;
import turtlekit.kernel.Turtle;

@GenericViewer
public class PopulationCharter extends AbstractObserver{
	
	private XYSeriesCollection dataset = new XYSeriesCollection();
	private int index=0;
	private Map<Probe<Turtle>, XYSeries> series = new HashMap<>();
	private Set<String> handledRoles = new HashSet<>();
	private int timeFrame = 0;
	private boolean monitorTurtle;
	
	public PopulationCharter() {
		createGUIOnStartUp(); //prevent inappropriate launching and thus null pointer
	}

	@Override
	protected void activate() {
		setLogLevel(Level.ALL);
		super.activate();
		observe();
	}
	
	@Override
	public void setupFrame(JFrame frame) {
		final ChartPanel chartPanel = ChartsUtil.createChartPanel(dataset, "Population", null, null);
		chartPanel.setPreferredSize(new java.awt.Dimension(550, 250));
		frame.setContentPane(chartPanel);
		frame.setLocation(50, 0);
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
		handledRoles.add(role);
	}

	@Override
	protected void observe() {
		updateSeries();
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
	
	public void setMonitorTurtleRole(boolean b){
		monitorTurtle = b;
		if (isAlive()) {
			updateSeries();
		}
	}

	/**
	 * 
	 */
	protected void updateSeries() {
		TreeSet<String> roles = getExistingRoles();
		if(roles != null && roles.size() != handledRoles.size()){
			for (String role : roles) {
				if(handledRoles.add(role)){
					addSerie(role);
				}
			}
		}
	}

	/**
	 * @return
	 */
	private TreeSet<String> getExistingRoles() {
		TreeSet<String> roles = getExistingRoles(getCommunity(), TKOrganization.TURTLES_GROUP);
		if (roles != null) {
			roles.remove(Organization.GROUP_MANAGER_ROLE);
			if (! monitorTurtle) {
				roles.remove(TKOrganization.TURTLE_ROLE);
			}
		}
		return roles;
	}
	
}
