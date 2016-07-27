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
package turtlekit.pheromone;

import java.awt.Dimension;

import javax.swing.DefaultBoundedRangeModel;
import javax.swing.JPanel;
import javax.swing.border.TitledBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import madkit.gui.SwingUtil;

public class PheromoneView extends JPanel implements ChangeListener{

	private DefaultBoundedRangeModel red;
	private DefaultBoundedRangeModel green;
	private DefaultBoundedRangeModel blue;
	private Pheromone<Float> myPhero;
	//	private Environment env;
	private JPanel evapPanel;
	private JPanel diffusionPanel;

	public PheromoneView(Pheromone<Float> selectedPheromone) {
		myPhero = selectedPheromone;
		final DefaultBoundedRangeModel evaporationPercentage = selectedPheromone.getEvaporationCoefficientModel();
		evaporationPercentage.addChangeListener(this);
		evapPanel = SwingUtil.createSliderPanel(evaporationPercentage,"evaporation");
		add(evapPanel);
		final DefaultBoundedRangeModel diffusionPercentage = selectedPheromone.getDiffusionCoefficientModel();
		diffusionPercentage.addChangeListener(this);
		diffusionPanel = SwingUtil.createSliderPanel(diffusionPercentage, "diffusion");
		add(diffusionPanel);
		setPreferredSize(new Dimension(600,40));
		
		red = new DefaultBoundedRangeModel(50, 0, 0, 255);
		add(SwingUtil.createSliderPanel(red, "red percentage"));
		green = new DefaultBoundedRangeModel(50, 0, 0, 255);
		add(SwingUtil.createSliderPanel(green, "green percentage"));
		blue = new DefaultBoundedRangeModel(50, 0, 0, 255);
		add(SwingUtil.createSliderPanel(blue, "blue percentage"));
		stateChanged(null);
	}

	/**
	 * @return the red
	 */
	public DefaultBoundedRangeModel getRed() {
		return red;
	}

	/**
	 * @return the green
	 */
	public DefaultBoundedRangeModel getGreen() {
		return green;
	}

	/**
	 * @return the blue
	 */
	public DefaultBoundedRangeModel getBlue() {
		return blue;
	}

	/**
	 * @return the evapPanel
	 */
	public JPanel getEvapPanel() {
		return evapPanel;
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		((TitledBorder) diffusionPanel.getBorder())
		.setTitle("diffusion " + myPhero.getDiffusionCoefficient());
		diffusionPanel.repaint();
		((TitledBorder) evapPanel.getBorder())
		.setTitle("evaporation " + myPhero.getEvaporationCoefficient());
		evapPanel.repaint();
		
	}

}
