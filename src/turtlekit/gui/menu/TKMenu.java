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
package turtlekit.gui.menu;

import java.awt.event.KeyEvent;

import javax.swing.JMenu;

import madkit.kernel.AbstractAgent;
import turtlekit.kernel.TurtleKit;

/**
 * An out of the box menu for MaDKit applications
 * 
 * @author Fabien Michel
 * @since MaDKit 5.0.0.9
 * @version 0.9
 * 
 */
public class TKMenu extends JMenu {//TODO i18n

	private static final long serialVersionUID = 6177193453649323680L;

	/**
	 * Builds a menu featuring the following actions:
	 * <ul>
	 * </ul>
	 * 
	 * @param agent the agent for which this menu
	 * will be built.
	 */
	public TKMenu(final AbstractAgent agent){
		setText("TurtleKit");
		setMnemonic(KeyEvent.VK_T);
		TurtleKit.addTurleKitActionsTo(this, agent);
	}
}
