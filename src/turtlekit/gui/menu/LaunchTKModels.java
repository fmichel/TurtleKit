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

import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.JMenuItem;

import madkit.action.KernelAction;
import madkit.action.MDKAbstractAction;
import madkit.gui.menu.LaunchMDKConfigurations;
import madkit.kernel.MadkitClassLoader;
import turtlekit.kernel.TurtleKit;
import turtlekit.kernel.TurtleKit.Option;

/**
 * This class builds a {@link JMenu} containing all the 
 * MDK configuration files found on the class path.
 * Each item will launch a separate instance of MaDKit
 * using the corresponding configuration files
 * 
 * @author Fabien Michel
 * @since TurtleKit 3.0.0.1
 * @version 0.9
 * 
 */
public class LaunchTKModels extends LaunchMDKConfigurations {

	/**
	 * Builds a new menu.
	 * @param agent the agent according 
	 * to which this menu should be created, i.e. the
	 * agent that will be responsible of the launch.
	 * @param title the title to use
	 */
	public LaunchTKModels(final String title) {
		super(title);
		setMnemonic(KeyEvent.VK_M);
	}


	@Override
	public void update() {//TODO clean up xml related
		removeAll();
		final Action a = new MDKAbstractAction(KernelAction.LAUNCH_XML.getActionInfo() ) {
			private static final long	serialVersionUID	= 1L;
			@Override
			public void actionPerformed(ActionEvent e) {
				new TurtleKit(Option.model.toString(),e.getActionCommand());
				}
		};
		for (final String string : MadkitClassLoader.getXMLConfigurations()) {
				JMenuItem name = new JMenuItem(a);
				name.setActionCommand(string);
				name.setText(string);// + " " + madkitProperties);
				add(name);
		}
		setVisible(getItemCount() != 0);
	}

}
