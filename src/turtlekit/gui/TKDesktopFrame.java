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
package turtlekit.gui;

import javax.swing.JMenuBar;
import javax.swing.JToolBar;

import madkit.gui.MDKDesktopFrame;
import madkit.gui.menu.LaunchMain;
import madkit.kernel.AbstractAgent;
import turtlekit.gui.menu.LaunchTKConfigurations;
import turtlekit.gui.menu.LaunchTKModels;
import turtlekit.gui.menu.TKMenu;
import turtlekit.gui.toolbar.TKToolBar;
import turtlekit.kernel.TurtleKit;

/**
 * @author Fabien Michel
 * @since MadKit 5.0.0.22
 * @version 0.9
 * 
 */
public class TKDesktopFrame extends MDKDesktopFrame {


	/**
	 * 
	 */
	private static final long serialVersionUID = 5367066495662186797L;

	/**
	 * @return
	 */
	@Override
	public JMenuBar getMenuBar(final AbstractAgent guiManager) {
		TurtleKit.VERSION = guiManager.getMadkitProperty("turtlekit.version");
		setTitle("TurtleKit "+TurtleKit.VERSION);
		final JMenuBar menuBar = new JMenuBar();
		menuBar.add(new TKMenu(guiManager));
		menuBar.add(new LaunchMain("Quick launch"){
			private static final long serialVersionUID = 1L;
			@Override
			public void update() {
				super.update();
				for (int i = 0; i < getItemCount(); i++) {
//					System.err.println(getItem(i).getActionCommand());
					if(getItem(i).getActionCommand().contains("turtlekit.kernel")){
						remove(getItem(i--));//lol. Bad as hell but the api is poor for doing what we want
					}
				}
			}
		});
		menuBar.add(new LaunchTKConfigurations("Configs"));
		menuBar.add(new LaunchTKModels("XML Models"));
//		menuBar.add(new HelpMenu());
		return menuBar;
	}

	/**
	 * @return
	 */
	@Override
	public JToolBar getToolBar(AbstractAgent guiManager) {
		return new TKToolBar(guiManager);
	}

}
