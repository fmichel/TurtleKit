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

import static turtlekit.kernel.TurtleKit.Option.cuda;

import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;

import javax.swing.JCheckBoxMenuItem;
import javax.swing.JMenu;

import turtlekit.kernel.TKEnvironment;

public class CudaMenu extends JMenu {

	public CudaMenu(final TKEnvironment agent) {
		super("Cuda");
		setMnemonic(KeyEvent.VK_C);
		final JCheckBoxMenuItem item = new JCheckBoxMenuItem("Synchronized CPU / GPU");
		item.setSelected(true);
		item.addItemListener(new ItemListener() {
			@Override
			public void itemStateChanged(ItemEvent e) {
				agent.synchronizeEnvironment(item.isSelected());
			}
		});
		add(item);
		setToolTipText("only available when Cuda is in use");
		setEnabled(agent.isMadkitPropertyTrue(cuda));
	}
}
