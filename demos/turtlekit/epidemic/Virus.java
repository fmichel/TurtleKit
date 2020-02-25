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
/*
 * TurtleKit 3 - An Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package turtlekit.epidemic;

import static turtlekit.kernel.TurtleKit.Option.startSimu;

import java.awt.Color;
import java.util.List;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PopulationCharter;
import turtlekit.viewer.TKDefaultViewer;
import turtlekit.viewer.jfx.JFXViewer;


/** 
 * An infected turtle transmits virus, but not by sending a message,
	just by changing the color of the turtles who cross its way

  @author Fabien MICHEL
  @version 1.3 */

@SuppressWarnings("serial")
public class Virus extends Turtle 
{
	
	public Virus() {
		super("go");
		setColor(Color.GREEN);
	}

	@Override
	protected void activate() {
		super.activate();
		playRole("healty");
	}
	
	public String go(){
		if(getColor() == Color.RED)
			return "infected";
		if (generator.nextDouble() < 0.000001) {
			getInfected();
			return "infected";
		}
		wiggle();
		return "go";
	}
	
	private void getInfected(){
		setColor(Color.RED);
		giveUpRole("healty");
		playRole("infected");
	}
	
	private void getCured(){
		setColor(Color.GREEN);
		giveUpRole("infected");
		playRole("healty");
	}
	
	public String infected(){
//		if(getCurrentBehaviorCount() == 30)
//			return null;
		wiggle();
		List<Turtle> l = getPatchOtherTurtles();
		if(l.size() > 0){
			final Virus virus = (Virus)l.get(0);
			if (virus.getColor() == Color.GREEN) {
				virus.getInfected();
			}
		}
		if (generator.nextDouble() < 0.1) {
			getCured();
			return "go";
		}
		return "infected";
	}


	public static void main(String[] args) {
		executeThisTurtle(50000
				, Option.envDimension.toString(),"500,500"
//				,Option.startSimu.toString()
				,Option.viewers.toString(),
				PopulationCharter.class.getName()+";"+
				TKDefaultViewer.class.getName()+";"+
				JFXViewer.class.getName()
				,startSimu.toString()
				);

	}
}
