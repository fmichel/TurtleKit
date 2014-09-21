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
package turtlekit.mle;

import java.awt.Color;

import javax.swing.BoundedRangeModel;
import javax.swing.DefaultBoundedRangeModel;

import turtlekit.kernel.Turtle;
import turtlekit.pheromone.Pheromone;

public class AbstractMLEAgent extends Turtle {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1148771355405022926L;

	public static final int SPEED_UNIT = 1;

	public static final String ATT = "ATT";
	public static final String REP = "REP";
	public static final String PRE = "PRE";

	public static boolean MUTATION = false;

	public static BoundedRangeModel NRJ_MUTATION = new DefaultBoundedRangeModel(30,5,1,200){
		/**
		 * 
		 */
		private static final long serialVersionUID = -6156959089896905682L;

		public void setValue(int n) {
			super.setValue(n);
			System.err.println("NRJ_MUTATION = " + n);
		}
	};

	public static BoundedRangeModel BASE_QTY = new DefaultBoundedRangeModel(200000,1000,1,1000000){
		/**
		 * 
		 */
		private static final long serialVersionUID = -3239396008547848873L;

		public void setValue(int n) {
			super.setValue(n);
			System.err.println("base QTY = " + n);
		}
	};

	protected Pheromone upperAttraction, upperRepulsion, attraction, repulsion,
			presence, lowerPresence;

	protected int level;
	protected int speed;
	protected int nrj = 0;
	protected float attractQty;

	protected float repulsionQty;

	private int lastMutation;

	private int TEST_TIME = 0;

	static DefaultBoundedRangeModel repulsionFactor = new DefaultBoundedRangeModel(
			9, 1, 0, 100) {
		/**
				 * 
				 */
				private static final long serialVersionUID = 4126984409978383492L;

		@Override
		public void setValue(int n) {
			super.setValue(n);
			System.err.println("factor = " + n);
		}
	};

	static DefaultBoundedRangeModel speedFactor = new DefaultBoundedRangeModel(
			3, 1, 1, 101) {
		/**
				 * 
				 */
				private static final long serialVersionUID = 6815446930251298715L;

		@Override
		public void setValue(int n) {
			super.setValue(n);
			System.err.println("speed = " + n);
		}
	};

	public AbstractMLEAgent(String initMethod) {
		super(initMethod);
	}

	public void setLevel(int level) {
		if (level < 0)
			level = 0;
		giveUpRole("" + getLevel());
		this.level = level;
		playRole(""+getLevel());
		// System.err.println("mutating from "+getLevel()+" to "+level);
		// speed = (int) Math.pow(2, level);
		// speed = 3 * level;
		// speed = speed == 0 ? 1 : speed;
		initAttributes();
	}

	/**
	 * 
	 */
	private void initAttributes() {
		setLastMutation(0);
		setNrj(0);
		updatePheromones();
		updateAttributes();
		setCurrentBehaviorCount(0);
		updateColor(false);
	}

	public void updatePheromones() {
		upperAttraction = getPheromone(ATT + (level + 1));
		upperRepulsion = getPheromone(REP + (level + 1));
		presence = getPheromone(PRE + level);
		if (getLevel() != 0) {
			lowerPresence = getPheromone(ATT + (level - 1));
			attraction = getPheromone(ATT + (level));
			repulsion = getPheromone(REP + (level));
		}
	}

	@Override
	public String toString() {
		return "id "+hashCode()+" lvl "+getLevel()+ " h "+getHeading()+" nrj = "+getNrj();
	}

	/**
	 * @param level
	 */
//	public void updateAttributes() {
//		attractQty = (float) Math.pow(BASE_QTY.getValue(), level + 1);
//		repulsionQty = (float) Math.pow(attractQty,level+1);
////		repulsionQty = (float) Math.pow(attractQty, level + 1);
//		speed = (int) Math.pow(speedFactor.getValue(), level);
//	}

	public void updateAttributes() {
		attractQty = (float) Math.pow(level + 1, BASE_QTY.getValue());
//		repulsionQty = attractQty * 2;
		repulsionQty = (float) Math.pow(attractQty, level + 1);
		speed = (int) Math.pow(speedFactor.getValue(), level);

	}

	@Override
	protected void activate() {
		super.activate();
//		System.err.println(getMyRoles(getCommunity(), TKOrganization.TURTLES_GROUP));
//		randomHeading();
//		randomLocation();
//		setLevel(generator.nextInt(4));
//		home();
		initAttributes();
//		setLevel(0);
	}

	public int getLevel() {
		return level;
	}

	public void emitPheros(int code) {
//		TEST_TIME++;
//		if(TEST_TIME<3000 && TEST_TIME>1500){
//			return;
//		}
		if (level != 0) {
			attraction.incValue(code, attractQty);
			repulsion
					.incValue(code, (float) (attractQty * Math.pow(
							AbstractMLEAgent.getRepulsionFactor().getValue(),
							(level))));
		}
	}

//	public void steel() {
//		for (Turtle t : getPatchOtherTurtles()) {
//				AbstractMLEAgent mle = (AbstractMLEAgent) t;
//				if (mle.level == level && getNrj() >= mle.getNrj()) {
//					setNrj(getNrj() + 1);
//					mle.setNrj(0);
//					mle.setLastMutation(0);
//				}
//		}
//	}

	/**
	 * @param level
	 */
	public void updateColor(boolean membrane) {
		switch (level) {
		case 0:
			changeColor(membrane, Color.RED);
			break;
		case 1:
			changeColor(membrane, Color.BLUE);
			break;
		case 2:
			changeColor(membrane, Color.ORANGE);
			break;
		case 3:
			changeColor(membrane, Color.PINK);
			break;
		case 4:
			changeColor(membrane, Color.GREEN);
			break;
		case 5:
			changeColor(membrane, Color.RED);
			break;
		default:
			break;
		}
	}

	/**
	 * @param membrane
	 * @param red
	 */
	public void changeColor(boolean membrane, Color red) {
		if (membrane) {
			setColor(red.brighter());
		} else {
			setColor(red);
		}
	}

	public boolean doNotAct() {
		return getCurrentBehaviorCount() % speed != 0;
	}

//	public boolean checkOthers() {
//		boolean occupied = false;
//		for (Turtle t : getPatchOtherTurtles()) {
//				AbstractMLEAgent mle = (AbstractMLEAgent) t;
//				if (mle.level == level) {
//					occupied = true;
//					if (getNrj() >= mle.getNrj()) {
//						setNrj(getNrj() + 1);
//						mle.setNrj(0);
//					}
//			}
//		}
//		return occupied;
//	}
	
	public boolean nextPatchIsOccupied(int code){
		boolean occupied = false;
		for (Turtle t : getPatchOtherTurtles()) {
				AbstractMLEAgent mle = (AbstractMLEAgent) t;
				if (mle.level == level) {
					occupied = true;
					if (getNrj() >= mle.getNrj()) {
						setNrj(getNrj() + 1);
						mle.setNrj(0);
					}
					break;
			}
		}
//		for (Turtle t : getPatch().getTurtles()) {
//				AbstractMLEAgent mle = (AbstractMLEAgent) t;
//				if (t != this && mle.level == level) {
//					occupied = true;
//					if (getNrj() >= mle.getNrj()) {
//						setNrj(getNrj() + 1);
//						mle.setNrj(0);
//					}
//					break;
//			}
//		}
		return occupied;
	}

//	protected boolean occupied() {
//		for (Turtle t : getPosition().getTurtlesHere()) {
//			if (t != this) {
//				AbstractMLEAgent mle = (AbstractMLEAgent) t;
//				if (mle.level == level) {
//					return true;
//				}
//			}
//		}
//		return false;
//	}

	public boolean mutate() {
		if (MUTATION && getLastMutation() > 50 && getNrj() > NRJ_MUTATION.getValue()) {
			if (generator.nextFloat() > .5) {
				setLevel(getLevel() + 1);
			}
			setNrj(0);
			return true;
		}
		setLastMutation(getLastMutation() + 1);
		return false;
	}

	public static DefaultBoundedRangeModel getRepulsionFactor() {
		return repulsionFactor;
	}

	/**
	 * @param mUTATION the mUTATION to set
	 */
	public static void setMUTATION(boolean mUTATION) {
		MUTATION = mUTATION;
	}

	/**
	 * @return the speedFactor
	 */
	public static DefaultBoundedRangeModel getSpeedFactor() {
		return speedFactor;
	}

	public int getNrj() {
		return nrj;
	}

	public void setNrj(int nrj) {
		this.nrj = nrj;
	}

	public int getLastMutation() {
		return lastMutation;
	}

	public void setLastMutation(int lastMutation) {
		this.lastMutation = lastMutation;
	}

}
