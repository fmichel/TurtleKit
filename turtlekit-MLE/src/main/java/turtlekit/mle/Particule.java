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

import madkit.gui.UIProperty;
import madkit.simulation.EngineAgents;

@EngineAgents(environment = MLEEnvironment.class)
public class Particule extends MLEAgent {

	@UIProperty
	private static boolean test = false;

	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("vaccum");
		updateAttributes();
	}

	@Override
	public void updateAttributes() {
		attractQty = (float) Math.pow(BASE_QTY, level + 1);
		repulsionQty = (float) Math.pow(attractQty, level + 1);
		speed = (int) Math.pow(speedFactor, level);
	}
	

//	public void updateAttributes() {
//		attractQty = (float) Math.pow(level + 1, BASE_QTY.getValue());
//		repulsionQty = (float) Math.pow(attractQty, level + 1);
//		speed = (int) Math.pow(getSpeedFactor().getValue(), level);
//	}

	public void vaccum() {
		int code = get1DIndex();
		emitPheros(code);
		presence.incValue(code, attractQty);
		if (doNotAct())
			return;

		float upperAtt = upperAttraction.get(code);
		
		if(upperAtt > 0.001){
			nrj = 0;
			changeNextBehavior("findMembrane");
		}
//		steel();
		if(getLevel() > 1){
			if(lowerPresence.get(code) < 0.000001){
//				mutate();
				setLevel(level - 1);
			}
		}
//		int dir = presence.getMinDir(xcor(), ycor());
//		setHeading(dir);
		float others = presence.get(code);
		if (others > attractQty * 2) {
			int othersDir = presence.getMaxDirection(xcor(), ycor());
			setHeading(othersDir);
		}
			nrj++;
		wiggle(90);
		mutate();
	}

	public void findMembrane() {
		int code = get1DIndex();
		emitPheros(code);
		if (doNotAct())
			return;

		float att = upperAttraction.get(code);
		if(att < 0.1){
			return;
		}
		
//		float rep = upperRepulsion.get(code);
//		float others = attraction.get(code);

		setHeading(upperAttraction.getMaxDirection(xcor(), ycor()));
		int nextPatch = nextPatchCode();

		if(nextPatch == code){// This is a real tip !!
			fd(1);
			return;
		}
		if(upperRepulsion.get(nextPatch) > upperAttraction.get(nextPatch)){
			setNrj(0);
			changeNextBehavior("membrane");
		}
		if (! nextPatchIsOccupied(nextPatch)) {
//			fd(1);
			setNrj(getNrj() - 1);
			if(getNrj() < 0)
				setNrj(0);
		}
		else{
			mutate();
		}
		wiggle(45);
	}

	public void membrane() {
		int code = get1DIndex();
		emitPheros(code);
		if (doNotAct())
			return;
		int i = xcor();
		int j = ycor();
		float a = upperAttraction.get(i, j);
		if(a == 0){
			changeNextBehavior("vaccum");
		}
		float r = upperRepulsion.get(i, j);
		
//		if(a < attraction.get(code)){
//			setLevel(getLevel() - 1);
//			nrj = 0;
//		}
		
//		float others = attraction.get(code);

		setHeading(upperRepulsion.getMaxDirection(i, j)+45);
		int nextPatch = nextPatchCode();
		for(int u=0;u < 4;u++){
			if(nextPatch == code)// This is a real tip !!
				break;
			if(upperRepulsion.get(nextPatch) > upperAttraction.get(nextPatch)){
				setHeading(getHeading()+20);
				nextPatch = nextPatchCode();
			}
		}
		if (! nextPatchIsOccupied(nextPatch)) {
			setNrj(0);
		}
		else{
			nrj++;
		}
		fd(1);
		mutate();
//		if(getNrj() < 0)
//			setNrj(0);
//		if(getNrj() > NRJ_MUTATION.getValue()){
//			setNrj(getNrj() - 10);
//			return "findMembrane";
//		}
	}

//	public void steel() {
//		for (Turtle t : getPosition().getTurtlesHere()) {
//			if (t != this) {
//				AbstractMLEAgent mle = (AbstractMLEAgent) t;
//				if (mle.getLevel() == level) {
//					nrj = mle.getNrj() + 10;
//					mle.setNrj(0);
//				}
//			}
//		}
//	}

	public boolean mutate() {
		if (MUTATION && getLastMutation() > 10 && getNrj() > getNrjMutation()) {
			if (prng().nextFloat() > .99 && (getLevel() * 4) == prng().nextInt(getLevel() * 4 + 1)) {
				setLevel(level + 1);
			}
//			else{
//				setLevel(level - 1);
//			}
			setNrj(0);
			return true;
		}
		setLastMutation(getLastMutation() + 1);
		return false;
	}


	/**
	 * @return
	 */
	public int nextPatchCode() {
		return get1DIndex(xcor() + dx(),ycor() + dy());
	}
	
	public static void main(String[] args) {
		executeThisTurtle(30000
				, "--cuda"
				, "--noLog"
				, "--width", "400", "--height", "400"
				);

		
		
		
//		executeThisTurtle(2000
//				,Option.envDimension.toString(),"100,100"
////				,Option.noCuda.toString()
////				,LevelOption.agentLogLevel.toString(),"ALL"
////				,LevelOption.kernelLogLevel.toString(),"ALL"
//				,Option.startSimu.toString()
//				,Option.endTime.toString(),"10000"
//				,Option.environmentClass.toString(),MLEEnvGPUDiffusionAndGradients.class.getName()
//				,Option.viewers.toString(),PheromoneViewer.class.getName()
////				,Option.viewers.toString(),"null"
//				);
	}

	public static boolean isTest() {
		return test;
	}

	public static void setTest(boolean test) {
		Particule.test = test;
	}

}
