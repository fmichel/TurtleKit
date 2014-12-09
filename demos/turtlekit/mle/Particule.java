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

import turtlekit.kernel.TurtleKit.Option;
import turtlekit.viewer.PheromoneViewer;
import turtlekit.viewer.PopulationCharter;
import turtlekit.viewer.StatesPerSecondCharter;


public class Particule extends AbstractMLEAgent {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5108389312044401762L;


	public Particule() {
		super("vaccum");
	}

	@Override
	public void updateAttributes() {
		attractQty = (float) Math.pow(BASE_QTY.getValue(), level + 1);
		repulsionQty = (float) Math.pow(attractQty, level + 1);
		speed = (int) Math.pow(getSpeedFactor().getValue(), level);
	}
	

//	public void updateAttributes() {
//		attractQty = (float) Math.pow(level + 1, BASE_QTY.getValue());
//		repulsionQty = (float) Math.pow(attractQty, level + 1);
//		speed = (int) Math.pow(getSpeedFactor().getValue(), level);
//	}

	public String vaccum() {
		int code = get1DIndex();
		emitPheros(code);
		presence.incValue(code, attractQty);
		if (doNotAct())
			return "vaccum";

		float upperAtt = upperAttraction.get(code);
		
		if(upperAtt > 0.001){
			nrj = 0;
			return "findMembrane";
		}
//		steel();
		if(getLevel() > 1){
			if(lowerPresence.get(code) < 0.000001){
//				mutate();
				setLevel(level - 1);
				return "vaccum";
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
		return "vaccum";
	}

	public String findMembrane() {
		int code = get1DIndex();
		emitPheros(code);
		if (doNotAct())
			return "findMembrane";

		float att = upperAttraction.get(code);
		if(att < 0.1){
			return "vaccum";
		}
		
//		float rep = upperRepulsion.get(code);
//		float others = attraction.get(code);

		setHeading(upperAttraction.getMaxDirection(xcor(), ycor()));
		int nextPatch = nextPatchCode();

		if(nextPatch == code){// This is a real tip !!
			fd(1);
			return "findMembrane";
		}
		if(upperRepulsion.get(nextPatch) > upperAttraction.get(nextPatch)){
			setNrj(0);
			return "membrane";
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
		return "findMembrane";
	}

	public String membrane() {
		int code = get1DIndex();
		emitPheros(code);
		if (doNotAct())
			return "membrane";
		int i = xcor();
		int j = ycor();
		float a = upperAttraction.get(i, j);
		if(a == 0){
			return "vaccum";
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
		return "membrane";
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
		if (MUTATION && getLastMutation() > 10 && getNrj() > NRJ_MUTATION.getValue()) {
			if (generator.nextFloat() > .99 && (getLevel()*4) == generator.nextInt(getLevel()*4+1)) {
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
//		System.setProperty("sun.java2d.xrender", "true");
//		String[] args2 = {"128","10","--GPU_gradients"};
		String[] args2 = {"100","10","--test"};
//		String[] args2 = {"150","110","--GPU_gradients"};
		args = args2;
		float percentage = Float.parseFloat(args[1])/100;
		int size = Integer.parseInt(args[0]);
		int nbAgents = (int) ((float) size * size * percentage);
//		int nbAgents = 1;
		executeThisTurtle(nbAgents
				,Option.envDimension.toString(),args[0]+","+size
				,args[2]
//				,"--cvs.file",args[4]
//				,"--GPU_gradients"
				,Option.cuda.toString()
//				,LevelOption.agentLogLevel.toString(),"ALL"
//				,Option.startSimu.toString()
//				,Option.endTime.toString(),args[5]
				,Option.environment.toString(),MLEEnvironment.class.getName()
				,Option.scheduler.toString(),MLEScheduler.class.getName()
				,Option.viewers.toString(),
				PheromoneViewer.class.getName()
				+";"+MyPopulationViewer.class.getName()
				+";"+
StatesPerSecondCharter.class.getName()
//				,Option.viewers.toString(),"null"
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

}

class MyPopulationViewer extends PopulationCharter{
	public MyPopulationViewer() {
		setTimeFrame(1000);
	}
}
