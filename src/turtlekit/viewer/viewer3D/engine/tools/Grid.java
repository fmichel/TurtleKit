package turtlekit.viewer.viewer3D.engine.tools;

import java.util.ArrayList;

import javax.media.opengl.GL2;

import turtlekit.kernel.Patch;

public class Grid {
	private PatchJogl patch;
	
	private Patch[] patchGrid;
	
	private int sizeX;
	private int sizeY;
	private int cellSize;
	
//	private float translateZ;
	
	private int numberPatch;
	
	private ArrayList<PatchJogl> listPatch;
	
	public PatchJogl getPatch() {
		return patch;
	}

	public int getSizePatch() {
		return cellSize;
	}

	public int getSizeX() {
		return sizeX;
	}

	public int getSizeY() {
		return sizeY;
	}

	public int getNumberPatch() {
		return numberPatch;
	}

	public ArrayList<PatchJogl> getListPatch() {
		return listPatch;
	}

	public Grid(){
		this.sizeX = 32;
		this.sizeY = 32;
		this.cellSize = 1;
		listPatch = new ArrayList<PatchJogl>();
	}
	
	public Grid(int sizeGridX, int sizeGridY, int sizePatchGrid){
		this.sizeX = sizeGridX;
		this.sizeY = sizeGridY;
		this.cellSize = sizePatchGrid;
		listPatch = new ArrayList<PatchJogl>();
	}
	
	public Grid(int sizeGridX, int sizeGridY, int sizePatchGrid, Patch[] patchGrid){
		this.sizeX = sizeGridX;
		this.sizeY = sizeGridY;
		this.cellSize = sizePatchGrid;
		listPatch = new ArrayList<PatchJogl>();
	}
	
	public void createGrid(){
		for(int i = 0; i < sizeX; i = i + cellSize){
			for(int j = 0; j < sizeY; j = j + cellSize){
				patch = new PatchJogl((sizeX * j + i), cellSize);
				listPatch.add(patch);
			}
		}
	}
	
	public void drawGrid(GL2 gl,boolean lightning){
		int i = 0;
		int j = 0;
	    if(listPatch != null){
            gl.glTranslatef(0.0f,0.0f,0.0f);
	    	for(PatchJogl p : listPatch){	    

	            gl.glTranslatef(cellSize,0.0f,0.0f);
	            System.err.println(patchGrid);
	            float colorData = patchGrid[sizeX * j + i].getColor().getRed();
	            
           		p.drawPatch2D(gl,colorData,lightning);	            
				i++;
				
				if(i == sizeX/cellSize){
					gl.glTranslatef((-i*cellSize),cellSize,0.0f);
					i = 0;
					j++;
				}
	        }
	    }
	}
}
