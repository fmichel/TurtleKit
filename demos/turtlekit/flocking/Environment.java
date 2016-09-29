package turtlekit.flocking;

import java.util.List;
import java.util.Random;

import turtlekit.cuda.CudaAverageField;
import turtlekit.cuda.CudaEngine;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;

/**
 * The Environment in this flocking simulation
 * <p>
 * The environment is characterized by:
 * <ul>
 * <li>Its size</li>
 * </ul>
 * </p>
 * 
 * @author Emmanuel Hermellin
 * 
 * @version 0.1
 * 
 * @see turtlekit.kernel.TKEnvironment
 * 
 */

public class Environment extends TKEnvironment {
    
    /**
     * The GPU Module
     */
    private CudaAverageField cudaHeadingGrid;
     
    /**
     * The size of the environment
     */
    private static int envDimension = 512;
     
    /**
     * Do you want to use CUDA ?
     */
    private static boolean CUDA = true;
     
    /**
     * The array containing the heading of all the agents
     */
    private static float headingSheet[] = new float[envDimension * envDimension];
     
    /**
     * The array containing all the Patch of the Grid
     * @see Patch
     */
    private Patch patchSheet[] = new Patch[envDimension * envDimension];
     
    /**
     * Random number generator
     */
    protected static Random generator = new Random(); //TODO ne peut on pas réutiliser celui des Turtles ?????
 
    /**
     * Return the GPU module
     * @return cudaHeadingGrid
     */
    public CudaAverageField getCudaHeadingGrid() {
        return cudaHeadingGrid;
    }
     
    /**
     * Return the size of the environment
     * @return envDimension
     */
    public static int getEnvDimension() {
        return envDimension;
    }
 
    /**
     * Return if CUDA is used or not
     * @return CUDA
     */
    public static boolean isCUDA() {
        return CUDA;
    }
     
    /**
     * Activate of the Environment
     * @see TKEnvironment#activate()
     */
    protected void activate(){
        super.activate();
        patchSheet = this.getPatchGrid();
        initSpeedAndHeadingSheet();
        makeTheCache();
        cudaHeadingGrid = new CudaAverageField("Average",envDimension,envDimension,BirdFlockingUnify.getVision(),headingSheet);
    }
     
    /**
     * Forcing caching of the Patch
     * @see Environment#activate()
     */
    protected void makeTheCache(){
        for (Patch p : getPatchGrid()) {
            p.getNeighbors(10, true);
        }
    }
     
    /**
     * Initializing the array containing the heading of the agents
     */
    protected void initSpeedAndHeadingSheet(){
        for(int i = 0 ; i < envDimension * envDimension ; i++){
                headingSheet[i] = -1;       
        }
    }
     
    /**
     * Update the environment
     * @see TKEnvironment#update()
     */
    @Override
    protected void update() {
        if(CUDA){
            updateSpeedAndHeadingSheetV2();
 
            cudaHeadingGrid.computeAverage();
 
            if(isSynchronizeGPU()){
                CudaEngine.cuCtxSynchronizeAll();   
            }
        }
    }
     
    /**
     * Accessing data compute by the GPU module
     * @see CudaAverageField
     */
    public float getCudaHeadingValue(int xcor, int ycor){
        return cudaHeadingGrid.getResult(get1DIndex(xcor, ycor));
    }
     
    /**
     * Set data which will be compute by the GPU module
     * @see CudaAverageField
     */
    public void setCudaHeadingValue(int xcor, int ycor, double heading){
        cudaHeadingGrid.set(get1DIndex(xcor, ycor),((float)heading));
    }
     
    /**
     * Update the array containing heading of the agents
     */
    protected void updateSpeedAndHeadingSheet(){
            for(int i = 0 ; i < envDimension * envDimension ; i++){
                List<BirdFlockingUnify> turtleList = patchSheet[i].getTurtles(BirdFlockingUnify.class);
                if(turtleList.isEmpty()){
                    cudaHeadingGrid.set(i,-1);
                }
                else{
                    for(BirdFlockingUnify b : turtleList){
                        cudaHeadingGrid.set(i,(float)b.getHeading());
                    }
                }
            }
    }
     
    /**
     * Update the array containing heading of the agents (V2)
     */
    protected void updateSpeedAndHeadingSheetV2(){
        int j = envDimension * envDimension;
        for(int i = j - 1 ; i > 0 ; i--){
            if(patchSheet[i].isEmpty()){
                cudaHeadingGrid.set(i,-1);
            }
        }
    }
}

