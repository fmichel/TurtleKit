package turtlekit.kernel;

import static java.lang.Math.abs;
import static java.lang.Math.atan;
import static java.lang.Math.hypot;
import static java.lang.Math.min;
import static java.lang.Math.toDegrees;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class TKGridModel {

    private boolean torusMode;
    private int width;
    private int height;
    private Patch[] patchGrid;
    private TKEnvironment environment;

    public TKGridModel() {
	this(true, 200, 200);
    }

    /**
     * @param torusMode
     * @param width
     * @param height
     */
    public TKGridModel(Class<? extends Patch> patchClass, TKEnvironment env, boolean torusMode, int width, int height) {
	this.torusMode = torusMode;
	this.width = width;
	this.height = height;
	this.environment = env;
	initPatchGrid(patchClass);
    }

    /**
     * @param torusMode
     * @param width
     * @param height
     */
    public TKGridModel(boolean torusMode, int width, int height) {
	this(Patch.class, null, torusMode, width, height);
    }

    private void initPatchGrid(Class<? extends Patch> patchClass) {
	patchGrid = new Patch[width * height];
	IntStream.range(0, width).parallel().forEach(i -> {
	    IntStream.range(0, height).parallel().forEach(j -> {
		final int retrieveIndex = compute1DIndex(i, j);
		try {
		    final Patch patch = patchClass.newInstance();
		    patchGrid[retrieveIndex] = patch;
		    patch.setCoordinates(i, j);
		    patch.setEnvironment(environment);
		    patch.setGridModel(this);
		}
		catch(InstantiationException | IllegalAccessException e) {
		    e.printStackTrace();
		}
	    });
	});
	// for (int i = 0; i < width; i++) {
	// for (int j = 0; j < height; j++) {
	// final int retrieveIndex = compute1DIndex(i, j);
	// try {
	// final Patch patch = patchClass.newInstance();
	// patchGrid[retrieveIndex] = patch;
	// patch.setCoordinates(i, j);
	// patch.setEnvironment(environment);
	// patch.setGridModel(this);
	// } catch (InstantiationException | IllegalAccessException e) {
	// e.printStackTrace();
	// }
	// }
	// }
    }

    public List<Patch> getNeighborsOf(Patch p, int inRadius, boolean includeThisPatch) {

	int desiredLength = nbOfOfPatchesInRadius(p.x, p.y, inRadius);
	synchronized (p.neighbors) {
	    if (p.neighbors.size() < desiredLength) {
		// int startIndex = desiredLength - nbOfNeighborsOfPatchInRadius(inRadius, width, height);
		int startRadius = inRadius;
		while (nbOfNeighborsOfPatchInRadius(startRadius, width, height) != p.neighbors.size() && startRadius > 1) {
		    startRadius--;
		    // startIndex -= nbOfNeighborsOfPatchInRadius(startRadius, width, height);
		}
		for (; startRadius <= inRadius && nbOfNeighborsOfPatchInRadius(p.x, p.y, startRadius) > nbOfNeighborsOfPatchInRadius(p.x, p.y, startRadius - 1); startRadius++) {
		    Collection<Patch> tmp = new HashSet<>();
		    for (int u = -startRadius; u <= startRadius; u++) {
			for (int v = -startRadius; v <= startRadius; v++) {
			    if (Math.abs(u) < width && Math.abs(v) < height && (Math.abs(u) == startRadius || Math.abs(v) == startRadius)) {
				tmp.add(getPatch(p.x + u, p.y + v));
			    }
			}
		    }
		    p.neighbors.addAll(tmp);
		    p.neighbors.trimToSize();
		}
	    }
	    return p.neighbors.subList(includeThisPatch ? 0 : 1, desiredLength);
	}
    }

    public Patch getPatch(int i, int j) {
	return patchGrid[compute1DIndex(normalizeCoordinate(i, width), normalizeCoordinate(j, height))];
    }

    final int nbOfNeighborsOfPatchInRadius(int x, int y, int radius) {
	return nbOfOfPatchesInRadius(x, y, radius) - 1;
    }

    final int nbOfOfPatchesInRadius(int x, int y, int radius) {
	if (radius < 1)
	    return 1;
	if (torusMode) {
	    radius = radius * 2 + 1;
	    return Math.min(width, radius) * Math.min(height, radius);
	}
	else {
	    int distanceEast = ((x + radius) >= width) ? width - x - 1 : radius;
	    int distanceWest = ((x - radius) < 0) ? x : radius;
	    int distanceNorth = ((y + radius) >= height) ? height - y - 1 : radius;
	    int distanceSouth = ((y - radius) < 0) ? y : radius;
	    return (distanceEast + distanceWest + 1) * (distanceNorth + distanceSouth + 1);
	}
    }

    final int normalizeCoordinate(int a, final int dimensionThickness) {
	if (torusMode) {
	    a %= dimensionThickness;
	    return a < 0 ? a + dimensionThickness : a;
	}
	if (a >= dimensionThickness)
	    return dimensionThickness - 1;
	else
	    return a < 0 ? 0 : a;
    }

    final double normalizeCoordinate(double a, final int dimensionThickness) {
	if (torusMode) {
	    a %= dimensionThickness;
	    return a < 0 ? a + dimensionThickness : a;
	}
	if (a >= dimensionThickness)
	    return dimensionThickness - .01;
	else
	    return a < 0 ? 0 : a;
    }

    /**
     * Returns the normalized value of x, so that it is inside the environment's boundaries
     * 
     * @param x
     *            x-coordinate
     * @return the normalized value
     */
    final public double normalizeX(double x) {
	return normalizeCoordinate(x, width);
    }

    /**
     * Returns the normalized value of y, so that it is inside the environment's boundaries
     * 
     * @param y
     *            y-coordinate
     * @return the normalized value
     */
    final public double normalizeY(double y) {
	return normalizeCoordinate(y, height);
    }

    /**
     * Returns the normalized value of x, so that it is inside the environment's boundaries
     * 
     * @param x
     *            x-coordinate
     * @return the normalized value
     */
    final public double normalizeX(int x) {
	return normalizeCoordinate(x, width);
    }

    /**
     * Returns the normalized value of y, so that it is inside the environment's boundaries
     * 
     * @param y
     *            y-coordinate
     * @return the normalized value
     */
    final public double normalizeY(int y) {
	return normalizeCoordinate(y, height);
    }

    /**
     * @return the torusMode
     */
    public boolean isTorusModeOn() {
	return torusMode;
    }

    /**
     * @param torusMode
     *            the torusMode to set
     */
    public void setTorusMode(boolean torusMode) {
	this.torusMode = torusMode;
    }

    /**
     * @return the width
     */
    public int getWidth() {
	return width;
    }

    /**
     * @param width
     *            the width to set
     */
    public void setWidth(int width) {
	this.width = width;
    }

    /**
     * @return the height
     */
    public int getHeight() {
	return height;
    }

    /**
     * @param height
     *            the height to set
     */
    public void setHeight(int height) {
	this.height = height;
    }

    private int compute1DIndex(int u, int v) {
	return v * width + u;
    }

    public Patch[] getPatchGrid() {
	return patchGrid;
    }

    public Stream<Patch> getGridParallelStream() {
	return Arrays.stream(patchGrid).parallel();
    }

    public double getDistanceBetweenPatchCenters(Patch from, Patch to) {
	if (torusMode) {
	    return hypot(getAbscisseDistanceBetweenPatch(from, to), getOrdinnateDistanceBetweenPatch(from, to));
	}
	else {
	    return getDistanceBetweenPatchWithoutTorusMode(from, to);
	}
    }

    public double getDistanceBetweenPatchWithoutTorusMode(Patch from, Patch to) {
	return hypot(from.x - to.x, from.y - to.y);
    }

    private int getAbscisseDistanceBetweenPatch(Patch from, Patch to) {
	final int smallestX, biggest;
	if (from.x > to.x) {
	    biggest = from.x;
	    smallestX = to.x;
	}
	else {
	    biggest = to.x;
	    smallestX = from.x;
	}
	return min(biggest - smallestX, smallestX + getWidth() - biggest);
    }

    private int getOrdinnateDistanceBetweenPatch(Patch from, Patch to) {
	final int smallestY, biggest;
	if (from.y > to.y) {
	    biggest = from.y;
	    smallestY = to.y;
	}
	else {
	    biggest = to.y;
	    smallestY = from.y;
	}
	return min(biggest - smallestY, smallestY + getHeight() - biggest);
    }

    public double getDirectionFromTo(Patch from, Patch to) {
	return angleToPoint(relativeX(from, to), relativeY(from, to));
    }

    final private double relativeX(Patch from, Patch to) {// TODO facto and bench
	int distance = to.x - from.x;
	if (torusMode && abs(distance) > getWidth() / 2) {
	    return from.x - to.x;
	}
	return distance;
    }

    final private double relativeY(Patch from, Patch to) {// TODO facto and bench
	int distance = to.y - from.y;
	if (torusMode && abs(distance) > getHeight() / 2) {
	    return from.y - to.y;
	}
	return distance;
    }

    /**
     * return the direction to a location which is i,j units away
     * 
     * @param u
     *            the x-coordinate of the direction vector
     * @param v
     *            the y-coordinate of the direction vector
     * @return the heading towards a relative location
     */
    private double angleToPoint(final double u, final double v) {
	if (u == 0 && v == 0)
	    throw new ArithmeticException("directionAngleToPoint(0,0) makes no sense");
	if (u >= 0)
	    if (v > 0)
		return toDegrees(atan(v / u));
	    else
		return 360.0 + toDegrees(atan(v / u));
	else
	    return 180.0 + toDegrees(atan(v / u));
    }

}
