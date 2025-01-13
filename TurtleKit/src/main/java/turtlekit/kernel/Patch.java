package turtlekit.kernel;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
import java.util.stream.Stream;

import javafx.scene.paint.Color;

/**
 * This class represents the default cell model of TurtleKit
 *
 * @author Fabien Michel
 *
 */
public class Patch {

	// manage another list for caching read -> UNEFFICIENT IDEA
	private final ArrayList<Turtle<?>> turtlesHere = new ArrayList<>();
	private Color color;
	public int x;
	public int y;
	private Map<String, Object> marks;
	static TKGridModel<?> gridModel;
	ArrayList<Patch> neighbors;

	/**
	 * Gets the gridModel the patch is in
	 *
	 * @return the gridModel the patch is in
	 */
	public TKGridModel<?> getGridModel() {
		return gridModel;
	}

	/**
	 * Creates a patch having a black color
	 */
	public Patch() {
		color = Color.BLACK;
	}

	void setCoordinates(int x, int y) {
		this.x = x;
		this.y = y;
	}

	public void ensureCapacity(int capacity) {
		turtlesHere.ensureCapacity(capacity);
	}

	/**
	 * Checks if there is at least one turtle on this patch
	 *
	 * @return <code>true</code> if there is no turtle
	 */
	public boolean isEmpty() {
		return turtlesHere.isEmpty();
	}

	public List<Patch> getNeighbors(int inRadius, boolean includeThisPatch) {
		return gridModel.getNeighborsOf(this, inRadius, includeThisPatch);
	}

	/**
	 * Drops a labeled object on the patch
	 *
	 * @param markName mark name
	 * @param value    mark itself, can be any java object
	 */
	public final void dropObject(String markName, Object value) {
		if (marks == null)
			marks = new ConcurrentHashMap<>(1);
		marks.put(markName, value);
	}

	/**
	 * gets an labeled object previously deposed on the patch
	 *
	 * @return the corresponding java object which thus is removed from the patch,
	 *         or <code>null</code> if not present
	 */
	public final Object getMark(String markName) {
		try {
			return marks.remove(markName);
		} catch (NullPointerException e) {
			return null;
		}
	}

	/** tests if the corresponding mark is present on the patch (true or false) */
	public final boolean isObjectPresent(String markName) {
		try {
			return marks.containsKey(markName);
		} catch (NullPointerException e) {
			return false;
		}
	}

	public <T extends Turtle<?>> List<T> getTurtles(int inRadius, boolean includeThisPatch,
			Class<T> turtleType) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtles(turtleType));
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getTurtlesWithRole(int inRadius, boolean includeThisPatch, String role,
			Class<T> turtleType) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtlesWithRole(role, turtleType));
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getTurtlesWithRole(int inRadius, boolean includeThisPatch,
			String role) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtlesWithRole(role));
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getTurtles(int inRadius, boolean includeThisPatch) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtles());
		}
		return l;
	}

	public <T extends Turtle<?>> List<T> getTurtles(int inRadius, boolean includeThisPatch,
			Predicate<? super Turtle<?>> test) {
		List<T> l = new ArrayList<>();
		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
			l.addAll(p.getTurtles(test));
		}
		return l;
	}

	/**
	 * Gets all the turtles on this patch according to their type.
	 *
	 * @param turtleType
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getTurtles(Class<T> turtleType) {
		return (List<T>) getTurtles(t -> turtleType.isAssignableFrom(t.getClass()));
	}

	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getTurtles() {
		synchronized (turtlesHere) {
			return (List<T>) new ArrayList<>(turtlesHere);
//			return (List<T>) Collections.unmodifiableList(turtlesHere);
		}
	}

	@SuppressWarnings("unchecked")
	public <A extends Turtle<?>> Stream<A> turtleStream() {
		return (Stream<A>) getTurtles().stream();
	}

	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getTurtles(Predicate<? super Turtle<?>> filter) {
		return (List<T>) getTurtles().stream().filter(filter).toList();
	}

	/**
	 * Gets the nearest turtle of type T in the vicinity of the patch.
	 *
	 * @param inRadius         the range of the search
	 * @param includeThisPatch for the search
	 * @param turtleType       the type of the turtle as a {@link Class}
	 * @return the corresponding turtle or <code>null</code> if no such turtle is
	 *         found
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> T getNearestTurtle(int inRadius, boolean includeThisPatch,
			Class<T> turtleType) {
		for (final Patch p : getNeighbors(inRadius, includeThisPatch)) {
			final List<Turtle<?>> turtles = p.getTurtles();
			for (final Turtle<?> t : turtles) {
				if (turtleType.isAssignableFrom(t.getClass())) {
					return (T) t;
				}
			}
		}
		return null;
	}

	/**
	 * Gets the nearest turtle of type T in the vicinity of the patch.
	 *
	 * @param inRadius         the range of the search
	 * @param includeThisPatch for the search
	 * @return the corresponding turtle or <code>null</code> if there is no turtle
	 *         around
	 *
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> T getNearestTurtle(int inRadius, boolean includeThisPatch) {
		return (T) getNeighbors(inRadius, includeThisPatch).stream()
				.flatMap(p -> Stream.of(getTurtles()))
				.findFirst()
				.orElseGet(() -> null);
//		for (Patch p : getNeighbors(inRadius, includeThisPatch)) {
//			for (Turtle<?> t : p.getTurtles()) {
//				return t;
//			}
//		}
//		return null;
	}

	/**
	 * Get all the turtles on this patch according to their type and role.
	 *
	 * @param role
	 * @param turtleType
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getTurtlesWithRole(String role, Class<T> turtleType) {
		return (List<T>) getTurtles(t -> turtleType.isAssignableFrom(t.getClass()) && t.isPlayingRole(role));
	}

	/**
	 * Get all the turtles which are on this patch and having this role.
	 *
	 * @param role
	 * @return a list of turtles which could be empty
	 */
	@SuppressWarnings("unchecked")
	public <T extends Turtle<?>> List<T> getTurtlesWithRole(String role) {
		return (List<T>) getTurtles(t -> t.isPlayingRole(role));
	}

	public int countTurtles() {
		return turtlesHere.size();
	}

	public Color getColor() {
		return color;
	}

	public void setColor(Color c) {
		color = c;
	}

	final void removeAgent(Turtle<?> a) {
		synchronized (turtlesHere) {
			turtlesHere.remove(a);
		}
	}

	final void addAgent(Turtle<?> a) {
		synchronized (turtlesHere) {
			turtlesHere.add(a);
		}
		a.setPosition(this);
	}

	/**
	 * Could be used to define the dynamics of a patch, which could trigger in the
	 * environment dynamics
	 */
	protected void update() {

	}

	public double getDirectionToPatch(Patch p) {
		return getGridModel().getDirectionFromTo(this, p);
	}

	public double getDistanceToPatch(Patch p) {
		return getGridModel().getDistanceBetweenPatchCenters(this, p);
	}

	@Override
	public String toString() {
		return "P(" + x + "," + y + ")";
	}

	public void dropPheromone(String name, float quantity, Float... parameters) {
		gridModel.getEnvironment().getPheromone(name).incValue(x, y, quantity);
	}

	static void setGridModel(TKGridModel<?> tkGridModel) {
		gridModel = tkGridModel;
	}

	public Color swapColor(Color color2) {
		Color c = color;
		setColor(color2);
		return c;
	}

}
