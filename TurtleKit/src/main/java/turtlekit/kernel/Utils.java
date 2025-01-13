package turtlekit.kernel;

import java.util.List;

import madkit.kernel.MadkitClassLoader;
import net.jodah.typetools.TypeResolver;

public class Utils {

	public static String getClassFromMainStackTrace() {
		StackTraceElement element = null;
		for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
			if (stackTraceElement.getMethodName().equals("main")) {
				element = stackTraceElement;
				break;
			}
		}
		return element.getClassName();
	}

	static StackTraceElement getElementFromMainStackTrace() {
		StackTraceElement element = null;
		for (StackTraceElement stackTraceElement : new Throwable().getStackTrace()) {
			if (stackTraceElement.getMethodName().equals("main")) {
				element = stackTraceElement;
				break;
			}
		}
		return element;
	}

	static List<String> getPatchClassArgsFromClasses(Class<?> typeDef, String actual) {
		try {
			Class<?> actualClass = MadkitClassLoader.getLoader().loadClass(actual);
			Class<?> patchClass = TypeResolver.resolveRawArguments(typeDef, actualClass)[0];
			return List.of("--patchClass", patchClass.getName());
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	static boolean argsContainsLauncherClass(List<String> args) {
		for (String string : args) {
			try {
				Class<?> c = MadkitClassLoader.getLoader().loadClass(string);
				if(TurtleKit.class.isAssignableFrom(c)) {
					return true;
				}
			} catch (ClassNotFoundException e) {
			}
		}
		return false;
	}
}
