package turtlekit.kernel;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import madkit.kernel.Madkit;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;

@Command(name = "TutleKit4", version = "4.0", description = "Lightweight ALife platform: Multi-Agent Systems as artificial organizations")
public class TutleKit4 extends Madkit {
	
	@Mixin
	TKCommandLine tkOptions = new TKCommandLine();
	
	public static final Logger TK_LOGGER = Logger.getLogger("[MDK] ");

	private Logger tkLogger;

	public TutleKit4(String... args) {
		super(args);
		initLogging();
	}

	public static void main(String[] args) {
		List<String> arguments = new ArrayList<>(List.of(args));
		if (! Utils.argsContainsLauncherClass(arguments)) {
			arguments.addAll(List.of("-la", TurtleKit.class.getName()));
		}
		new TutleKit4(arguments.toArray(new String[arguments.size()]));
	}

	private void initLogging() {
		tkLogger = Logger.getLogger("[TK] ");
		tkLogger.setParent(TK_LOGGER);
		tkLogger.setLevel(tkOptions.tkLogLevel);
	}

}
