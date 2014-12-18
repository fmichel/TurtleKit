package turtlekit.viewer.viewer3D.engine.tools;

import java.awt.Font;
import java.text.DecimalFormat;

import javax.media.opengl.GLDrawable;
import javax.media.opengl.GLException;

import com.jogamp.opengl.util.awt.TextRenderer;

public class PrintText {
	public static final int UPPER_LEFT  = 1;
	public static final int UPPER_RIGHT = 2;
	public static final int LOWER_LEFT  = 3;
	public static final int LOWER_RIGHT = 4;

	private int textLocation = LOWER_LEFT;
	private final GLDrawable drawable;
	private final TextRenderer renderer;
	private final DecimalFormat format = new DecimalFormat("#####.00");
	private int frameCount;
	private String printText;
	private int printWidth;
	private int printHeight;
	private int printOffset;
	
	public PrintText(GLDrawable drawable, int textSize) throws GLException {
		this(drawable, new Font("SansSerif", Font.BOLD, textSize));
		frameCount = 0;
	}
	
	public PrintText(GLDrawable drawable, Font font) throws GLException {
		this(drawable, font, true, true);
	}
	
	public PrintText(GLDrawable drawable, Font font, boolean antialiased, boolean useFractionalMetrics) throws GLException {
		this.drawable = drawable;
		renderer = new TextRenderer(font, antialiased, useFractionalMetrics);
	}
	
	public int getTextLocation() {
		return textLocation;
	}
	
	public void setTextLocation(int textLocation) {
		if (textLocation < UPPER_LEFT || textLocation > LOWER_RIGHT) {
			throw new IllegalArgumentException("textLocation");
		}
		this.textLocation = textLocation;
	}
	
	public void setColor(float r, float g, float b, float a) throws GLException {
		renderer.setColor(r, g, b, a);
	}
	
	// Réactiver le if pour optimiser les performances : l'affichage du texte ne se fera plus à chaque frame mais tous les X frames
	public void draw(float data, String title) {

		printText = title + " : " + format.format(data);

	    if (printText != null) {
	    	renderer.beginRendering(drawable.getSurfaceWidth(), drawable.getSurfaceHeight());
	    	// Figure out the location at which to draw the text
	    	int x = 0;
	    	int y = 0;
	    	switch (textLocation) {
	    		case UPPER_LEFT:
	    			x = printOffset;
	    			y = drawable.getSurfaceHeight() - printHeight - printOffset;
	    			break;

	    		case UPPER_RIGHT:
			        x = drawable.getSurfaceWidth() - printWidth - printOffset;
			        y = drawable.getSurfaceHeight() - printHeight - printOffset;
			        break;

	    		case LOWER_LEFT:
		            x = printOffset;
		            y = printOffset;
		            break;

	    		case LOWER_RIGHT:
		            x = drawable.getSurfaceWidth() - printWidth - printOffset;
		            y = printOffset;
		            break;
	     	}

	      renderer.draw(printText, x, y);
	      renderer.endRendering();
	    }
	  }
}
