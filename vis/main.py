from OpenGL.GL import *
from OpenGL.GLUT import *


def draw_square():
    # We have to declare the points in this sequence: bottom left, bottom right, top right, top left
    glBegin(GL_QUADS)  # Begin the sketch
    glVertex2f(100, 100)  # Coordinates for the bottom left point
    glVertex2f(200, 100)  # Coordinates for the bottom right point
    glVertex2f(200, 200)  # Coordinates for the top right point
    glVertex2f(100, 200)  # Coordinates for the top left point
    glEnd()  # Mark the end of drawing


def setup_projection():
    glViewport(0, 0, 500, 500)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 500, 0, 500, 0, 1)


def show_screen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Remove everything from screen (i.e. displays all white)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()  # Reset all graphic/shape's position

    glColor3f(1.0, 0.0, 3.0)
    draw_square()  # Draw a square using our function

    glutSwapBuffers()


if __name__ == '__main__':
    glutInit()  # Initialize a glut instance which will allow us to customize our window
    glutInitDisplayMode(GLUT_RGBA)  # Set the display mode to be colored
    glutInitWindowSize(500, 500)    # Set the width and height of your window
    glutInitWindowPosition(0, 0)    # Set the position at which this windows should appear
    wind = glutCreateWindow("Visualization - Smoothed Particle Hydrodynamics")  # Give your window a title

    setup_projection()

    glutDisplayFunc(show_screen)  # Tell OpenGL to call the showScreen method continuously
    glutIdleFunc(show_screen)     # Draw any graphics or shapes in the showScreen function at all times
    glutMainLoop()  # Keeps the window created above displaying/running in a loop
