# Import and initialize the pygame library
import pygame
import numpy as np

pygame.init()


class Screen:
    width = None
    height = None
    screen = None

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def ratio(self):
        return self.width / self.height

    def draw(self, buffer):
        if buffer.shape != (self.width, self.height, 3):
            raise Exception("buffer shape incorrect")

        # Set up the drawing window
        self.screen = pygame.display.set_mode([self.width, self.height])
        buffer = np.fliplr(buffer)
        pygame.pixelcopy.array_to_surface(self.screen, buffer)

    def show(self):
        # Run until the user asks to quit
        running = True
        while running:

            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Flip the display
            pygame.display.flip()

        # Done! Time to quit.
        pygame.quit()

    def screen_to_pixel(self, p):
        x = int(((p[0] + 1) * self.width) / 2)
        y = int(((p[2] + 1) * self.height) / 2)
        retVal = np.array([x, y])
        return retVal

    def pixel_to_screen(self, x, y):
        x_screen = (2 * (x + 0.5) / self.width) - 1.0
        y_screen = 0.0
        z_screen = (2 * (y + 0.5) / self.height) - 1.0
        return np.array([x_screen, y_screen, z_screen])
