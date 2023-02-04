class Game:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Create a game window with the specified size
        self.screen = pygame.display.set_mode((1024, 768))

        # Load the texture image
        self.texture = pygame.image.load("texture.png")

        # Create the surfaces and rectangles for the control, main scene, and minimap
        self.create_control_surface()
        self.create_main_surface()
        self.create_minimap_surface()

        # Create a unit object
        self.unit = Unit()

        # Create a flag to track whether the player is currently clicking on the minimap
        self.is_clicking = False

        # Create a rectangle for the game field
        self.field_rect = pygame.Rect(0, 0, 800, 768)

    def update(self):
        # Update the game state
        self.unit.update()

        # Clear the surfaces
        self.control_surface.fill((255, 255, 255))  
        self.minimap_surface.fill((226, 135, 67))  
        self.main_surface.fill((30,129,176))

        # Check if the player is currently clicking on the minimap
        if self.is_clicking:
            # Get the current position of the mouse cursor
            cursor_pos = pygame.mouse.get_pos()

            # Check if the cursor is inside the minimap rectangle
            if self.minimap_rect.collidepoint(cursor_pos):
                # Calculate the position on the main screen corresponding to the cursor position on the minimap
                screen_pos = (cursor_pos[0] * self.main_rect.width / self.minimap_rect.width,
                              cursor_pos[1] * self.main_rect.height / self.minimap_rect.height)

                # Update the position of the unit on the main screen
                self.unit.rect.x = screen_pos[0] - self.unit.rect.width / 2
                self.unit.rect.y = screen_pos[1] - self.unit.rect.height / 2

        # Check if the player is pressing the up arrow key
        if pygame.key.get_pressed()[pygame.K_UP]:
            # Move the game field rectangle up
            self.field_rect.move_ip(0, -10)

        # Check if the player is pressing the down arrow key
        if pygame.key.get_pressed()[pygame.K_DOWN]:
            # Move the game field rectangle down
            self.field_rect.move_ip(0, 10)

        # Check if the player is pressing the left arrow key
        if pygame.key.get_pressed()[pygame.K_LEFT]:
            # Move the game field rectangle left
            self.field_rect.move_ip(-10, 0)

        # Check if the player is pressing the right arrow key
        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            # Move the game field rectangle right
            self.field_rect.move_ip(10, 0)