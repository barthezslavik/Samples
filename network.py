# Import the necessary libraries and modules
import pygame
import socket

# Initialize Pygame and create the game window
pygame.init()
window = pygame.display.set_mode((800, 600))

# Create a UDP socket for the Pygame application
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the local IP address and port
sock.bind(("192.168.0.129", 10000))  # Use a different IP address here

# Set the socket to non-blocking mode
sock.setblocking(False)

# Create a dictionary to store the state of other Pygame instances
instance_states = {}

# Run the main game loop
while True:
    # Check for events and handle them
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break

    # Update the state of the current Pygame instance
    state = {"x": 100, "y": 100}

    # Send the state of the current Pygame instance to other instances
    sock.sendto(str(state).encode(), ("192.168.0.129", 10000))

    # Receive data from other Pygame instances
    try:
        data, addr = sock.recvfrom(1024)
        instance_states[addr] = eval(data.decode())
    except BlockingIOError:
        pass
