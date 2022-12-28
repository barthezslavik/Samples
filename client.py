# Import the socket module
import socket

# Create a socket object
s = socket.socket()

# Connect to the socket server
s.connect(("localhost", 10000))

# Send a message to the server
s.send("Hello, this is the client.".encode())

# Receive a response from the server
response = s.recv(1024)

# Print the response
print(response.decode())

# Close the connection
s.close()
