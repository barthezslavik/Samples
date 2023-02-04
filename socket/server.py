# Import the socket module
import socket

# Create a socket object
s = socket.socket()

# Bind the socket to a local address and port
s.bind(("localhost", 10000))

# Listen for incoming connections
s.listen()

# Accept the incoming connection
connection, address = s.accept()

# Receive a message from the client
message = connection.recv(1024)

# Print the received message
print(message.decode())

# Send a response to the client
connection.send("Hello, this is the server.".encode())

# Close the connection
connection.close()