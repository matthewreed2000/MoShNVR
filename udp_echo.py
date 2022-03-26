import socket

# Set up constants
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 20001
BUFFER_SIZE = 1024
GET_MESSAGE = str.encode("GET")

# Set up UDP socket
udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

print("UDP echo client listening...\n")

# Constantly make requests and listen
addr = (LOCAL_IP, LOCAL_PORT)
while(True):
    udp_socket.sendto(GET_MESSAGE, addr)
    message, addr = udp_socket.recvfrom(BUFFER_SIZE)
    print(message)

# Sources
# [1] https://pythontic.com/modules/socket/udp-client-server-example