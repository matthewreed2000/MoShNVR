import socket

# Set up constants
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 20001
BUFFER_SIZE = 1024

# Set up UDP socket
udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
udp_socket.bind((LOCAL_IP, LOCAL_PORT))

while(True):
    # Get request from client
    _, addr = udp_socket.recvfrom(BUFFER_SIZE)

    # Send reply to client
    ret_message = str.encode(str(count))
    udp_socket.sendto(ret_message, addr)

    # Change the message
    count += 1

# Sources
# [1] https://pythontic.com/modules/socket/udp-client-server-example
# [2] https://quick-adviser.com/can-multiple-udp-sockets-on-same-port/