import socket
import json
import matplotlib.pyplot as plt
import time

# Set up constants
LOCAL_IP = "127.0.0.1"
LOCAL_PORT = 20001
BUFFER_SIZE = 2048
GET_MESSAGE = str.encode("GET")

def main():
    # Set up UDP socket
    udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    udp_socket.settimeout(1)

    print("UDP echo client listening...\n")

    # Constantly make requests and listen
    addr = (LOCAL_IP, LOCAL_PORT)

    # Make initial request
    data = None
    while data is None:
        try:
            udp_socket.sendto(GET_MESSAGE, addr)
            message, addr = udp_socket.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            print("Server disconnected...")
            return
            
        message_str = message.decode()
        data = json.loads(message_str)

    plt.ion()

    figure, ax = plt.subplots(figsize=(10,8))
    bars = ax.barh(list(data.keys()), list(data.values()), align='center')

    while(True):
        try:
            udp_socket.sendto(GET_MESSAGE, addr)
            message, addr = udp_socket.recvfrom(BUFFER_SIZE)
        except socket.timeout:
            print("Server disconnected...")
            break
        
        message_str = message.decode()
        data = json.loads(message_str)

        bars.remove()
        bars = ax.barh(list(data.keys()), list(data.values()), color='blue', align='center')
        figure.canvas.draw()
        figure.canvas.flush_events()

if __name__ == '__main__':
    main()

# Sources
# [1] https://pythontic.com/modules/socket/udp-client-server-example
# [2] https://matplotlib.org/stable/gallery/lines_bars_and_markers/barh.html
# [3] https://stackoverflow.com/questions/45185970/how-to-update-barchart-in-matplotlib
# [4] https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html