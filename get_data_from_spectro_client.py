#!/usr/bin/python3

import socket

IP = socket.gethostbyname(socket.gethostname())
IP = "192.168.1.58"

PORT = 5566
ADDR = (IP, PORT)
FORMAT = "utf-8"
SIZE = 1024

def main():
    """ Staring a TCP socket. """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    """ Connecting to the server. """
    client.connect(ADDR)
    """ Opening and reading the file data. """
    file = open("test.txt", "r")
    data = file.read()
    """ Sending the filename to the server. """
    client.send("yt.txt".encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)
    print(f"[SERVER]: {msg}")
    """ Sending the file data to the server. """
    client.send(data.encode(FORMAT))
    msg = client.recv(SIZE).decode(FORMAT)
    print(f"[SERVER]: {msg}")
    """ Closing the file. """
    file.close()
    """ Closing the connection from the server. """
    client.close()

if __name__ == "__main__":
    main()