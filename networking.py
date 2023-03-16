import struct
import socket
import time


def recv_msg(c):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(4, c)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        # print("mes len", ms
    # glen)
    #print("length=", msglen)
    return recvall(msglen, c)


def recvall(n, c):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = c.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    #print(len(data))
    return data


def send_msg(c, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    c.sendall(msg)


def connect(host, port):
    while True:
        try:
            urb_channel = socket.socket()
            urb_channel.connect((host, port))
            return urb_channel
        except:
            time.sleep(1)


def server_discovery(port):
    server_channel = []
    while True:
        try:
            dispatcher_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dispatcher_socket.bind(("", port))
            break
        except:
            port += 1
            print(port)
    dispatcher_socket.listen(1)
    print("start to listening:{} -> server......".format(port))

    while True:
        try:
            channel, server_addr = dispatcher_socket.accept()
            server_channel.append(channel)
            print("server connected, start to communicate with server [{}]".format(server_addr))
            break
        except:
            break
    return server_channel[0]
