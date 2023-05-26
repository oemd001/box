import socket

def connect_to_server(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.sendall(b'Hello, server!')
    data = s.recv(1024)
    print(f'Received data: {data}')
    s.close()

if __name__ == "__main__":
    connect_to_server('ec2-13-56-161-92.us-west-1.compute.amazonaws.com', 6436)
