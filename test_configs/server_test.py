import socket

def start_server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', port))
    s.listen(1)
    print(f'Server started. Listening on port {port}...')
    while True:
        conn, addr = s.accept()
        print(f'Accepted connection from {addr}.')
        data = conn.recv(1024)
        print(f'Received data: {data}')
        conn.sendall(b'Thank you for connecting.')
        conn.close()

if __name__ == "__main__":
    start_server(6435)
