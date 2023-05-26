import socket
import pickle

# Define the IP address and port to listen on
master_ip = 'IP_ADDRESS_OF_MASTER'
master_port = 12345

def process_image(image_path, slaves):
    # Send the image path to each slave for processing
    for slave in slaves:
        slave_socket = slave['socket']
        slave_socket.sendall(f'image:{image_path}'.encode())

    # Receive predictions from the slaves
    predictions = []
    for slave in slaves:
        slave_socket = slave['socket']
        response = slave_socket.recv(4096)
        prediction = pickle.loads(response)
        predictions.append(prediction)

    # Aggregate and process the predictions
    # Replace this with your actual aggregation and processing logic
    aggregated_prediction = process_predictions(predictions)
    print("Aggregated Prediction:", aggregated_prediction)

def run_master(slaves):
    # Start listening for incoming connections
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((master_ip, master_port))
        s.listen()

        print("Master is listening for connections.")

        while True:
            conn, addr = s.accept()
            print("Slave connected:", addr)

            # Add the slave to the list of connected slaves
            slave = {'socket': conn, 'address': addr}
            slaves.append(slave)

            # Process an image when requested
            image_path = 'path_to_image.jpg'  # Replace with your actual image path
            process_image(image_path, slaves)

            # Clean up the connection after processing
            conn.close()

    print("Master process completed.")

if __name__ == '__main__':
    slaves = []
    run_master(slaves)
