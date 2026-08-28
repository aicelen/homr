# client.py
from multiprocessing.connection import Client
from homr.segmentation.segnet_server import SOCK_PATH

class SegnetClient:
    def __init__(self):    
        self.conn = Client(SOCK_PATH, family="AF_UNIX")

    def extract_server(self, inputs: tuple) -> list:
        self.conn.send(inputs)
        return self.conn.recv()

    def close(self):
        # when the process is shutting down
        self.conn.close()
