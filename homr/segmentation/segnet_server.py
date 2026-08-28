# This introduces a small server that can be shared across python instances to speed up inference.
# It is mainly meant to be used as a dev util. For now only in the datagen.

import os
from multiprocessing.connection import Listener
import threading

from homr.segmentation.inference_segnet import extract

SOCK_PATH = "/tmp/segnet_server.sock"

_lock = threading.Lock()

def handle(conn):
    try:
        while True:
            try:
                inputs = conn.recv()
            except EOFError:
                break
            with _lock:
                outputs = extract(*inputs)
            conn.send(outputs)
    finally:
        conn.close()

def serve():
    if os.path.exists(SOCK_PATH):
        print("remove sock path")
        os.remove(SOCK_PATH)
    with Listener(SOCK_PATH, family="AF_UNIX") as listener:
        while True:
            conn = listener.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

if __name__ == "__main__":
    serve()