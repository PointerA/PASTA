import pickle, struct, socket
import time
from datetime import datetime, timedelta

SERVER_ADDR= '192.168.70.21'   # When running in a real distributed setting, change to the server's IP address
SERVER_PORT = 8888
TIME_OUT = 666
CLIENT_ADDRS= [('192.168.70.12', 1111), ('192.168.70.12', 2222), ('192.168.70.12', 3333), ('192.168.70.13', 4444), ('192.168.70.13', 5555), ('192.168.70.14', 6666),('192.168.70.14', 7777)]
# def send_thread(sock, msg_pickle):
#     sock.sendall(struct.pack(">I", len(msg_pickle)))
#     sock.sendall(msg_pickle)

# def send_msg(sock, msg):
#     msg_pickle = pickle.dumps(msg)
#     #print(len(msg_pickle))
#     thread = threading.Thread(target=send_thread, args=(sock, msg_pickle,))
#     thread.start()
#     #print(msg, 'sent to', sock.getpeername())
#     return len(msg_pickle)*8/1024/1024

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    #print(len(msg_pickle))
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    #print(msg, 'sent to', sock.getpeername())
    return len(msg_pickle)*8/1024/1024

def recv_msg(sock, expect_msg_type=None):
    msg_len_bytes = b''
    while len(msg_len_bytes) < 4:
        msg_len_bytes += sock.recv(4 - len(msg_len_bytes))
    msg_len = struct.unpack(">I", msg_len_bytes)[0]

    received = b''
    while len(received) < msg_len:
        chunk = sock.recv(msg_len - len(received))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        received += chunk

    mb = len(received)*8/1024/1024
    msg = pickle.loads(received)
    #print(msg, 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg, mb

def timer_function(start_minutes=10):
    # now
    current_time = datetime.now()
    
    # next
    current_minute = current_time.minute
    delta = start_minutes - current_minute % 10
    delta = delta if delta > 0 else delta+10
    
    current_time = datetime.now()
    
    next_time = current_time + timedelta(minutes=delta)
    next_time = next_time.replace(second=0, microsecond=0)
    
    # wait
    wait_time = (next_time - current_time).total_seconds()
    time.sleep(wait_time)