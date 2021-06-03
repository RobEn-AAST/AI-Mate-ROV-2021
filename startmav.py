import pymavlink
from pymavlink import mavutil
import time

attia = mavutil.mavlink_connection("udpcast:192.168.2.1:14551")
karim = mavutil.mavlink_connection("udpcast:192.168.2.5:14552")

def wait_conn(master):
    """
    Sends a ping to stabilish the UDP communication and awaits for a response
    """
    msg = None
    while not msg:
        master.mav.ping_send(
            int(time.time() * 1e6), # Unix time in microseconds
            0, # Ping number
            0, # Request ping of all systems
            0 # Request ping of all components
        )
        msg = master.recv_match()
        time.sleep(0.5)

wait_conn(attia)
wait_conn(karim)