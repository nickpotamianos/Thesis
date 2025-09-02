#swarm_net/udp.py
from __future__ import annotations
import socket, struct
from dataclasses import dataclass
from typing import Dict, Any, Optional
import msgpack

@dataclass
class UdpGroup:
    mcast_ip: str = "239.0.0.1"
    port: int = 5001
    ttl: int = 1

def make_tx(g: UdpGroup):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, g.ttl)
    addr = (g.mcast_ip, g.port)
    def send(obj: Dict[str, Any]):
        buf = msgpack.packb(obj, use_bin_type=True)
        sock.sendto(buf, addr)
    return send

def make_rx(g: UdpGroup, iface_ip: str = "0.0.0.0"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((iface_ip, g.port))
    mreq = struct.pack("4sl", socket.inet_aton(g.mcast_ip), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    def recv():
        data, _ = sock.recvfrom(65535)
        return msgpack.unpackb(data, raw=False)
    return recv

# NEW: non-blocking receiver (returns None when no packet)
def make_rx_nb(g: UdpGroup, iface_ip: str = "0.0.0.0"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((iface_ip, g.port))
    mreq = struct.pack("4sl", socket.inet_aton(g.mcast_ip), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.setblocking(False)
    def try_recv() -> Optional[Dict[str, Any]]:
        try:
            data, _ = sock.recvfrom(65535)
            return msgpack.unpackb(data, raw=False)
        except (BlockingIOError, InterruptedError):
            return None
    return try_recv
