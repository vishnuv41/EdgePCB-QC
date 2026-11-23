"""
Local Network Status & Offline IP Discovery for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Enables:
- Local IP address discovery (Ethernet eth0, Wi-Fi wlan0)
- Local network Streamlit URL generation (e.g. http://192.168.1.50:8501)
- Laptop-to-Pi connection diagnostics
- 100% offline verification (zero external pings, zero DNS lookups)
"""

import socket
import psutil
from typing import List, Dict, Any


def get_local_interfaces() -> List[Dict[str, Any]]:
    """Discovers all non-loopback local network interfaces and their IPv4 addresses."""
    interfaces = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for iface_name, addr_list in addrs.items():
        if iface_name.lower().startswith("lo"):
            continue

        stat = stats.get(iface_name)
        is_up = stat.isup if stat else False

        for addr in addr_list:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                iface_type = "Ethernet"
                if any(k in iface_name.lower() for k in ["wlan", "wifi", "wi-fi", "wireless"]):
                    iface_type = "Wi-Fi"

                interfaces.append({
                    "interface": iface_name,
                    "type": iface_type,
                    "ip_address": addr.address,
                    "netmask": addr.netmask,
                    "is_up": is_up,
                    "url": f"http://{addr.address}:8501"
                })

    return interfaces


def get_primary_connection_url(port: int = 8501) -> str:
    """
    Returns the recommended local network URL for accessing the dashboard from a laptop.
    Prefers Ethernet over Wi-Fi.
    """
    ifaces = get_local_interfaces()
    for iface in ifaces:
        if iface["type"] == "Ethernet" and iface["is_up"]:
            return f"http://{iface['ip_address']}:{port}"

    for iface in ifaces:
        if iface["is_up"]:
            return f"http://{iface['ip_address']}:{port}"

    if ifaces:
        return f"http://{ifaces[0]['ip_address']}:{port}"

    return f"http://localhost:{port}"


def get_network_diagnostics(port: int = 8501) -> Dict[str, Any]:
    """Returns comprehensive network telemetry for dashboard display."""
    ifaces = get_local_interfaces()
    primary_url = get_primary_connection_url(port)
    hostname = socket.gethostname()

    primary_ip = "127.0.0.1"
    for iface in ifaces:
        if iface["is_up"]:
            primary_ip = iface["ip_address"]
            break
    if primary_ip == "127.0.0.1" and ifaces:
        primary_ip = ifaces[0]["ip_address"]

    return {
        "offline_mode": True,
        "primary_url": primary_url,
        "primary_ip": primary_ip,
        "hostname": hostname,
        "port": port,
        "localhost_url": f"http://localhost:{port}",
        "hostname_url": f"http://{hostname}:{port}",
        "interfaces": ifaces,
        "is_network_accessible": len(ifaces) > 0,
        "note": "EdgePCB-QC operates in pure air-gapped offline mode. All communication is confined to the local subnet."
    }
