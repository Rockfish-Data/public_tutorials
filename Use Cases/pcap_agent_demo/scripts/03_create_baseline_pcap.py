#!/usr/bin/env python3
"""
Baseline PCAP Dataset Generator (No Rockfish Dependencies)

This script creates a PCAP dataset from scratch using only basic Python libraries
(Faker, numpy, pandas). This serves as a comparison baseline generated without
any Rockfish APIs or templates.

Usage:
    python 03_create_baseline_pcap.py [--output OUTPUT_DIR] [--sessions N]

Example:
    python 03_create_baseline_pcap.py --output baseline_data --sessions 1000

Dependencies:
    - pandas
    - numpy
    - faker
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()


# =============================================================================
# SERVICE DEFINITIONS
# =============================================================================

SERVICES = {
    "HTTP": {"port": 80, "protocol": "TCP"},
    "HTTPS": {"port": 443, "protocol": "TCP"},
    "SSH": {"port": 22, "protocol": "TCP"},
    "FTP": {"port": 21, "protocol": "TCP"},
    "DNS": {"port": 53, "protocol": "UDP"},
    "SMTP": {"port": 25, "protocol": "TCP"},
    "IMAP": {"port": 143, "protocol": "TCP"},
    "MYSQL": {"port": 3306, "protocol": "TCP"},
    "POSTGRESQL": {"port": 5432, "protocol": "TCP"},
    "SMB": {"port": 445, "protocol": "TCP"},
    "RDP": {"port": 3389, "protocol": "TCP"},
    "REDIS": {"port": 6379, "protocol": "TCP"},
}

TCP_FLAGS = ["SYN", "SYN,ACK", "ACK", "PSH,ACK", "FIN,ACK", "RST"]


# =============================================================================
# IP ADDRESS GENERATION
# =============================================================================

def generate_ip_address(internal: bool = True) -> str:
    """Generate realistic IP addresses using Faker."""
    if internal:
        # Internal subnets
        subnet = random.choice(["10.1", "10.2", "192.168.1", "192.168.2"])
        return f"{subnet}.{fake.pyint(min_value=1, max_value=254)}.{fake.pyint(min_value=1, max_value=254)}"
    else:
        return fake.ipv4_public()


def generate_mac_address() -> str:
    """Generate MAC address using Faker."""
    return fake.mac_address()


# =============================================================================
# NORMAL SESSION GENERATORS
# =============================================================================

def generate_normal_session(
    session_id: str,
    service_name: str,
    start_time: datetime,
) -> list:
    """Generate a normal TCP session with realistic packet flow."""
    service = SERVICES[service_name]
    src_ip = generate_ip_address(internal=True)

    # External servers for web traffic, internal for others
    if service_name in ["HTTP", "HTTPS"]:
        dst_ip = generate_ip_address(internal=False)
    else:
        dst_ip = generate_ip_address(internal=True)

    src_port = fake.pyint(min_value=49152, max_value=65535)  # Ephemeral port
    dst_port = service["port"]
    src_mac = generate_mac_address()
    dst_mac = generate_mac_address()

    packets = []
    current_time = start_time

    # TCP Handshake
    handshake_flags = ["SYN", "SYN,ACK", "ACK"]
    for i, flag in enumerate(handshake_flags):
        is_server = flag == "SYN,ACK"
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": dst_ip if is_server else src_ip,
            "dst_ip": src_ip if is_server else dst_ip,
            "src_port": dst_port if is_server else src_port,
            "dst_port": src_port if is_server else dst_port,
            "src_mac": dst_mac if is_server else src_mac,
            "dst_mac": src_mac if is_server else dst_mac,
            "protocol": service["protocol"],
            "tcp_flags": flag,
            "tcp_state": ["SYN_SENT", "SYN_RECEIVED", "ESTABLISHED"][i],
            "packet_type": flag.replace("-", "_"),
            "packet_size": fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": service_name,
            "direction": "S2C" if is_server else "C2S",
            "packet_size_category": "control",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=50))

    # Data exchange (5-20 packets)
    n_data_packets = fake.pyint(min_value=5, max_value=20)
    for i in range(n_data_packets):
        is_client = i % 2 == 0
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip if is_client else dst_ip,
            "dst_ip": dst_ip if is_client else src_ip,
            "src_port": src_port if is_client else dst_port,
            "dst_port": dst_port if is_client else src_port,
            "src_mac": src_mac if is_client else dst_mac,
            "dst_mac": dst_mac if is_client else src_mac,
            "protocol": service["protocol"],
            "tcp_flags": "PSH,ACK",
            "tcp_state": "ESTABLISHED",
            "packet_type": "PSH_ACK_C2S" if is_client else "PSH_ACK_S2C",
            "packet_size": fake.pyint(min_value=100, max_value=1500),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": service_name,
            "direction": "C2S" if is_client else "S2C",
            "packet_size_category": "data",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=10, max_value=500))

    # TCP Teardown: FIN (C2S) → FIN,ACK (S2C) → ACK (C2S)
    teardown = [
        ("FIN", "FIN_WAIT", "FIN", False),
        ("FIN,ACK", "CLOSE_WAIT", "FIN_ACK", True),
        ("ACK", "TIME_WAIT", "ACK", False),
    ]
    for flag, state, pkt_type, is_server in teardown:
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": dst_ip if is_server else src_ip,
            "dst_ip": src_ip if is_server else dst_ip,
            "src_port": dst_port if is_server else src_port,
            "dst_port": src_port if is_server else dst_port,
            "src_mac": dst_mac if is_server else src_mac,
            "dst_mac": src_mac if is_server else dst_mac,
            "protocol": service["protocol"],
            "tcp_flags": flag,
            "tcp_state": state,
            "packet_type": pkt_type,
            "packet_size": fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": service_name,
            "direction": "S2C" if is_server else "C2S",
            "packet_size_category": "control",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=50))

    return packets


# =============================================================================
# ANOMALY GENERATORS
# =============================================================================

def generate_anomaly_port_scan(session_id: str, start_time: datetime) -> list:
    """Generate port scan pattern - many SYN packets to different ports, RST responses."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = generate_ip_address(internal=True)
    src_mac = generate_mac_address()
    dst_mac = generate_mac_address()
    packets = []
    current_time = start_time

    # Scan 20-50 ports rapidly
    n_ports = fake.pyint(min_value=20, max_value=50)
    scanned_ports = random.sample(range(1, 1024), n_ports)

    for port in scanned_ports:
        src_port = fake.pyint(min_value=49152, max_value=65535)

        # SYN packet
        packets.append({
            "session_id": f"{session_id}_port_{port}",
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": port,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "protocol": "TCP",
            "tcp_flags": "SYN",
            "tcp_state": "SYN_SENT",
            "packet_type": "SYN",
            "packet_size": fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=1024, max_value=4096),
            "service_name": "SCAN",
            "direction": "C2S",
            "packet_size_category": "control",
        })

        # RST response (port closed) 70% of the time
        if random.random() > 0.3:
            packets.append({
                "session_id": f"{session_id}_port_{port}",
                "packet_timestamp": (
                    current_time + timedelta(milliseconds=fake.pyint(min_value=1, max_value=10))
                ).isoformat(),
                "src_ip": dst_ip,
                "dst_ip": src_ip,
                "src_port": port,
                "dst_port": src_port,
                "src_mac": dst_mac,
                "dst_mac": src_mac,
                "protocol": "TCP",
                "tcp_flags": "RST",
                "tcp_state": "RESET",
                "packet_type": "RST",
                "packet_size": fake.pyint(min_value=40, max_value=60),
                "ttl": fake.pyint(min_value=60, max_value=128),
                "window_size": 0,
                "service_name": "SCAN",
                "direction": "S2C",
                "packet_size_category": "control",
            })

        current_time += timedelta(milliseconds=fake.pyint(min_value=5, max_value=50))

    return packets


def generate_anomaly_beaconing(session_id: str, start_time: datetime) -> list:
    """Generate C2 beaconing pattern - regular interval small packets."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = fake.ipv4_public()  # External C2 server
    src_mac = generate_mac_address()
    dst_mac = generate_mac_address()
    packets = []
    current_time = start_time

    # Beacon every 60 seconds (+/- small jitter) for 10-20 beacons
    beacon_interval = 60  # seconds
    n_beacons = fake.pyint(min_value=10, max_value=20)
    src_port = fake.pyint(min_value=49152, max_value=65535)

    for i in range(n_beacons):
        # Small beacon packet (check-in)
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": 443,  # Hiding in HTTPS
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "protocol": "TCP",
            "tcp_flags": "PSH,ACK",
            "tcp_state": "ESTABLISHED",
            "packet_type": "PSH_ACK_C2S",
            "packet_size": fake.pyint(min_value=50, max_value=100),  # Small, consistent
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": "HTTPS",
            "direction": "C2S",
            "packet_size_category": "data",
        })

        # Response
        packets.append({
            "session_id": session_id,
            "packet_timestamp": (
                current_time + timedelta(milliseconds=fake.pyint(min_value=100, max_value=500))
            ).isoformat(),
            "src_ip": dst_ip,
            "dst_ip": src_ip,
            "src_port": 443,
            "dst_port": src_port,
            "src_mac": dst_mac,
            "dst_mac": src_mac,
            "protocol": "TCP",
            "tcp_flags": "PSH,ACK",
            "tcp_state": "ESTABLISHED",
            "packet_type": "PSH_ACK_S2C",
            "packet_size": fake.pyint(min_value=50, max_value=150),
            "ttl": fake.pyint(min_value=40, max_value=64),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": "HTTPS",
            "direction": "S2C",
            "packet_size_category": "data",
        })

        # Add jitter to next beacon
        jitter = fake.pyint(min_value=-5, max_value=5)
        current_time += timedelta(seconds=beacon_interval + jitter)

    return packets


def generate_anomaly_large_upload(session_id: str, start_time: datetime) -> list:
    """Generate data exfiltration pattern - large upload to external server."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = fake.ipv4_public()  # External destination
    src_mac = generate_mac_address()
    dst_mac = generate_mac_address()
    packets = []
    current_time = start_time

    src_port = fake.pyint(min_value=49152, max_value=65535)
    dst_port = 443  # HTTPS

    # Handshake
    for flag, state in [("SYN", "SYN_SENT"), ("SYN,ACK", "SYN_RECEIVED"), ("ACK", "ESTABLISHED")]:
        is_server = flag == "SYN,ACK"
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": dst_ip if is_server else src_ip,
            "dst_ip": src_ip if is_server else dst_ip,
            "src_port": dst_port if is_server else src_port,
            "dst_port": src_port if is_server else dst_port,
            "src_mac": dst_mac if is_server else src_mac,
            "dst_mac": src_mac if is_server else dst_mac,
            "protocol": "TCP",
            "tcp_flags": flag,
            "tcp_state": state,
            "packet_type": flag.replace("-", "_"),
            "packet_size": fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": 65535,
            "service_name": "HTTPS",
            "direction": "S2C" if is_server else "C2S",
            "packet_size_category": "control",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=20))

    # Large upload (50-100 max size packets, mostly C2S)
    n_upload_packets = fake.pyint(min_value=50, max_value=100)
    for i in range(n_upload_packets):
        # 90% upload, 10% ACKs from server
        is_upload = random.random() < 0.90
        packets.append({
            "session_id": session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip if is_upload else dst_ip,
            "dst_ip": dst_ip if is_upload else src_ip,
            "src_port": src_port if is_upload else dst_port,
            "dst_port": dst_port if is_upload else src_port,
            "src_mac": src_mac if is_upload else dst_mac,
            "dst_mac": dst_mac if is_upload else src_mac,
            "protocol": "TCP",
            "tcp_flags": "PSH,ACK" if is_upload else "ACK",
            "tcp_state": "ESTABLISHED",
            "packet_type": "PSH_ACK_C2S" if is_upload else "ACK",
            "packet_size": fake.pyint(min_value=1400, max_value=1500) if is_upload else fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": 65535,
            "service_name": "HTTPS",
            "direction": "C2S" if is_upload else "S2C",
            "packet_size_category": "data" if is_upload else "control",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=10))

    return packets


def generate_anomaly_syn_flood(session_id: str, start_time: datetime, target_ip: str = None) -> list:
    """Generate SYN flood pattern - many SYN packets from spoofed IPs."""
    if target_ip is None:
        target_ip = generate_ip_address(internal=True)

    dst_mac = generate_mac_address()
    packets = []
    current_time = start_time

    # Many SYN packets from random sources
    n_packets = fake.pyint(min_value=100, max_value=200)
    target_port = random.choice([80, 443])

    for i in range(n_packets):
        src_ip = fake.ipv4_public()  # Spoofed source
        src_mac = generate_mac_address()
        src_port = fake.pyint(min_value=1024, max_value=65535)

        packets.append({
            "session_id": f"{session_id}_{i}",
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": target_ip,
            "src_port": src_port,
            "dst_port": target_port,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "protocol": "TCP",
            "tcp_flags": "SYN",
            "tcp_state": "SYN_SENT",
            "packet_type": "SYN",
            "packet_size": fake.pyint(min_value=40, max_value=60),
            "ttl": fake.pyint(min_value=64, max_value=255),
            "window_size": 65535,
            "service_name": "HTTP" if target_port == 80 else "HTTPS",
            "direction": "C2S",
            "packet_size_category": "control",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=5))

    return packets


# =============================================================================
# SUSPICIOUS GENERATORS
# =============================================================================

def generate_suspicious_brute_force(session_id: str, start_time: datetime) -> list:
    """Generate brute force pattern - many short SSH/RDP sessions."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = generate_ip_address(internal=True)
    src_mac = generate_mac_address()
    dst_mac = generate_mac_address()
    packets = []
    current_time = start_time

    service = random.choice(["SSH", "RDP"])
    dst_port = SERVICES[service]["port"]

    # Multiple auth attempts (20-50)
    n_attempts = fake.pyint(min_value=20, max_value=50)

    for attempt in range(n_attempts):
        src_port = fake.pyint(min_value=49152, max_value=65535)
        attempt_session_id = f"{session_id}_attempt_{attempt}"

        # Quick handshake
        for flag, state in [("SYN", "SYN_SENT"), ("SYN,ACK", "SYN_RECEIVED"), ("ACK", "ESTABLISHED")]:
            is_server = flag == "SYN,ACK"
            packets.append({
                "session_id": attempt_session_id,
                "packet_timestamp": current_time.isoformat(),
                "src_ip": dst_ip if is_server else src_ip,
                "dst_ip": src_ip if is_server else dst_ip,
                "src_port": dst_port if is_server else src_port,
                "dst_port": src_port if is_server else dst_port,
                "src_mac": dst_mac if is_server else src_mac,
                "dst_mac": src_mac if is_server else dst_mac,
                "protocol": "TCP",
                "tcp_flags": flag,
                "tcp_state": state,
                "packet_type": flag.replace("-", "_"),
                "packet_size": fake.pyint(min_value=40, max_value=60),
                "ttl": fake.pyint(min_value=60, max_value=128),
                "window_size": fake.pyint(min_value=16384, max_value=65535),
                "service_name": service,
                "direction": "S2C" if is_server else "C2S",
                "packet_size_category": "control",
            })
            current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=20))

        # Auth attempt (1-2 packets)
        packets.append({
            "session_id": attempt_session_id,
            "packet_timestamp": current_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "src_mac": src_mac,
            "dst_mac": dst_mac,
            "protocol": "TCP",
            "tcp_flags": "PSH,ACK",
            "tcp_state": "ESTABLISHED",
            "packet_type": "PSH_ACK_C2S",
            "packet_size": fake.pyint(min_value=100, max_value=200),
            "ttl": fake.pyint(min_value=60, max_value=128),
            "window_size": fake.pyint(min_value=16384, max_value=65535),
            "service_name": service,
            "direction": "C2S",
            "packet_size_category": "data",
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=50, max_value=200))

        # Auth failure response + RST (most of the time)
        if random.random() < 0.95:  # 95% failure rate
            packets.append({
                "session_id": attempt_session_id,
                "packet_timestamp": current_time.isoformat(),
                "src_ip": dst_ip,
                "dst_ip": src_ip,
                "src_port": dst_port,
                "dst_port": src_port,
                "src_mac": dst_mac,
                "dst_mac": src_mac,
                "protocol": "TCP",
                "tcp_flags": "RST",
                "tcp_state": "RESET",
                "packet_type": "RST",
                "packet_size": fake.pyint(min_value=40, max_value=60),
                "ttl": fake.pyint(min_value=60, max_value=128),
                "window_size": 0,
                "service_name": service,
                "direction": "S2C",
                "packet_size_category": "control",
            })

        # Short delay between attempts
        current_time += timedelta(seconds=fake.pyint(min_value=1, max_value=5))

    return packets


def generate_suspicious_lateral_movement(session_id: str, start_time: datetime) -> list:
    """Generate lateral movement pattern - internal host accessing many internal hosts."""
    src_ip = generate_ip_address(internal=True)
    src_mac = generate_mac_address()
    packets = []
    current_time = start_time

    # Access 5-15 internal hosts
    n_targets = fake.pyint(min_value=5, max_value=15)
    services = ["SMB", "SSH", "RDP"]

    for target_num in range(n_targets):
        dst_ip = generate_ip_address(internal=True)
        dst_mac = generate_mac_address()
        service = random.choice(services)
        dst_port = SERVICES[service]["port"]
        src_port = fake.pyint(min_value=49152, max_value=65535)
        target_session_id = f"{session_id}_target_{target_num}"

        # Normal-ish session to each target
        session_packets = generate_normal_session(
            target_session_id, service, current_time
        )

        # Override source IP to be consistent
        for pkt in session_packets:
            if pkt["direction"] == "C2S":
                pkt["src_ip"] = src_ip
                pkt["src_mac"] = src_mac
            else:
                pkt["dst_ip"] = src_ip
                pkt["dst_mac"] = src_mac

        packets.extend(session_packets)
        current_time += timedelta(seconds=fake.pyint(min_value=30, max_value=120))

    return packets


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def create_baseline_pcap_dataset(
    n_normal_sessions: int = 2600,
    n_anomalous_sessions: int = 220,
    n_suspicious_sessions: int = 75,
    seed: int = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Create a baseline PCAP dataset using Faker - no Rockfish dependencies.

    Args:
        n_normal_sessions: Number of normal sessions to generate
        n_anomalous_sessions: Number of anomalous sessions to generate
        n_suspicious_sessions: Number of suspicious sessions to generate
        seed: Random seed for reproducibility
        start_time: Base time for packet timestamps

    Returns:
        DataFrame with all generated packets
    """
    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if start_time is None:
        start_time = datetime(2025, 3, 1, 9, 0, 0)

    packets = []

    # Map services to Rockfish-style template names
    service_to_template = {
        "HTTP": "web_browsing",
        "HTTPS": "web_browsing",
        "SSH": "ssh_session",
        "FTP": "file_transfer",
        "DNS": "database_query",
        "SMTP": "email",
        "IMAP": "email",
        "MYSQL": "database_query",
        "POSTGRESQL": "database_query",
        "SMB": "file_transfer",
        "RDP": "ssh_session",
        "REDIS": "api_calls",
    }

    print(f"Generating {n_normal_sessions} normal sessions...")
    # Normal sessions
    normal_services = list(SERVICES.keys())
    for i in range(n_normal_sessions):
        service = random.choice(normal_services)
        session_start = start_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = generate_normal_session(
            session_id=f"NORMAL_{i:04d}",
            service_name=service,
            start_time=session_start,
        )
        for pkt in session_packets:
            pkt["traffic_category"] = "normal"
            pkt["template_type"] = service_to_template.get(service, service.lower())
        packets.extend(session_packets)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_normal_sessions} normal sessions")

    print(f"Generating {n_anomalous_sessions} anomalous sessions...")
    # Anomalous sessions
    anomaly_generators = {
        "port_scan": generate_anomaly_port_scan,
        "beaconing": generate_anomaly_beaconing,
        "large_upload": generate_anomaly_large_upload,
        "syn_flood": generate_anomaly_syn_flood,
    }
    for i in range(n_anomalous_sessions):
        anomaly_type = random.choice(list(anomaly_generators.keys()))
        session_start = start_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = anomaly_generators[anomaly_type](
            session_id=f"ANOMALY_{i:04d}",
            start_time=session_start,
        )
        for pkt in session_packets:
            pkt["traffic_category"] = "anomalous"
            pkt["template_type"] = anomaly_type
        packets.extend(session_packets)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_anomalous_sessions} anomalous sessions")

    print(f"Generating {n_suspicious_sessions} suspicious sessions...")
    # Suspicious sessions
    suspicious_generators = {
        "brute_force": generate_suspicious_brute_force,
        "lateral_movement": generate_suspicious_lateral_movement,
    }
    for i in range(n_suspicious_sessions):
        suspicious_type = random.choice(list(suspicious_generators.keys()))
        session_start = start_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = suspicious_generators[suspicious_type](
            session_id=f"SUSPICIOUS_{i:04d}",
            start_time=session_start,
        )
        for pkt in session_packets:
            pkt["traffic_category"] = "suspicious"
            pkt["template_type"] = suspicious_type
        packets.extend(session_packets)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_suspicious_sessions} suspicious sessions")

    df = pd.DataFrame(packets)
    df = df.sort_values("packet_timestamp").reset_index(drop=True)

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate baseline PCAP dataset without Rockfish"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_data",
        help="Output directory",
    )
    parser.add_argument(
        "--normal",
        type=int,
        default=2600,
        help="Number of normal sessions",
    )
    parser.add_argument(
        "--anomalous",
        type=int,
        default=220,
        help="Number of anomalous sessions",
    )
    parser.add_argument(
        "--suspicious",
        type=int,
        default=75,
        help="Number of suspicious sessions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    print("\n" + "=" * 60)
    print("BASELINE PCAP GENERATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Normal sessions: {args.normal}")
    print(f"  Anomalous sessions: {args.anomalous}")
    print(f"  Suspicious sessions: {args.suspicious}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print()

    df = create_baseline_pcap_dataset(
        n_normal_sessions=args.normal,
        n_anomalous_sessions=args.anomalous,
        n_suspicious_sessions=args.suspicious,
        seed=args.seed,
    )

    # Save dataset
    output_path = output_dir / "baseline_pcap.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved dataset to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal packets: {len(df):,}")
    print(f"Unique sessions: {df['session_id'].nunique():,}")

    print("\nTraffic Category Distribution:")
    print(df["traffic_category"].value_counts())

    print("\nTemplate Type Distribution:")
    print(df["template_type"].value_counts())

    print("\nTCP State Distribution:")
    print(df["tcp_state"].value_counts())

    print("\nService Distribution:")
    print(df["service_name"].value_counts())


if __name__ == "__main__":
    main()
