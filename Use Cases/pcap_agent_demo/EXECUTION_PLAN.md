# PCAP Agent Demo - Execution Plan

This document outlines the execution plan for creating a comprehensive PCAP analysis demo that includes:
1. Synthetic PCAP data generation with Rockfish APIs (normal + anomalous + suspicious flows)
2. A baseline dataset without Rockfish
3. A PCAP analysis agent using the agentfuel framework
4. Quality testing of the agent using agentfuel
5. Data quality reports for both synthetic and baseline datasets

---

## Phase 1: PCAP Template Generation (Notebook 1)

**File:** `01_pcap_template_generation.ipynb`

**Purpose:** Create reusable templates for normal, anomalous, and suspicious PCAP flows using Rockfish's Entity Data Generator.

### 1.1 Normal Flow Templates

Build on the existing `pcap_packet_generation.ipynb` patterns to create templates for:

| Template | Description | State Machine Characteristics |
|----------|-------------|-------------------------------|
| `normal_web_browsing` | Standard HTTP/HTTPS sessions | Full handshake → 5-15 data packets → graceful close |
| `normal_api_calls` | REST API request/response | Full handshake → 1-3 request packets → 1-5 response packets → close |
| `normal_ssh_session` | Interactive SSH traffic | Full handshake → sustained bidirectional traffic → close |
| `normal_file_transfer` | FTP/SMB file downloads | Full handshake → many large S2C packets → close |
| `normal_database_query` | MySQL/PostgreSQL queries | Full handshake → small request → variable response → close |
| `normal_email` | SMTP/IMAP traffic | Full handshake → protocol-specific exchanges → close |

### 1.2 Anomalous Flow Templates

| Template | Description | State Machine Characteristics |
|----------|-------------|-------------------------------|
| `anomaly_port_scan` | Sequential port probing | SYN → RST (repeated across many ports), very short sessions |
| `anomaly_syn_flood` | SYN flood pattern | SYN only (no SYN-ACK response), high volume |
| `anomaly_large_upload` | Unusual data exfiltration | Full handshake → abnormally large C2S data volume |
| `anomaly_beaconing` | C2 beaconing pattern | Regular intervals, small fixed-size packets, long session |
| `anomaly_dns_tunnel` | DNS tunneling attempt | Oversized DNS-like packets, unusual query patterns |
| `anomaly_slow_loris` | Slow HTTP attack | Incomplete HTTP requests, connection held open |

### 1.3 Suspicious Flow Templates

| Template | Description | State Machine Characteristics |
|----------|-------------|-------------------------------|
| `suspicious_brute_force` | Auth brute force | Multiple short sessions, same dest, different src ports |
| `suspicious_lateral_movement` | Internal reconnaissance | Internal-to-internal, unusual port access patterns |
| `suspicious_data_staging` | Pre-exfil staging | Large internal transfers to single host |
| `suspicious_encrypted_tunnel` | TLS on non-standard port | TLS handshake on unusual ports (not 443) |
| `suspicious_protocol_anomaly` | Protocol mismatch | HTTP-like traffic on non-HTTP ports |

### 1.4 Implementation Approach

```python
# Schema function signature pattern
def create_pcap_template(
    template_type: str,          # "normal", "anomalous", "suspicious"
    template_name: str,          # e.g., "web_browsing", "port_scan"
    n_sessions: int = 100,
    **kwargs
) -> DataSchema:
    """Create a PCAP schema template with specific behavioral characteristics."""
    pass

# Store templates as callable functions in a module
# templates/pcap_templates.py
TEMPLATE_REGISTRY = {
    "normal": {
        "web_browsing": create_normal_web_browsing_schema,
        "api_calls": create_normal_api_calls_schema,
        # ...
    },
    "anomalous": {
        "port_scan": create_anomaly_port_scan_schema,
        "syn_flood": create_anomaly_syn_flood_schema,
        # ...
    },
    "suspicious": {
        "brute_force": create_suspicious_brute_force_schema,
        # ...
    }
}
```

### 1.5 Deliverables
- `templates/pcap_templates.py` - Module with all template creation functions
- `01_pcap_template_generation.ipynb` - Notebook demonstrating template creation and validation
- `templates/template_configs.yaml` - YAML configuration for each template's parameters

---

## Phase 2: Blended Dataset Generation (Script + Config)

**Files:**
- `02_generate_blended_pcap.py` - Main generation script
- `configs/blended_generation_config.yaml` - Configuration file

**Purpose:** Generate a week-long dataset with time-varying volumes and proportions of normal/anomalous/suspicious traffic.

### 2.1 Configuration Schema

```yaml
# configs/blended_generation_config.yaml
output:
  directory: "generated_data/week_simulation"
  format: "csv"  # or "parquet"

time_range:
  start: "2025-03-01T00:00:00Z"
  end: "2025-03-07T23:59:59Z"
  granularity: "1hour"  # Generate data in 1-hour chunks

volume_profile:
  # Sessions per hour by time of day (24-hour pattern, repeated daily)
  hourly_pattern:
    - { hour: 0, base_volume: 500 }
    - { hour: 6, base_volume: 1000 }
    - { hour: 9, base_volume: 5000 }   # Business hours start
    - { hour: 12, base_volume: 4000 }  # Lunch dip
    - { hour: 14, base_volume: 6000 }  # Peak afternoon
    - { hour: 17, base_volume: 3000 }  # End of business
    - { hour: 20, base_volume: 1500 }
    - { hour: 23, base_volume: 800 }

  # Day-of-week multipliers
  daily_multipliers:
    monday: 1.0
    tuesday: 1.1
    wednesday: 1.1
    thursday: 1.0
    friday: 0.9
    saturday: 0.3
    sunday: 0.2

traffic_mix:
  # Base proportions (must sum to 1.0)
  base_proportions:
    normal: 0.92
    anomalous: 0.05
    suspicious: 0.03

  # Time-based variations (optional overrides)
  scheduled_events:
    - name: "security_incident_monday"
      start: "2025-03-03T14:00:00Z"
      end: "2025-03-03T18:00:00Z"
      proportions:
        normal: 0.70
        anomalous: 0.20
        suspicious: 0.10

    - name: "maintenance_window"
      start: "2025-03-05T02:00:00Z"
      end: "2025-03-05T04:00:00Z"
      proportions:
        normal: 0.99
        anomalous: 0.005
        suspicious: 0.005
      volume_multiplier: 0.1

# Template selection weights within each category
template_weights:
  normal:
    web_browsing: 0.40
    api_calls: 0.25
    ssh_session: 0.10
    file_transfer: 0.10
    database_query: 0.10
    email: 0.05

  anomalous:
    port_scan: 0.30
    syn_flood: 0.20
    large_upload: 0.20
    beaconing: 0.20
    slow_loris: 0.10

  suspicious:
    brute_force: 0.35
    lateral_movement: 0.25
    data_staging: 0.20
    encrypted_tunnel: 0.10
    protocol_anomaly: 0.10

# Host and network configuration
network:
  n_internal_hosts: 200
  n_external_hosts: 500
  n_servers: 50
  internal_subnets:
    - "10.1.0.0/16"
    - "10.2.0.0/16"
    - "192.168.0.0/16"

# Labeling for ground truth
labeling:
  include_labels: true
  label_column: "traffic_category"  # "normal", "anomalous", "suspicious"
  include_template_name: true
  template_column: "template_type"
```

### 2.2 Script Architecture

```python
# 02_generate_blended_pcap.py

class BlendedPCAPGenerator:
    """Generates week-long PCAP datasets with configurable traffic mix."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.template_registry = load_template_registry()
        self.conn = rf.Connection.from_env()

    def generate(self) -> pd.DataFrame:
        """Main generation loop."""
        all_packets = []

        for time_chunk in self._iterate_time_chunks():
            # Determine volume for this chunk
            volume = self._calculate_volume(time_chunk)

            # Determine traffic mix proportions
            proportions = self._get_proportions(time_chunk)

            # Generate sessions for each category
            for category, proportion in proportions.items():
                n_sessions = int(volume * proportion)
                template_weights = self.config['template_weights'][category]

                # Allocate sessions across templates
                for template_name, weight in template_weights.items():
                    n_template_sessions = int(n_sessions * weight)
                    if n_template_sessions > 0:
                        packets = self._generate_from_template(
                            category, template_name,
                            n_template_sessions, time_chunk
                        )
                        packets['traffic_category'] = category
                        packets['template_type'] = template_name
                        all_packets.append(packets)

        return pd.concat(all_packets).sort_values('packet_timestamp')

    async def _generate_from_template(
        self, category: str, template_name: str,
        n_sessions: int, time_range: tuple
    ) -> pd.DataFrame:
        """Generate packets using a specific template."""
        schema_fn = self.template_registry[category][template_name]
        schema = schema_fn(n_sessions=n_sessions, time_range=time_range)

        config = ra.GenerateFromDataSchema.Config(
            schema=schema, upload_datasets=False
        )
        # ... execute workflow and return packets
```

### 2.3 Deliverables
- `02_generate_blended_pcap.py` - Main generation script
- `configs/blended_generation_config.yaml` - Example configuration
- `utils/volume_calculator.py` - Helper for time-varying volume calculations
- Generated dataset: `generated_data/week_simulation/pcap_week.csv`

---

## Phase 3: Baseline Dataset (Pure Python + Faker - No Rockfish)

**File:** `03_create_baseline_pcap.py`

**Purpose:** Create a PCAP dataset from scratch using only basic Python libraries (Faker, numpy, pandas). This serves as a comparison baseline generated without any Rockfish APIs or templates.

### 3.1 Approach: Pure Faker-Based Generation

Generate PCAP data entirely from scratch using the Faker library for realistic network data:

```python
# 03_create_baseline_pcap.py

from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

fake = Faker()

# Service definitions with realistic port mappings
SERVICES = {
    'HTTP': {'port': 80, 'protocol': 'TCP'},
    'HTTPS': {'port': 443, 'protocol': 'TCP'},
    'SSH': {'port': 22, 'protocol': 'TCP'},
    'FTP': {'port': 21, 'protocol': 'TCP'},
    'DNS': {'port': 53, 'protocol': 'UDP'},
    'SMTP': {'port': 25, 'protocol': 'TCP'},
    'MySQL': {'port': 3306, 'protocol': 'TCP'},
    'PostgreSQL': {'port': 5432, 'protocol': 'TCP'},
}

TCP_FLAGS = ['SYN', 'SYN-ACK', 'ACK', 'PSH-ACK', 'FIN-ACK', 'RST']

def generate_ip_address(internal: bool = True) -> str:
    """Generate realistic IP addresses using Faker."""
    if internal:
        # Internal subnets
        subnet = random.choice(['10.1', '10.2', '192.168.1', '192.168.2'])
        return f"{subnet}.{fake.pyint(min_value=1, max_value=254)}.{fake.pyint(min_value=1, max_value=254)}"
    else:
        return fake.ipv4_public()

def generate_mac_address() -> str:
    """Generate MAC address using Faker."""
    return fake.mac_address()

def generate_normal_session(session_id: str, service_name: str, start_time: datetime) -> list:
    """Generate a normal TCP session with realistic packet flow."""
    service = SERVICES[service_name]
    src_ip = generate_ip_address(internal=True)
    dst_ip = generate_ip_address(internal=False) if service_name in ['HTTP', 'HTTPS'] else generate_ip_address(internal=True)
    src_port = fake.pyint(min_value=49152, max_value=65535)  # Ephemeral port
    dst_port = service['port']

    packets = []
    current_time = start_time

    # TCP Handshake
    for flag in ['SYN', 'SYN-ACK', 'ACK']:
        packets.append({
            'session_id': session_id,
            'packet_timestamp': current_time.isoformat(),
            'src_ip': src_ip if flag != 'SYN-ACK' else dst_ip,
            'dst_ip': dst_ip if flag != 'SYN-ACK' else src_ip,
            'src_port': src_port if flag != 'SYN-ACK' else dst_port,
            'dst_port': dst_port if flag != 'SYN-ACK' else src_port,
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': service['protocol'],
            'tcp_flags': flag,
            'packet_size': fake.pyint(min_value=40, max_value=80),
            'ttl': fake.pyint(min_value=60, max_value=128),
            'window_size': fake.pyint(min_value=16384, max_value=65535),
            'service_name': service_name,
            'direction': 'C2S' if flag != 'SYN-ACK' else 'S2C',
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=50))

    # Data exchange (5-20 packets)
    n_data_packets = fake.pyint(min_value=5, max_value=20)
    for i in range(n_data_packets):
        is_client = i % 2 == 0
        packets.append({
            'session_id': session_id,
            'packet_timestamp': current_time.isoformat(),
            'src_ip': src_ip if is_client else dst_ip,
            'dst_ip': dst_ip if is_client else src_ip,
            'src_port': src_port if is_client else dst_port,
            'dst_port': dst_port if is_client else src_port,
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': service['protocol'],
            'tcp_flags': 'PSH-ACK',
            'packet_size': fake.pyint(min_value=100, max_value=1500),
            'ttl': fake.pyint(min_value=60, max_value=128),
            'window_size': fake.pyint(min_value=16384, max_value=65535),
            'service_name': service_name,
            'direction': 'C2S' if is_client else 'S2C',
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=10, max_value=500))

    # TCP Teardown
    for flag in ['FIN-ACK', 'ACK']:
        packets.append({
            'session_id': session_id,
            'packet_timestamp': current_time.isoformat(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': service['protocol'],
            'tcp_flags': flag,
            'packet_size': fake.pyint(min_value=40, max_value=80),
            'ttl': fake.pyint(min_value=60, max_value=128),
            'window_size': fake.pyint(min_value=16384, max_value=65535),
            'service_name': service_name,
            'direction': 'C2S',
        })
        current_time += timedelta(milliseconds=fake.pyint(min_value=1, max_value=50))

    return packets

def generate_anomaly_port_scan(session_id: str, start_time: datetime) -> list:
    """Generate port scan pattern - many SYN packets to different ports, RST responses."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = generate_ip_address(internal=True)
    packets = []
    current_time = start_time

    # Scan 20-50 ports rapidly
    n_ports = fake.pyint(min_value=20, max_value=50)
    scanned_ports = random.sample(range(1, 1024), n_ports)

    for port in scanned_ports:
        # SYN packet
        packets.append({
            'session_id': f"{session_id}_port_{port}",
            'packet_timestamp': current_time.isoformat(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': fake.pyint(min_value=49152, max_value=65535),
            'dst_port': port,
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': 'TCP',
            'tcp_flags': 'SYN',
            'packet_size': fake.pyint(min_value=40, max_value=60),
            'ttl': fake.pyint(min_value=60, max_value=128),
            'window_size': fake.pyint(min_value=1024, max_value=4096),
            'service_name': 'SCAN',
            'direction': 'C2S',
        })
        # RST response (port closed) or no response
        if random.random() > 0.3:  # 70% ports closed
            packets.append({
                'session_id': f"{session_id}_port_{port}",
                'packet_timestamp': (current_time + timedelta(milliseconds=fake.pyint(min_value=1, max_value=10))).isoformat(),
                'src_ip': dst_ip,
                'dst_ip': src_ip,
                'src_port': port,
                'dst_port': packets[-1]['src_port'],
                'src_mac': generate_mac_address(),
                'dst_mac': generate_mac_address(),
                'protocol': 'TCP',
                'tcp_flags': 'RST',
                'packet_size': fake.pyint(min_value=40, max_value=60),
                'ttl': fake.pyint(min_value=60, max_value=128),
                'window_size': 0,
                'service_name': 'SCAN',
                'direction': 'S2C',
            })
        current_time += timedelta(milliseconds=fake.pyint(min_value=5, max_value=50))

    return packets

def generate_anomaly_beaconing(session_id: str, start_time: datetime) -> list:
    """Generate C2 beaconing pattern - regular interval small packets."""
    src_ip = generate_ip_address(internal=True)
    dst_ip = fake.ipv4_public()  # External C2 server
    packets = []
    current_time = start_time

    # Beacon every 60 seconds (+/- small jitter) for 10-20 beacons
    beacon_interval = 60  # seconds
    n_beacons = fake.pyint(min_value=10, max_value=20)

    for i in range(n_beacons):
        # Small beacon packet (check-in)
        packets.append({
            'session_id': session_id,
            'packet_timestamp': current_time.isoformat(),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': fake.pyint(min_value=49152, max_value=65535),
            'dst_port': 443,  # Hiding in HTTPS
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': 'TCP',
            'tcp_flags': 'PSH-ACK',
            'packet_size': fake.pyint(min_value=50, max_value=100),  # Small, consistent
            'ttl': fake.pyint(min_value=60, max_value=128),
            'window_size': fake.pyint(min_value=16384, max_value=65535),
            'service_name': 'HTTPS',
            'direction': 'C2S',
        })
        # Response
        packets.append({
            'session_id': session_id,
            'packet_timestamp': (current_time + timedelta(milliseconds=fake.pyint(min_value=100, max_value=500))).isoformat(),
            'src_ip': dst_ip,
            'dst_ip': src_ip,
            'src_port': 443,
            'dst_port': packets[-1]['src_port'],
            'src_mac': generate_mac_address(),
            'dst_mac': generate_mac_address(),
            'protocol': 'TCP',
            'tcp_flags': 'PSH-ACK',
            'packet_size': fake.pyint(min_value=50, max_value=150),
            'ttl': fake.pyint(min_value=40, max_value=64),
            'window_size': fake.pyint(min_value=16384, max_value=65535),
            'service_name': 'HTTPS',
            'direction': 'S2C',
        })
        # Add jitter to next beacon
        jitter = fake.pyint(min_value=-5, max_value=5)
        current_time += timedelta(seconds=beacon_interval + jitter)

    return packets

def create_baseline_pcap_dataset(
    n_normal_sessions: int = 500,
    n_anomalous_sessions: int = 50,
    n_suspicious_sessions: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """Create a baseline PCAP dataset using Faker - no Rockfish dependencies."""

    Faker.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    packets = []
    base_time = datetime(2025, 3, 1, 9, 0, 0)

    # Normal sessions
    for i in range(n_normal_sessions):
        service = random.choice(list(SERVICES.keys()))
        start_time = base_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = generate_normal_session(
            session_id=f"NORMAL_{i:04d}",
            service_name=service,
            start_time=start_time
        )
        for pkt in session_packets:
            pkt['traffic_category'] = 'normal'
            pkt['template_type'] = f'normal_{service.lower()}'
        packets.extend(session_packets)

    # Anomalous sessions
    anomaly_generators = {
        'port_scan': generate_anomaly_port_scan,
        'beaconing': generate_anomaly_beaconing,
    }
    for i in range(n_anomalous_sessions):
        anomaly_type = random.choice(list(anomaly_generators.keys()))
        start_time = base_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = anomaly_generators[anomaly_type](
            session_id=f"ANOMALY_{i:04d}",
            start_time=start_time
        )
        for pkt in session_packets:
            pkt['traffic_category'] = 'anomalous'
            pkt['template_type'] = f'anomaly_{anomaly_type}'
        packets.extend(session_packets)

    # Suspicious sessions (reuse patterns with slight variations)
    for i in range(n_suspicious_sessions):
        service = random.choice(['SSH', 'MySQL'])
        start_time = base_time + timedelta(seconds=random.randint(0, 86400))
        session_packets = generate_normal_session(
            session_id=f"SUSPICIOUS_{i:04d}",
            service_name=service,
            start_time=start_time
        )
        for pkt in session_packets:
            pkt['traffic_category'] = 'suspicious'
            pkt['template_type'] = 'suspicious_unusual_access'
        packets.extend(session_packets)

    df = pd.DataFrame(packets)
    df = df.sort_values('packet_timestamp').reset_index(drop=True)
    return df
```

### 3.2 Key Differences from Rockfish Approach

| Aspect | Rockfish (Phase 1-2) | Baseline (Phase 3) |
|--------|---------------------|-------------------|
| **IP Generation** | Rockfish Entity Generator | Faker `ipv4_public()` / `ipv4_private()` |
| **Timing** | State machine with distribution modeling | Simple `timedelta` with random offsets |
| **Packet Sizes** | Distribution-based from schema | Faker `pyint()` with min/max ranges |
| **MAC Addresses** | Entity-linked generation | Faker `mac_address()` |
| **Session Flow** | Complex state machines | Hardcoded TCP handshake sequence |
| **Realism** | Statistical modeling of real patterns | Rule-based with randomization |
| **Dependencies** | rockfish, rockfish-actions | faker, pandas, numpy only |

### 3.2 Baseline Query Set

Create a comprehensive set of PCAP analysis queries with ground-truth answers:

```python
# baseline_queries.py

BASELINE_QUERIES = [
    # Counting queries
    {
        "query_id": "count_1",
        "query": "How many total packets are in the dataset?",
        "answer": lambda df: len(df),
        "query_type": "aggregation"
    },
    {
        "query_id": "count_2",
        "query": "How many unique sessions are there?",
        "answer": lambda df: df['session_id'].nunique(),
        "query_type": "aggregation"
    },

    # Traffic analysis queries
    {
        "query_id": "traffic_1",
        "query": "What percentage of traffic is HTTP or HTTPS?",
        "answer": lambda df: round(100 * df[df['service_name'].isin(['HTTP', 'HTTPS'])].shape[0] / len(df), 2),
        "query_type": "aggregation"
    },
    {
        "query_id": "traffic_2",
        "query": "What is the average packet size for data packets?",
        "answer": lambda df: round(df[df['packet_size_category'] == 'data']['packet_size'].mean(), 2),
        "query_type": "aggregation"
    },

    # Security-focused queries
    {
        "query_id": "security_1",
        "query": "How many sessions ended with a RST (reset)?",
        "answer": lambda df: df.groupby('session_id')['tcp_state'].last().value_counts().get('RESET', 0),
        "query_type": "security"
    },
    {
        "query_id": "security_2",
        "query": "What percentage of sessions are classified as anomalous?",
        "answer": lambda df: round(100 * df[df['traffic_category'] == 'anomalous']['session_id'].nunique() / df['session_id'].nunique(), 2),
        "query_type": "security"
    },
    {
        "query_id": "security_3",
        "query": "Which source IP has the most failed connections (RST)?",
        "answer": lambda df: df[df['tcp_state'] == 'RESET'].groupby('src_ip').size().idxmax(),
        "query_type": "security"
    },

    # Pattern detection queries
    {
        "query_id": "pattern_1",
        "query": "Are there any IPs that connected to more than 10 different destination ports?",
        "answer": lambda df: df.groupby('src_ip')['dst_port'].nunique().gt(10).any(),
        "query_type": "detection"
    },
    {
        "query_id": "pattern_2",
        "query": "What is the maximum bytes transferred in a single session?",
        "answer": lambda df: df.groupby('session_id')['packet_size'].sum().max(),
        "query_type": "aggregation"
    },

    # Time-based queries
    {
        "query_id": "time_1",
        "query": "During which hour was traffic volume highest?",
        "answer": lambda df: pd.to_datetime(df['packet_timestamp']).dt.hour.value_counts().idxmax(),
        "query_type": "temporal"
    },
]
```

### 3.3 Deliverables
- `03_create_baseline_pcap.py` - Baseline dataset generator
- `baseline_data/baseline_pcap.csv` - Generated baseline dataset
- `baseline_data/baseline_queries.json` - Queries with ground-truth answers
- `workload_data/pcap_baseline_workload.csv` - Formatted for agentfuel evaluation

---

## Phase 4: PCAP Analysis Agent

**Files:**
- `agents/pcap_agent/runner.py` - Agent implementation
- `agents/pcap_agent/.env.template` - Environment template
- `configs/agent/pcap_agent.yaml` - Agent configuration

**Purpose:** Create a PCAP analysis agent using the agentfuel framework patterns.

### 4.1 Agent Architecture

Following the agentfuel `rflangchain` pattern:

```python
# agents/pcap_agent/runner.py

from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

from config import RESPONSE_DATA_HEADERS


class PCAPAnalysisAgent:
    """Agent for analyzing PCAP packet capture data."""

    SYSTEM_PROMPT = """You are a network security analyst specialized in PCAP analysis.
    You have access to a packet capture dataset with the following columns:
    {schema_description}

    When answering questions:
    1. Use the provided tools to query the data
    2. Focus on security-relevant insights
    3. Be precise with numeric answers
    4. Identify patterns that might indicate attacks or anomalies
    """

    def __init__(self, env_filepath: str, data_filepath: str):
        load_dotenv(env_filepath)
        self.df = pd.read_csv(data_filepath)
        self.llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        df = self.df

        @tool
        def query_packets(query: str) -> str:
            """Execute a pandas query on the PCAP dataset."""
            try:
                result = df.query(query)
                return f"Query returned {len(result)} rows.\n{result.head(10).to_string()}"
            except Exception as e:
                return f"Query error: {e}"

        @tool
        def get_statistics(column: str, groupby: Optional[str] = None) -> str:
            """Get statistics for a column, optionally grouped."""
            try:
                if groupby:
                    return df.groupby(groupby)[column].describe().to_string()
                return df[column].describe().to_string()
            except Exception as e:
                return f"Error: {e}"

        @tool
        def count_unique(column: str) -> int:
            """Count unique values in a column."""
            return df[column].nunique()

        @tool
        def aggregate(column: str, operation: str, groupby: Optional[str] = None) -> str:
            """Perform aggregation (sum, mean, max, min, count) on a column."""
            ops = {'sum': 'sum', 'mean': 'mean', 'max': 'max', 'min': 'min', 'count': 'count'}
            if operation not in ops:
                return f"Invalid operation. Use: {list(ops.keys())}"
            try:
                if groupby:
                    result = df.groupby(groupby)[column].agg(operation)
                else:
                    result = getattr(df[column], operation)()
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        @tool
        def detect_port_scan(threshold: int = 10) -> str:
            """Detect potential port scans (IPs connecting to many ports)."""
            port_counts = df.groupby('src_ip')['dst_port'].nunique()
            scanners = port_counts[port_counts > threshold]
            if len(scanners) == 0:
                return "No potential port scans detected."
            return f"Potential port scanners:\n{scanners.to_string()}"

        @tool
        def get_session_summary(session_id: str) -> str:
            """Get detailed summary of a specific session."""
            session = df[df['session_id'] == session_id]
            if len(session) == 0:
                return f"Session {session_id} not found."
            return session.to_string()

        return [query_packets, get_statistics, count_unique,
                aggregate, detect_port_scan, get_session_summary]

    def _create_agent(self):
        schema_desc = f"Columns: {', '.join(self.df.columns)}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT.format(schema_description=schema_desc)),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def run(self, workload_df: pd.DataFrame) -> pd.DataFrame:
        """Run agent on workload queries."""
        responses = []
        for _, row in workload_df.iterrows():
            query = row['query']
            try:
                result = self.agent.invoke({"input": query})
                response = result.get('output', 'No response')
            except Exception as e:
                response = f"Error: {e}"
            responses.append({
                'query_id': row['query_id'],
                'query': query,
                'response': response
            })
        return pd.DataFrame(responses)


# Agentfuel-compatible runner wrapper
class Runner:
    def __init__(self, env_filepath: str, data_filepath: str):
        self.agent = PCAPAnalysisAgent(env_filepath, data_filepath)

    def run(self, workload_df: pd.DataFrame) -> pd.DataFrame:
        return self.agent.run(workload_df)
```

### 4.2 Agent Configuration

```yaml
# configs/agent/pcap_agent.yaml
agent_type: pcap_agent
env_filepath: agents/pcap_agent/.env
```

```bash
# agents/pcap_agent/.env.template
ANTHROPIC_API_KEY=your_key_here
```

### 4.3 Deliverables
- `agents/pcap_agent/runner.py` - Agent implementation
- `agents/pcap_agent/.env.template` - Environment template
- `configs/agent/pcap_agent.yaml` - Agent configuration
- `configs/agent/pcap_agent_baseline.yaml` - Config for baseline data

---

## Phase 5: Test Suite Generation (Rockfish + Baseline)

**Files:**
- `04_generate_pcap_test_suite.py` - Test suite generator
- `workload_data/pcap_rockfish_workload.csv` - Workload for Rockfish-generated data
- `workload_data/pcap_baseline_workload.csv` - Workload for baseline data

**Purpose:** Generate test suites for both Rockfish-generated and baseline datasets.

### 5.1 Test Suite Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Suite Generation                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Rockfish Path:                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Week PCAP    │───>│ Rockfish API │───>│ Test Suite   │       │
│  │ Dataset      │    │ /generate-   │    │ JSON         │       │
│  │ (Phase 2)    │    │ test-suite   │    │              │       │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                                                  v               │
│                                          ┌──────────────┐       │
│                                          │ Workload CSV │       │
│                                          │ (agentfuel)  │       │
│                                          └──────────────┘       │
│                                                                  │
│  Baseline Path:                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Baseline     │───>│ Manual Query │───>│ Workload CSV │       │
│  │ Dataset      │    │ Generation   │    │ (agentfuel)  │       │
│  │ (Phase 3)    │    │ (Python)     │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Suite Configuration

```yaml
# configs/basic/generate_pcap_test_suite.yaml
csv_file: "generated_data/week_simulation/pcap_week.csv"
dataset_name: "pcap_week_simulation"
categorical:
  - traffic_category
  - template_type
  - service_name
  - tcp_flags
  - direction
  - client_device_type
  - server_type
measurement:
  - packet_size
  - window_size
  - ttl
timestamp_column: "packet_timestamp"
variations: 2
max_cases: 50
output: "pcap_rockfish_test_suite.json"
```

### 5.3 Deliverables
- `04_generate_pcap_test_suite.py` - Test suite generator script
- `configs/basic/generate_pcap_test_suite.yaml` - Rockfish test suite config
- `workload_data/pcap_rockfish_workload.csv` - Rockfish workload
- `workload_data/pcap_baseline_workload.csv` - Baseline workload

---

## Phase 6: Agent Evaluation (Agentfuel)

**Files:**
- `05_evaluate_pcap_agent.py` - Evaluation runner
- `configs/config.yaml` - Hydra configuration

**Purpose:** Run the PCAP agent against both datasets and score its performance.

### 6.1 Evaluation Configuration

```yaml
# configs/config.yaml
defaults:
  - agent: pcap_agent
  - workload: pcap_rockfish

# Alternative configs:
# configs/agent/pcap_agent_baseline.yaml - for baseline evaluation
# configs/workload/pcap_baseline.yaml - for baseline workload
```

```yaml
# configs/workload/pcap_rockfish.yaml
workload_type: pcap
data_filepath: workload_data/pcap_rockfish_workload.csv

# configs/workload/pcap_baseline.yaml
workload_type: pcap_baseline
data_filepath: workload_data/pcap_baseline_workload.csv
```

### 6.2 Running Evaluations

```bash
# Run on Rockfish-generated data
python main.py agent=pcap_agent workload=pcap_rockfish

# Run on baseline data
python main.py agent=pcap_agent workload=pcap_baseline

# Compare scores
python collect_scores.py --dirs outputs/pcap_agent_rockfish outputs/pcap_agent_baseline
```

### 6.3 Deliverables
- `configs/workload/pcap_rockfish.yaml` - Rockfish workload config
- `configs/workload/pcap_baseline.yaml` - Baseline workload config
- `outputs/pcap_agent_rockfish/` - Evaluation results (Rockfish data)
- `outputs/pcap_agent_baseline/` - Evaluation results (baseline data)

---

## Phase 7: Data Quality Reports

**Files:**
- `06_data_quality_report.py` - Data quality report generator
- `reports/` - Output directory for reports

**Purpose:** Generate comprehensive data quality reports for both datasets.

### 7.1 Quality Metrics

| Metric Category | Metrics |
|-----------------|---------|
| **Completeness** | Missing values, null rates, coverage |
| **Consistency** | TCP state machine validity, port range compliance, IP format validity |
| **Statistical** | Distributions, outliers, correlations |
| **Temporal** | Time gaps, ordering, seasonality |
| **Network-specific** | Handshake completion rate, bidirectional flow balance, service port accuracy |
| **Security** | Attack pattern fidelity, anomaly realism |

### 7.2 Report Structure

```python
# 06_data_quality_report.py

class PCAPDataQualityReport:
    """Generate comprehensive data quality report for PCAP datasets."""

    def __init__(self, df: pd.DataFrame, dataset_name: str):
        self.df = df
        self.dataset_name = dataset_name
        self.metrics = {}

    def generate_report(self) -> dict:
        """Generate full quality report."""
        self.metrics = {
            'dataset_info': self._dataset_info(),
            'completeness': self._completeness_metrics(),
            'consistency': self._consistency_metrics(),
            'statistical': self._statistical_metrics(),
            'temporal': self._temporal_metrics(),
            'network': self._network_metrics(),
            'security': self._security_metrics(),
        }
        return self.metrics

    def _completeness_metrics(self) -> dict:
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_by_column': self.df.isnull().sum().to_dict(),
            'missing_rate': self.df.isnull().mean().to_dict(),
        }

    def _consistency_metrics(self) -> dict:
        # Check TCP state machine validity
        valid_handshakes = self._check_handshake_validity()
        valid_port_ranges = self._check_port_ranges()
        valid_ips = self._check_ip_validity()

        return {
            'valid_handshake_rate': valid_handshakes,
            'valid_port_range_rate': valid_port_ranges,
            'valid_ip_rate': valid_ips,
            'tcp_state_distribution': self.df['tcp_state'].value_counts().to_dict(),
        }

    def _network_metrics(self) -> dict:
        return {
            'bidirectional_balance': self._bidirectional_balance(),
            'service_port_accuracy': self._service_port_accuracy(),
            'packet_size_by_type': self._packet_size_by_type(),
            'sessions_per_category': self._sessions_per_category(),
        }

    def _security_metrics(self) -> dict:
        return {
            'traffic_category_distribution': self._category_distribution(),
            'anomaly_types_present': self._anomaly_types(),
            'attack_pattern_fidelity': self._attack_fidelity(),
        }

    def to_html(self, output_path: str):
        """Export report as HTML."""
        # Generate interactive HTML report with charts
        pass

    def to_json(self, output_path: str):
        """Export report as JSON."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
```

### 7.3 Comparison Report

```python
# 06_compare_datasets.py

def compare_datasets(rockfish_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict:
    """Generate comparison report between Rockfish and baseline datasets."""

    rockfish_report = PCAPDataQualityReport(rockfish_df, "rockfish").generate_report()
    baseline_report = PCAPDataQualityReport(baseline_df, "baseline").generate_report()

    comparison = {
        'size_comparison': {
            'rockfish_rows': rockfish_report['dataset_info']['total_rows'],
            'baseline_rows': baseline_report['dataset_info']['total_rows'],
        },
        'completeness_comparison': compare_metrics(
            rockfish_report['completeness'],
            baseline_report['completeness']
        ),
        'consistency_comparison': compare_metrics(
            rockfish_report['consistency'],
            baseline_report['consistency']
        ),
        'network_comparison': compare_metrics(
            rockfish_report['network'],
            baseline_report['network']
        ),
    }

    return comparison
```

### 7.4 Deliverables
- `06_data_quality_report.py` - Quality report generator
- `reports/rockfish_quality_report.html` - Rockfish dataset report
- `reports/rockfish_quality_report.json` - Rockfish dataset report (JSON)
- `reports/baseline_quality_report.html` - Baseline dataset report
- `reports/baseline_quality_report.json` - Baseline dataset report (JSON)
- `reports/comparison_report.html` - Side-by-side comparison

---

## File Structure Summary

```
pcap_agent_demo/
├── EXECUTION_PLAN.md                    # This document
├── requirements.txt                      # Python dependencies
│
├── templates/
│   ├── __init__.py
│   ├── pcap_templates.py                # Template creation functions
│   └── template_configs.yaml            # Template parameters
│
├── configs/
│   ├── config.yaml                      # Main Hydra config
│   ├── blended_generation_config.yaml   # Week simulation config
│   ├── agent/
│   │   ├── pcap_agent.yaml
│   │   └── pcap_agent_baseline.yaml
│   ├── workload/
│   │   ├── pcap_rockfish.yaml
│   │   └── pcap_baseline.yaml
│   └── basic/
│       └── generate_pcap_test_suite.yaml
│
├── agents/
│   └── pcap_agent/
│       ├── __init__.py
│       ├── runner.py                    # PCAP analysis agent
│       └── .env.template
│
├── notebooks/
│   └── 01_pcap_template_generation.ipynb
│
├── scripts/
│   ├── 02_generate_blended_pcap.py
│   ├── 03_create_baseline_pcap.py
│   ├── 04_generate_pcap_test_suite.py
│   ├── 05_evaluate_pcap_agent.py
│   └── 06_data_quality_report.py
│
├── workload_data/
│   ├── pcap_rockfish_test_suite.json
│   ├── pcap_rockfish_workload.csv
│   └── pcap_baseline_workload.csv
│
├── generated_data/
│   └── week_simulation/
│       └── pcap_week.csv
│
├── baseline_data/
│   ├── baseline_pcap.csv
│   └── baseline_queries.json
│
├── orig_data/
│   ├── pcap_rockfish.csv               # Copy for agentfuel
│   └── pcap_baseline.csv               # Copy for agentfuel
│
├── reports/
│   ├── rockfish_quality_report.html
│   ├── rockfish_quality_report.json
│   ├── baseline_quality_report.html
│   ├── baseline_quality_report.json
│   └── comparison_report.html
│
└── outputs/                             # Hydra outputs
    ├── pcap_agent_rockfish/
    │   ├── workload.csv
    │   ├── responses.csv
    │   ├── responses_evaluated.csv
    │   └── scores.csv
    └── pcap_agent_baseline/
        ├── workload.csv
        ├── responses.csv
        ├── responses_evaluated.csv
        └── scores.csv
```

---

## Execution Order

| Step | Phase | Script/Notebook | Inputs | Outputs | Dependencies |
|------|-------|-----------------|--------|---------|--------------|
| 1 | Phase 1 | `01_pcap_template_generation.ipynb` | Existing PCAP notebook | `templates/pcap_templates.py` | Rockfish |
| 2 | Phase 2 | `02_generate_blended_pcap.py` | Templates + config | `generated_data/week_simulation/` | Rockfish |
| 3 | Phase 3 | `03_create_baseline_pcap.py` | None (self-contained) | `baseline_data/` | **Faker only** |
| 4 | Phase 4 | Agent setup | - | `agents/pcap_agent/` | LangChain |
| 5 | Phase 5 | `04_generate_pcap_test_suite.py` | Datasets | `workload_data/` | Rockfish (optional) |
| 6 | Phase 6 | `05_evaluate_pcap_agent.py` | Agent + workloads | `outputs/` | Hydra |
| 7 | Phase 7 | `06_data_quality_report.py` | Datasets | `reports/` | pandas, matplotlib |

**Note:** Phase 3 is completely independent and can run without any Rockfish installation.

---

## Success Criteria

1. **Template Generation**: All 18 templates (6 normal, 6 anomalous, 6 suspicious) generate valid PCAP data
2. **Blended Dataset**: Week-long dataset with 1M+ packets, proper time-varying volume
3. **Baseline Dataset**: 5K+ packets with ground-truth labels and 30+ test queries
4. **Agent Performance**: Agent achieves >70% accuracy on basic queries
5. **Quality Reports**: Both datasets pass consistency checks (>95% valid handshakes, port ranges)
6. **Comparison**: Clear metrics showing Rockfish vs baseline data characteristics

---

## Dependencies

```txt
# requirements.txt

# Core (all phases)
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0

# Rockfish phases (1, 2, 5 Rockfish path)
rockfish>=0.1.0

# Baseline generation (Phase 3 - NO Rockfish needed)
faker>=18.0.0

# Agent framework (Phase 4, 5, 6)
hydra-core>=1.3.0
omegaconf>=2.3.0
langchain>=0.1.0
langchain-anthropic>=0.1.0
python-dotenv>=1.0.0

# Utilities
requests>=2.31.0
matplotlib>=3.7.0
seaborn>=0.12.0
jinja2>=3.1.0  # For HTML reports
```

**Note:** Phase 3 (Baseline Dataset) can run with minimal dependencies:
```txt
# Minimal requirements for baseline only
pandas>=2.0.0
numpy>=1.24.0
faker>=18.0.0
```
