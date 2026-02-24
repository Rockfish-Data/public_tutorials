"""
IoT Network Traffic Schema-Based Data Generation

This module implements a schema-based generation workflow for IoT network traffic data,
inspired by the MUDactivity behavior grammar framework from:

"Towards Behavior Grammar-Driven IoT Network Traffic Generation using MUD Specifications"
Zhang et al., CPSIoTSec '25, ACM CCS 2025

The approach models IoT device behaviors through four key attributes:
1. Packet count per flow
2. Packet sizes per flow
3. Packet inter-arrival times (IAT) within flows
4. Flow inter-arrival times between successive flows

Each attribute can be:
- Independent: Generated from its own state machine
- Dependent: Generated based on another attribute's current state

This implementation uses Rockfish's Entity Data Generator to create synthetic
IoT network traffic data with realistic temporal patterns, state machines,
and entity relationships.
"""

from rockfish.actions.ent import (
    CategoricalParams,
    Column,
    ColumnCategoryType,
    ColumnType,
    DataSchema,
    Derivation,
    DerivationFunctionType,
    Domain,
    DomainType,
    Entity,
    EntityRelationship,
    EntityRelationshipType,
    GlobalTimestamp,
    IDParams,
    MapValuesParams,
    NormalDistParams,
    SampleFromColumnParams,
    StateMachineParams,
    SumParams,
    TimeseriesParams,
    Timestamp,
    Transition,
    UniformDistParams,
    MultiplyParams,
)


def create_iot_device_schema(
    n_devices: int = 10,
    n_service_flows: int = 50,
    n_packets: int = 500,
    global_start_time: str = "2025-01-01T00:00:00Z",
    global_end_time: str = "2025-01-02T00:00:00Z",
    global_time_interval: str = "1min",
) -> DataSchema:
    """
    Create a comprehensive IoT network traffic schema based on MUDactivity concepts.

    The schema models three entities following the MUD (Manufacturer Usage Description)
    framework:

    1. iot_device: Individual IoT devices with their identifiers and characteristics
    2. service_flow: MUD-defined service flows (ACE entries) with flow-level behaviors
    3. packet_activity: Packet-level measurements within each service flow

    This maps to the MUDactivity behavior grammar which describes:
    - flow-info: Device, MUD URL, endpoint address, protocol, port
    - flow-activity: pkt-count, pkt-size, pkt-iat, flow-iat

    Args:
        n_devices: Number of IoT devices to simulate
        n_service_flows: Number of service flows across all devices
        n_packets: Number of packet activity records
        global_start_time: Start of the simulation period
        global_end_time: End of the simulation period
        global_time_interval: Time granularity for measurements

    Returns:
        DataSchema: Complete schema for IoT network traffic generation
    """

    # ==========================================================================
    # ENTITY 1: iot_device
    # Represents individual IoT devices in the network
    # Maps to the "device" field in MUDactivity's flow-info section
    # ==========================================================================
    iot_device = Entity(
        name="iot_device",
        cardinality=n_devices,
        columns=[
            # Device identifier (unique per device)
            Column(
                name="Device_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="IOT_DEV_{id}"),
                ),
            ),
            # Device type - based on devices studied in the paper
            Column(
                name="Device_Type",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[
                            "Amazon_Echo",
                            "Awair_Air_Quality",
                            "LIFX_Lightbulb",
                            "Pixstar_Photo_Frame",
                            "Ring_Doorbell",
                            "Samsung_Smartcam",
                            "TPLink_Camera",
                            "Triby_Speaker",
                            "Withings_Baby_Monitor",
                            "Withings_Sleep_Sensor",
                        ],
                        with_replacement=True,
                        seed=100,
                    ),
                ),
            ),
            # MUD URL - reference to the device's MUD profile
            Column(
                name="MUD_URL",
                data_type="string",
                column_type=ColumnType.DERIVED,
                column_category_type=ColumnCategoryType.METADATA,
                derivation=Derivation(
                    function_type=DerivationFunctionType.MAP_VALUES,
                    dependent_columns=["Device_Type"],
                    params=MapValuesParams(
                        mapping=[
                            {"from": "Amazon_Echo", "to":  "https://mud.amazon.com/echo"},
                             {"from": "Awair_Air_Quality", "to": "https://mud.awair.is/airquality"},
                              {"from": "LIFX_Lightbulb", "to": "https://mud.lifx.com/lightbulb"},
                               {"from": "Pixstar_Photo_Frame", "to": "https://mud.pixstar.com/frame"},
                                {"from": "Ring_Doorbell", "to": "https://mud.ring.com/doorbell"},
                                 { "from": "Samsung_Smartcam", "to": "https://mud.samsung.com/smartcam"},
                                  {"from": "TPLink_Camera", "to": "https://mud.tplink.com/camera"},
                                   {"from": "Triby_Speaker", "to": "https://mud.triby.io/speaker"},
                                    {"from": "Withings_Baby_Monitor", "to": "https://mud.withings.com/babymonitor"},
                                     {"from": "Withings_Sleep_Sensor", "to": "https://mud.withings.com/sleepsensor"},
                            ],
                        #default_value="https://mud.unknown.com/device",
                    ),
                ),
            ),
            # Device MAC address
            Column(
                name="MAC_Address",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="AA:BB:CC:DD:EE:{id}"),
                ),
            ),
            # IP address assigned to device
            Column(
                name="IP_Address",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="192.168.1.{id}"),
                ),
            ),
        ],
    )

    # ==========================================================================
    # ENTITY 2: service_flow
    # Represents MUD service flows (Access Control Entries)
    # Maps to the MUDactivity flow-info section
    # ==========================================================================
    service_flow = Entity(
        name="service_flow",
        cardinality=n_service_flows,
        timestamp=Timestamp(column_name="Flow_Timestamp", data_type="timestamp"),
        columns=[
            # Service flow identifier
            Column(
                name="Flow_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="FLOW_{id}"),
                ),
            ),
            # Foreign key to iot_device
            Column(
                name="Device_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            # Endpoint address (cloud server)
            Column(
                name="Endpoint_Address",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[
                            "dcape-na.amazon.com",
                            "ota.awair.is",
                            "timeserver.awair.is",
                            "messaging.awair.is",
                            "broker.lifx.co",
                            "api.pixstar.com",
                            "fw.ring.com",
                            "samsungsmartcam.com",
                            "n-devs.tplinkcloud.com",
                            "triby.invoxia.io",
                            "scalews.withings.com",
                        ],
                        with_replacement=True,
                        seed=200,
                    ),
                ),
            ),
            # Transport protocol (6=TCP, 17=UDP)
            Column(
                name="Transport_Protocol",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[6, 17],  # TCP=6, UDP=17
                        weights=[0.8, 0.2],  # 80% TCP, 20% UDP
                        with_replacement=True,
                        seed=201,
                    ),
                ),
            ),
            # Transport port
            Column(
                name="Transport_Port",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[443, 8883, 80, 123, 53],  # HTTPS, MQTT, HTTP, NTP, DNS
                        weights=[0.5, 0.2, 0.15, 0.1, 0.05],
                        with_replacement=True,
                        seed=202,
                    ),
                ),
            ),
            # =================================================================
            # FLOW-LEVEL ACTIVITY METRICS (from MUDactivity flow-activity)
            # =================================================================

            # Packet count per flow - modeled with state machine
            # Based on Figure 4(b) from the paper showing packet count alternating
            # between deterministic states (e.g., 5 and 6 packets)
            Column(
                name="Packet_Count",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.STATE_MACHINE,
                    params=StateMachineParams(
                        column_name="Packet_Count",
                        initial_state="normal_count",
                        states=[
                            {"name": "normal_count", "value": 5},
                            {"name": "extended_count", "value": 6},
                            {"name": "burst_count", "value": 10},
                        ],
                        transitions=[
                            Transition(
                                trigger="tick",
                                source="normal_count",
                                dest="extended_count",
                                probability=0.44,
                            ),
                            Transition(
                                trigger="tick",
                                source="normal_count",
                                dest="normal_count",
                                probability=0.50,
                            ),
                            Transition(
                                trigger="tick",
                                source="normal_count",
                                dest="burst_count",
                                probability=0.06,
                            ),
                            Transition(
                                trigger="tick",
                                source="extended_count",
                                dest="normal_count",
                                probability=0.56,
                            ),
                            Transition(
                                trigger="tick",
                                source="extended_count",
                                dest="extended_count",
                                probability=0.37,
                            ),
                            Transition(
                                trigger="tick",
                                source="extended_count",
                                dest="burst_count",
                                probability=0.07,
                            ),
                            Transition(
                                trigger="tick",
                                source="burst_count",
                                dest="normal_count",
                                probability=0.80,
                            ),
                            Transition(
                                trigger="tick",
                                source="burst_count",
                                dest="extended_count",
                                probability=0.20,
                            ),
                        ],
                        #seed=300,
                    ),
                ),
            ),
            # Flow inter-arrival time (seconds)
            # Based on Figure 4(a) showing state machine with short (Δ1) and long (Δ2) IATs
            Column(
                name="Flow_IAT_State",
                data_type="string",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.STATE_MACHINE,
                    params=StateMachineParams(
                        column_name="Flow_IAT_State",
                        initial_state="short_iat",
                        states=[
                            {"name": "short_iat", "value": "short"},
                            {"name": "long_iat", "value": "long"},
                        ],
                        transitions=[
                            # Short IAT always followed by long IAT
                            Transition(
                                trigger="tick",
                                source="short_iat",
                                dest="long_iat",
                                probability=1.0,
                            ),
                            # Long IAT usually followed by short, sometimes stays long
                            Transition(
                                trigger="tick",
                                source="long_iat",
                                dest="short_iat",
                                probability=0.93,
                            ),
                            Transition(
                                trigger="tick",
                                source="long_iat",
                                dest="long_iat",
                                probability=0.07,
                            ),
                        ],
                        seed=301,
                    ),
                ),
            ),
            # Flow IAT in milliseconds - timeseries with patterns
            Column(
                name="Flow_IAT_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=3300000.0,  # ~55 minutes base (from paper: 3000-3600s)
                        min_value=60000.0,     # 1 minute minimum
                        max_value=21600000.0,  # 6 hours maximum
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.3,
                        noise_level=0.25,
                        spike_probability=0.05,
                        spike_magnitude=0.5,
                        interval_minutes=1,
                        seed=302,
                    ),
                ),
            ),
            # Total bytes transferred in flow
            Column(
                name="Flow_Bytes",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=2500.0,
                        min_value=500.0,
                        max_value=15000.0,
                        seasonality_type="none",
                        noise_level=0.4,
                        spike_probability=0.08,
                        spike_magnitude=0.6,
                        interval_minutes=1,
                        seed=303,
                    ),
                ),
            ),
            # Flow duration in milliseconds
            Column(
                name="Flow_Duration_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=500.0,
                        min_value=50.0,
                        max_value=2000.0,
                        seasonality_type="none",
                        noise_level=0.3,
                        spike_probability=0.05,
                        spike_magnitude=0.4,
                        interval_minutes=1,
                        seed=304,
                    ),
                ),
            ),
        ],
    )

    # ==========================================================================
    # ENTITY 3: packet_activity
    # Represents packet-level measurements within service flows
    # Maps to pkt-size and pkt-iat attributes from MUDactivity
    # ==========================================================================
    packet_activity = Entity(
        name="packet_activity",
        cardinality=n_packets,
        timestamp=Timestamp(column_name="Packet_Timestamp", data_type="timestamp"),
        columns=[
            # Packet identifier
            Column(
                name="Packet_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="PKT_{id}"),
                ),
            ),
            # Foreign key to service_flow
            Column(
                name="Flow_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            # Packet index within flow (1, 2, 3, ...)
            Column(
                name="Packet_Index",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        weights=[0.18, 0.17, 0.16, 0.15, 0.14, 0.08, 0.05, 0.04, 0.02, 0.01],
                        with_replacement=True,
                        seed=400,
                    ),
                ),
            ),
            # Packet direction (0=inbound, 1=outbound)
            Column(
                name="Direction",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[0, 1],
                        weights=[0.45, 0.55],
                        with_replacement=True,
                        seed=401,
                    ),
                ),
            ),
            # =================================================================
            # PACKET-LEVEL MEASUREMENTS (from MUDactivity)
            # =================================================================

            # Packet size - based on Figure 2(b) showing deterministic patterns
            # e.g., [287, 188, 1514, 188, 91] bytes for Amazon Echo TCP/443
            Column(
                name="Packet_Size",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.STATE_MACHINE,
                    params=StateMachineParams(
                        column_name="Packet_Size",
                        initial_state="small",
                        states=[
                            {"name": "small", "value": 91},      # ACK/small response
                            {"name": "medium", "value": 188},    # Typical data packet
                            {"name": "large", "value": 287},     # Request packet
                            {"name": "max_mtu", "value": 1514},  # Full MTU packet
                        ],
                        transitions=[
                            # Small packets often followed by medium
                            Transition(trigger="tick", source="small", dest="medium", probability=0.50),
                            Transition(trigger="tick", source="small", dest="large", probability=0.30),
                            Transition(trigger="tick", source="small", dest="small", probability=0.20),
                            # Medium packets - varied transitions
                            Transition(trigger="tick", source="medium", dest="small", probability=0.35),
                            Transition(trigger="tick", source="medium", dest="large", probability=0.25),
                            Transition(trigger="tick", source="medium", dest="max_mtu", probability=0.15),
                            Transition(trigger="tick", source="medium", dest="medium", probability=0.25),
                            # Large packets often followed by max_mtu or medium
                            Transition(trigger="tick", source="large", dest="max_mtu", probability=0.40),
                            Transition(trigger="tick", source="large", dest="medium", probability=0.40),
                            Transition(trigger="tick", source="large", dest="small", probability=0.20),
                            # Max MTU packets followed by smaller
                            Transition(trigger="tick", source="max_mtu", dest="medium", probability=0.50),
                            Transition(trigger="tick", source="max_mtu", dest="small", probability=0.40),
                            Transition(trigger="tick", source="max_mtu", dest="max_mtu", probability=0.10),
                        ],
                        seed=402,
                    ),
                ),
            ),
            # Packet inter-arrival time (IAT) in milliseconds
            # Based on Figure 2(c) - Folded Cauchy distribution, mostly (0, 300] ms
            Column(
                name="Packet_IAT_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=50.0,     # 50ms base IAT
                        min_value=1.0,       # 1ms minimum
                        max_value=500.0,     # 500ms maximum (paper shows most < 400ms)
                        seasonality_type="none",
                        noise_level=0.5,     # High variance as seen in paper
                        spike_probability=0.1,
                        spike_magnitude=0.6,
                        interval_minutes=1,
                        seed=403,
                    ),
                ),
            ),
            # TCP flags (for TCP packets)
            Column(
                name="TCP_Flags",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["SYN", "SYN-ACK", "ACK", "PSH-ACK", "FIN-ACK", "RST"],
                        weights=[0.05, 0.05, 0.40, 0.40, 0.08, 0.02],
                        with_replacement=True,
                        seed=404,
                    ),
                ),
            ),
            # Payload indicator (1=has payload, 0=no payload)
            Column(
                name="Has_Payload",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[0, 1],
                        weights=[0.35, 0.65],
                        with_replacement=True,
                        seed=405,
                    ),
                ),
            ),
        ],
    )

    # ==========================================================================
    # ENTITY RELATIONSHIPS
    # ==========================================================================
    relationships = [
        # iot_device -> service_flow (one-to-many)
        # One device can have multiple service flows (MUD ACE entries)
        EntityRelationship(
            parent_entity="iot_device",
            child_entity="service_flow",
            relationship_type=EntityRelationshipType.ONE_TO_MANY,
            join_columns={"Device_ID": "Device_ID"},
        ),
        # service_flow -> packet_activity (one-to-many)
        # One flow contains multiple packets
        EntityRelationship(
            parent_entity="service_flow",
            child_entity="packet_activity",
            relationship_type=EntityRelationshipType.ONE_TO_MANY,
            join_columns={"Flow_ID": "Flow_ID"},
        ),
    ]

    # ==========================================================================
    # GLOBAL TIMESTAMP CONFIGURATION
    # ==========================================================================
    global_timestamp = GlobalTimestamp(
        t_start=global_start_time,
        t_end=global_end_time,
        time_interval=global_time_interval,
    )

    return DataSchema(
        entities=[iot_device, service_flow, packet_activity],
        entity_relationships=relationships,
        global_timestamp=global_timestamp,
    )


def create_single_device_flow_schema(
    device_type: str = "Amazon_Echo",
    endpoint: str = "dcape-na.amazon.com",
    transport_protocol: int = 6,  # TCP
    transport_port: int = 443,
    n_flows: int = 100,
    global_start_time: str = "2025-01-01T00:00:00Z",
    global_end_time: str = "2025-01-03T00:00:00Z",
    global_time_interval: str = "15min",
) -> DataSchema:
    """
    Create a focused schema for a single IoT device and its service flow.

    This mirrors the MUDactivity JSON structure more closely, focusing on
    detailed packet-level behavior for a specific MUD flow.

    Based on the paper's analysis of specific flows like:
    - Amazon Echo TCP/443 to dcape-na.amazon.com
    - Awair TCP/443 to ota.awair.is

    Args:
        device_type: Type of IoT device
        endpoint: Cloud endpoint address
        transport_protocol: 6 for TCP, 17 for UDP
        transport_port: Service port number
        n_flows: Number of flow instances to generate
        global_start_time: Simulation start time
        global_end_time: Simulation end time
        global_time_interval: Time interval between measurements

    Returns:
        DataSchema: Schema for single device flow analysis
    """

    # Single flow entity with detailed MUDactivity attributes
    mud_flow = Entity(
        name="mud_flow",
        cardinality=n_flows,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # =================================================================
            # FLOW-INFO SECTION (from MUDactivity schema)
            # =================================================================
            Column(
                name="Flow_Instance_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="FLOW_INST_{id}"),
                ),
            ),
            # Device type (constant for this schema)
            Column(
                name="Device_Type",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[device_type],
                        with_replacement=True,
                        seed=100,
                    ),
                ),
            ),
            # Endpoint address
            Column(
                name="Endpoint_Address",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[endpoint],
                        with_replacement=True,
                        seed=101,
                    ),
                ),
            ),
            # Transport protocol
            Column(
                name="Transport_Protocol",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[transport_protocol],
                        with_replacement=True,
                        seed=102,
                    ),
                ),
            ),
            # Transport port
            Column(
                name="Transport_Port",
                data_type="int64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[transport_port],
                        with_replacement=True,
                        seed=103,
                    ),
                ),
            ),
            # =================================================================
            # FLOW-ACTIVITY SECTION (from MUDactivity schema)
            # =================================================================

            # pkt-count: Packet count per flow
            # State machine based on paper's Figure 4(b) for Awair
            Column(
                name="Pkt_Count",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.STATE_MACHINE,
                    params=StateMachineParams(
                        column_name="Pkt_Count",
                        initial_state="count_5",
                        states=[
                            {"name": "count_5", "value": 5},
                            {"name": "count_6", "value": 6},
                        ],
                        transitions=[
                            # Alternating pattern as seen in paper
                            Transition(
                                trigger="tick",
                                source="count_5",
                                dest="count_6",
                                probability=0.56,
                            ),
                            Transition(
                                trigger="tick",
                                source="count_5",
                                dest="count_5",
                                probability=0.44,
                            ),
                            Transition(
                                trigger="tick",
                                source="count_6",
                                dest="count_5",
                                probability=0.93,
                            ),
                            Transition(
                                trigger="tick",
                                source="count_6",
                                dest="count_6",
                                probability=0.07,
                            ),
                        ],
                        seed=200,
                    ),
                ),
            ),
            # flow-iat: Flow inter-arrival time state
            # Based on Figure 4(a) - alternating short/long IAT pattern
            Column(
                name="Flow_IAT_State",
                data_type="string",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.STATE_MACHINE,
                    params=StateMachineParams(
                        column_name="Flow_IAT_State",
                        initial_state="delta_1",
                        states=[
                            {"name": "delta_1", "value": "short"},  # Δ1: short IAT
                            {"name": "delta_2", "value": "long"},   # Δ2: long IAT
                        ],
                        transitions=[
                            # From paper: short always followed by long
                            Transition(
                                trigger="tick",
                                source="delta_1",
                                dest="delta_2",
                                probability=1.0,
                            ),
                            # Long usually followed by short
                            Transition(
                                trigger="tick",
                                source="delta_2",
                                dest="delta_1",
                                probability=0.93,
                            ),
                            Transition(
                                trigger="tick",
                                source="delta_2",
                                dest="delta_2",
                                probability=0.07,
                            ),
                        ],
                        seed=201,
                    ),
                ),
            ),
            # flow-iat: Short IAT value (Δ1) in milliseconds
            # Based on Figure 4(c) - typically < 1 second
            Column(
                name="Flow_IAT_Short_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=200.0,    # 200ms base
                        min_value=50.0,      # 50ms minimum
                        max_value=500.0,     # 500ms maximum
                        seasonality_type="none",
                        noise_level=0.3,
                        spike_probability=0.05,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=202,
                    ),
                ),
            ),
            # flow-iat: Long IAT value (Δ2) in hours
            # Based on Figure 4(d) - can span several hours
            Column(
                name="Flow_IAT_Long_hours",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=12.0,     # 12 hours base
                        min_value=1.0,       # 1 hour minimum
                        max_value=30.0,      # 30 hours maximum
                        seasonality_type="peak_offpeak",
                        peak_start_hour=0,
                        peak_end_hour=6,
                        seasonality_strength=0.2,
                        noise_level=0.4,
                        spike_probability=0.03,
                        spike_magnitude=0.5,
                        interval_minutes=15,
                        seed=203,
                    ),
                ),
            ),
            # pkt-iat: Average packet IAT within flow (milliseconds)
            # Based on Figure 2(c) - Folded Cauchy distribution
            Column(
                name="Avg_Pkt_IAT_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=80.0,
                        min_value=10.0,
                        max_value=400.0,
                        seasonality_type="none",
                        noise_level=0.45,
                        spike_probability=0.08,
                        spike_magnitude=0.4,
                        interval_minutes=15,
                        seed=204,
                    ),
                ),
            ),
            # pkt-size: Total bytes in flow
            Column(
                name="Total_Bytes",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=2300.0,   # ~2.3KB based on typical packet patterns
                        min_value=800.0,
                        max_value=8000.0,
                        seasonality_type="none",
                        noise_level=0.35,
                        spike_probability=0.06,
                        spike_magnitude=0.5,
                        interval_minutes=15,
                        seed=205,
                    ),
                ),
            ),
            # Flow duration in milliseconds
            Column(
                name="Flow_Duration_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=400.0,
                        min_value=100.0,
                        max_value=1500.0,
                        seasonality_type="none",
                        noise_level=0.4,
                        spike_probability=0.05,
                        spike_magnitude=0.4,
                        interval_minutes=15,
                        seed=206,
                    ),
                ),
            ),
        ],
    )

    # Global timestamp configuration
    global_timestamp = GlobalTimestamp(
        t_start=global_start_time,
        t_end=global_end_time,
        time_interval=global_time_interval,
    )

    return DataSchema(
        entities=[mud_flow],
        entity_relationships=[],
        global_timestamp=global_timestamp,
    )


# Example usage and visualization helpers
def get_device_configs():
    """
    Return configuration for the 10 IoT devices studied in the paper.

    Based on Appendix A of the paper listing devices analyzed.
    """
    return {
        "Amazon_Echo": {
            "flows": [
                {"endpoint": "dcape-na.amazon.com", "protocol": 6, "port": 443},
                {"endpoint": "device-metrics-us.amazon.com", "protocol": 6, "port": 443},
            ]
        },
        "Awair_Air_Quality": {
            "flows": [
                {"endpoint": "ota.awair.is", "protocol": 6, "port": 443},
                {"endpoint": "timeserver.awair.is", "protocol": 6, "port": 443},
                {"endpoint": "messaging.awair.is", "protocol": 6, "port": 8883},
            ]
        },
        "LIFX_Lightbulb": {
            "flows": [
                {"endpoint": "broker.lifx.co", "protocol": 6, "port": 8883},
            ]
        },
        "Pixstar_Photo_Frame": {
            "flows": [
                {"endpoint": "api.pixstar.com", "protocol": 6, "port": 443},
            ]
        },
        "Ring_Doorbell": {
            "flows": [
                {"endpoint": "fw.ring.com", "protocol": 6, "port": 443},
                {"endpoint": "ntp.ring.com", "protocol": 17, "port": 123},
            ]
        },
        "Samsung_Smartcam": {
            "flows": [
                {"endpoint": "samsungsmartcam.com", "protocol": 6, "port": 443},
            ]
        },
        "TPLink_Camera": {
            "flows": [
                {"endpoint": "n-devs.tplinkcloud.com", "protocol": 6, "port": 443},
            ]
        },
        "Triby_Speaker": {
            "flows": [
                {"endpoint": "triby.invoxia.io", "protocol": 6, "port": 443},
            ]
        },
        "Withings_Baby_Monitor": {
            "flows": [
                {"endpoint": "scalews.withings.com", "protocol": 6, "port": 443},
            ]
        },
        "Withings_Sleep_Sensor": {
            "flows": [
                {"endpoint": "scalews.withings.com", "protocol": 6, "port": 443},
            ]
        },
    }
