"""
PCAP Template Module for generating various traffic patterns using Rockfish.

This module provides schema creation functions for:
- Normal traffic templates (6 types)
- Anomalous traffic templates (6 types)
- Suspicious traffic templates (5 types)

Each template creates a DataSchema configured for specific traffic patterns
with appropriate state machines, packet distributions, and timing.
"""

from typing import Optional, Tuple
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
    SequentialIntParams,
    StateMachineParams,
    Timestamp,
    Transition,
    UniformDistParams,
    ExponentialDistParams,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_client_host_entity(n_hosts: int = 100, internal_only: bool = True) -> Entity:
    """Create a client host entity with internal IP addresses."""
    columns = [
        Column(
            name="client_id",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.ID,
                params=IDParams(template_str="CLIENT_{id}"),
            ),
        ),
        Column(
            name="client_ip_octet_1",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=[10, 10, 10, 192] if internal_only else [8, 13, 17, 20, 34],
                    with_replacement=True,
                ),
            ),
        ),
        Column(
            name="client_ip_octet_2",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=[1, 2, 3, 4, 5, 10, 20, 168],
                    with_replacement=True,
                ),
            ),
        ),
        Column(
            name="client_ip_octet_3",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.UNIFORM_DIST,
                params=UniformDistParams(lower=0, upper=255),
            ),
        ),
        Column(
            name="client_ip_octet_4",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.UNIFORM_DIST,
                params=UniformDistParams(lower=1, upper=254),
            ),
        ),
        Column(
            name="client_device_type",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=["workstation", "workstation", "workstation", "laptop", "laptop", "mobile", "iot_device"],
                    with_replacement=True,
                ),
            ),
        ),
    ]

    return Entity(
        name="client_host",
        cardinality=n_hosts,
        columns=columns,
    )


def _create_server_host_entity(n_hosts: int = 50, external: bool = True) -> Entity:
    """Create a server host entity with external or internal IP addresses."""
    if external:
        ip_values = [8, 13, 17, 20, 34, 35, 52, 54, 64, 72, 93, 104, 142, 172, 199, 204]
    else:
        ip_values = [10, 10, 10, 192, 192, 172]

    columns = [
        Column(
            name="server_id",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.ID,
                params=IDParams(template_str="SERVER_{id}"),
            ),
        ),
        Column(
            name="server_ip_octet_1",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=ip_values,
                    with_replacement=True,
                ),
            ),
        ),
        Column(
            name="server_ip_octet_2",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.UNIFORM_DIST,
                params=UniformDistParams(lower=0, upper=255),
            ),
        ),
        Column(
            name="server_ip_octet_3",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.UNIFORM_DIST,
                params=UniformDistParams(lower=0, upper=255),
            ),
        ),
        Column(
            name="server_ip_octet_4",
            data_type="int64",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.UNIFORM_DIST,
                params=UniformDistParams(lower=1, upper=254),
            ),
        ),
        Column(
            name="server_type",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=["web_server", "web_server", "api_server", "database", "mail_server", "file_server", "cdn"],
                    with_replacement=True,
                ),
            ),
        ),
    ]

    return Entity(
        name="server_host",
        cardinality=n_hosts,
        columns=columns,
    )


def _create_service_entity(
    services: list[str],
    port_mapping: dict[str, int],
) -> Entity:
    """Create a service entity with specified services and ports."""
    columns = [
        Column(
            name="service_id",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.ID,
                params=IDParams(template_str="SVC_{id}"),
            ),
        ),
        Column(
            name="service_name",
            data_type="string",
            column_type=ColumnType.INDEPENDENT,
            column_category_type=ColumnCategoryType.METADATA,
            domain=Domain(
                type=DomainType.CATEGORICAL,
                params=CategoricalParams(
                    values=services,
                    with_replacement=False,
                ),
            ),
        ),
        Column(
            name="server_port",
            data_type="int64",
            column_type=ColumnType.DERIVED,
            column_category_type=ColumnCategoryType.METADATA,
            derivation=Derivation(
                function_type=DerivationFunctionType.MAP_VALUES,
                dependent_columns=["service_name"],
                params=MapValuesParams(
                    mapping=[{"from": k, "to": str(v)} for k, v in port_mapping.items()],
                    default="80",
                ),
            ),
        ),
    ]

    return Entity(
        name="service",
        cardinality=len(services),
        columns=columns,
    )


def _create_standard_relationships() -> list[EntityRelationship]:
    """Create standard entity relationships."""
    return [
        EntityRelationship(
            parent_entity="client_host",
            child_entity="tcp_session",
            relationship_type=EntityRelationshipType.ONE_TO_MANY,
            join_columns={"client_id": "fk_client_id"},
        ),
        EntityRelationship(
            parent_entity="server_host",
            child_entity="tcp_session",
            relationship_type=EntityRelationshipType.ONE_TO_MANY,
            join_columns={"server_id": "fk_server_id"},
        ),
        EntityRelationship(
            parent_entity="service",
            child_entity="tcp_session",
            relationship_type=EntityRelationshipType.ONE_TO_MANY,
            join_columns={"service_id": "fk_service_id"},
        ),
    ]


def _create_global_timestamp(
    t_start: str = "2025-03-01T09:00:00+00:00",
    t_end: str = "2025-03-01T17:00:00+00:00",
    time_interval: str = "1min",
) -> GlobalTimestamp:
    """Create global timestamp configuration."""
    return GlobalTimestamp(
        t_start=t_start,
        t_end=t_end,
        time_interval=time_interval,
    )


# =============================================================================
# NORMAL FLOW TEMPLATES
# =============================================================================

def create_normal_web_browsing_schema(
    n_sessions: int = 100,
    n_client_hosts: int = 50,
    n_server_hosts: int = 20,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal HTTP/HTTPS web browsing sessions.

    Characteristics:
    - Full TCP handshake
    - 5-15 data packets (request + response)
    - Graceful close
    - Mix of HTTP (80) and HTTPS (443)
    """
    services = ["HTTP", "HTTPS", "HTTPS", "HTTPS"]  # Weighted towards HTTPS
    port_mapping = {"HTTP": 80, "HTTPS": 443}

    # State machine for normal web browsing
    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"data_packets_sent": False},
                transitions=[
                    # Handshake
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Data transfer (5-15 packets typically)
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.30, context_updates={"data_packets_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40, context_updates={"data_packets_sent": True}),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    # Graceful close
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="WEB_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=600))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[16384, 32768, 65535, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 64, 128, 128, 255], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "30s"),
    )


def create_normal_api_calls_schema(
    n_sessions: int = 100,
    n_client_hosts: int = 30,
    n_server_hosts: int = 10,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal REST API request/response patterns.

    Characteristics:
    - Full handshake
    - 1-3 request packets
    - 1-5 response packets
    - Quick close
    """
    services = ["HTTPS", "HTTP"]
    port_mapping = {"HTTP": 80, "HTTPS": 443}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"request_sent": False, "response_received": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Short request/response cycle
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.25, context_updates={"request_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35, context_updates={"response_received": True}),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.25),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="API_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=300))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "10s"),
    )


def create_normal_ssh_session_schema(
    n_sessions: int = 50,
    n_client_hosts: int = 20,
    n_server_hosts: int = 10,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal interactive SSH traffic.

    Characteristics:
    - Full handshake
    - Sustained bidirectional traffic (many small packets)
    - Long session duration
    - Graceful close
    """
    services = ["SSH"]
    port_mapping = {"SSH": 22}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"packets_exchanged": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Long sustained data transfer (interactive commands)
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35, context_updates={"packets_exchanged": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40, context_updates={"packets_exchanged": True}),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.20),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.05),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="SSH_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=100))),  # Small packets for interactive
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal servers for SSH
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "5s"),  # More frequent packets
    )


def create_normal_file_transfer_schema(
    n_sessions: int = 50,
    n_client_hosts: int = 30,
    n_server_hosts: int = 10,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal FTP/SMB file download patterns.

    Characteristics:
    - Full handshake
    - Many large S2C packets (server sending file)
    - Small C2S packets (ACKs and commands)
    - Graceful close
    """
    services = ["FTP", "SMB"]
    port_mapping = {"FTP": 21, "SMB": 445}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"transfer_started": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Heavy S2C traffic for file downloads
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.10),  # Commands
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.55, context_updates={"transfer_started": True}),  # File data
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.25),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.10),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="FILE_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=1000))),  # Large packets
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535, 65535, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal file servers
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "20s"),
    )


def create_normal_database_query_schema(
    n_sessions: int = 100,
    n_client_hosts: int = 20,
    n_server_hosts: int = 5,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal MySQL/PostgreSQL query patterns.

    Characteristics:
    - Full handshake
    - Small request packet (query)
    - Variable response size (result set)
    - Quick close or connection pooling
    """
    services = ["MYSQL", "POSTGRESQL"]
    port_mapping = {"MYSQL": 3306, "POSTGRESQL": 5432}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"query_sent": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Query/response pattern
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.20, context_updates={"query_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.45),  # Result set
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.20),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="DB_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=400))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal DB servers
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "15s"),
    )


def create_normal_email_schema(
    n_sessions: int = 50,
    n_client_hosts: int = 30,
    n_server_hosts: int = 5,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for normal SMTP/IMAP email traffic.

    Characteristics:
    - Full handshake
    - Protocol-specific exchanges (HELO, AUTH, DATA, etc.)
    - Moderate packet sizes
    - Graceful close
    """
    services = ["SMTP", "IMAP"]
    port_mapping = {"SMTP": 25, "IMAP": 143}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"mail_sent": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Email protocol exchange
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.30, context_updates={"mail_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.20),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="MAIL_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=200))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[16384, 32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),  # External mail servers
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "30s"),
    )


# =============================================================================
# ANOMALOUS FLOW TEMPLATES
# =============================================================================

def create_anomaly_port_scan_schema(
    n_sessions: int = 200,
    n_client_hosts: int = 5,  # Few attackers
    n_server_hosts: int = 50,  # Many targets
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for port scanning attack patterns.

    Characteristics:
    - SYN -> RST (repeated across many ports)
    - Very short sessions (1-2 packets)
    - Many destination ports from same source
    - No data transfer
    """
    # Use a special "SCAN" service that maps to various ports
    services = ["SCAN"]
    port_mapping = {"SCAN": 0}  # Port will vary

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "RESET"],
                terminal_states=["RESET"],
                context_variables={},
                transitions=[
                    # Port scan: SYN -> RST (port closed) or no response (filtered)
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="RST", source="SYN_SENT", dest="RESET", probability=0.70),  # Port closed
                    # Some scans get no response (filtered) - modeled as immediate terminal
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="SCAN_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            # Target port varies widely (scanning behavior)
            Column(name="target_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=1, upper=1024))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=40, upper=60))),  # Only control packets
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[1024, 2048, 4096], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T02:00:00+00:00", "2025-03-01T02:30:00+00:00")  # Short burst

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal targets
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "100ms"),  # Very fast
    )


def create_anomaly_syn_flood_schema(
    n_sessions: int = 500,
    n_client_hosts: int = 100,  # Spoofed IPs
    n_server_hosts: int = 1,  # Single target
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for SYN flood attack patterns.

    Characteristics:
    - SYN only (no SYN-ACK response or ignored)
    - High volume, single target
    - Many spoofed source IPs
    - No data transfer
    """
    services = ["HTTP", "HTTPS"]
    port_mapping = {"HTTP": 80, "HTTPS": 443}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT"],
                terminal_states=["SYN_SENT"],  # Never completes handshake
                context_variables={},
                transitions=[
                    # SYN flood: only SYN packets, no completion
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="FLOOD_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=1024, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=40, upper=60))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128, 255], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T03:00:00+00:00", "2025-03-01T03:05:00+00:00")  # 5-minute attack

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts, internal_only=False),  # Spoofed external IPs
            _create_server_host_entity(n_server_hosts, external=False),  # Target server
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "10ms"),  # Very high rate
    )


def create_anomaly_large_upload_schema(
    n_sessions: int = 20,
    n_client_hosts: int = 5,
    n_server_hosts: int = 3,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for data exfiltration patterns.

    Characteristics:
    - Full handshake
    - Abnormally large C2S data volume (upload)
    - External destination
    - Long session duration
    """
    services = ["HTTPS", "FTP"]
    port_mapping = {"HTTPS": 443, "FTP": 21}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"data_exfiltrated": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Heavy C2S traffic (exfiltration)
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.60, context_updates={"data_exfiltrated": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.10),  # Minimal response
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.25),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.05),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="EXFIL_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=1400, upper=1500))),  # Max size packets
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T23:00:00+00:00", "2025-03-02T01:00:00+00:00")  # Night time

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),  # External dropsite
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "5s"),
    )


def create_anomaly_beaconing_schema(
    n_sessions: int = 50,
    n_client_hosts: int = 10,
    n_server_hosts: int = 2,  # C2 servers
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for C2 beaconing patterns.

    Characteristics:
    - Regular intervals between sessions
    - Small, fixed-size packets
    - Long overall duration
    - External destination
    """
    services = ["HTTPS"]  # Hide in encrypted traffic
    port_mapping = {"HTTPS": 443}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"beacon_sent": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Beacon pattern: small check-in, small response
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40, context_updates={"beacon_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.10),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="BEACON_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=50, upper=150))),  # Small, consistent
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T00:00:00+00:00", "2025-03-01T23:59:00+00:00")  # All day

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),  # External C2
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "60s"),  # Regular interval
    )


def create_anomaly_dns_tunnel_schema(
    n_sessions: int = 100,
    n_client_hosts: int = 5,
    n_server_hosts: int = 2,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for DNS tunneling patterns.

    Characteristics:
    - DNS port (53)
    - Unusually large packets for DNS
    - High query frequency
    - External DNS server
    """
    services = ["DNS"]
    port_mapping = {"DNS": 53}

    # DNS is typically UDP, but we model TCP for the schema
    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"tunnel_active": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # DNS tunnel: unusual large "queries"
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40, context_updates={"tunnel_active": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.10),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="DNSTUN_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=200, upper=512))),  # Large for DNS
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T00:00:00+00:00", "2025-03-01T23:59:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),  # External DNS
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "5s"),  # Frequent queries
    )


def create_anomaly_slow_loris_schema(
    n_sessions: int = 100,
    n_client_hosts: int = 20,
    n_server_hosts: int = 1,  # Single target
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for Slow Loris attack patterns.

    Characteristics:
    - Incomplete HTTP requests
    - Connection held open for long time
    - Very slow data rate
    - Many concurrent connections
    """
    services = ["HTTP"]
    port_mapping = {"HTTP": 80}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT", "RESET"],
                terminal_states=["TIME_WAIT", "RESET"],
                context_variables={"partial_request": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Slow drip of data (incomplete requests)
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.70, context_updates={"partial_request": True}),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.25),
                    # Eventually times out or gets RST
                    Transition(trigger="RST", source="ESTABLISHED", dest="RESET", probability=0.05),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="SLOWL_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=50, upper=100))),  # Tiny packets
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T10:00:00+00:00", "2025-03-01T12:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "30s"),  # Very slow rate
    )


# =============================================================================
# SUSPICIOUS FLOW TEMPLATES
# =============================================================================

def create_suspicious_brute_force_schema(
    n_sessions: int = 200,
    n_client_hosts: int = 3,  # Few attackers
    n_server_hosts: int = 5,  # Few targets
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for authentication brute force patterns.

    Characteristics:
    - Multiple short sessions
    - Same destination, different source ports
    - SSH/RDP targets
    - Quick failures (RST after auth attempt)
    """
    services = ["SSH", "RDP"]
    port_mapping = {"SSH": 22, "RDP": 3389}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT", "RESET"],
                terminal_states=["TIME_WAIT", "RESET"],
                context_variables={"auth_attempted": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Quick auth attempt
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.30, context_updates={"auth_attempted": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.20),  # Auth response
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.10),
                    # Most fail and close quickly
                    Transition(trigger="RST", source="ESTABLISHED", dest="RESET", probability=0.30),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.10),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="BRUTE_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=100))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T01:00:00+00:00", "2025-03-01T02:00:00+00:00")  # Night time

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal targets
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "2s"),  # Fast attempts
    )


def create_suspicious_lateral_movement_schema(
    n_sessions: int = 50,
    n_client_hosts: int = 5,
    n_server_hosts: int = 20,  # Many internal targets
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for lateral movement/internal reconnaissance.

    Characteristics:
    - Internal-to-internal traffic
    - Unusual port access patterns
    - WMI/SMB/RPC protocols
    - Admin tool usage
    """
    services = ["SMB", "WMI", "RPC"]
    port_mapping = {"SMB": 445, "WMI": 135, "RPC": 135}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT", "RESET"],
                terminal_states=["TIME_WAIT", "RESET"],
                context_variables={"recon_done": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=0.90),
                    Transition(trigger="RST", source="SYN_SENT", dest="RESET", probability=0.10),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35, context_updates={"recon_done": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.30),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="RST", source="ESTABLISHED", dest="RESET", probability=0.05),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="LATERAL_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=300))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[128], with_replacement=True))),  # Windows default
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T14:00:00+00:00", "2025-03-01T16:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # All internal
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "30s"),
    )


def create_suspicious_data_staging_schema(
    n_sessions: int = 30,
    n_client_hosts: int = 10,
    n_server_hosts: int = 1,  # Staging server
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for pre-exfiltration data staging.

    Characteristics:
    - Large internal transfers to single host
    - Multiple sources aggregating to one destination
    - SMB/FTP protocols
    - Off-hours timing
    """
    services = ["SMB", "FTP"]
    port_mapping = {"SMB": 445, "FTP": 21}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"data_staged": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # Heavy C2S traffic (staging data)
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.55, context_updates={"data_staged": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.20),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.10),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="STAGE_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=1400, upper=1500))),  # Max size
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T22:00:00+00:00", "2025-03-02T02:00:00+00:00")  # Night

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=False),  # Internal staging
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "10s"),
    )


def create_suspicious_encrypted_tunnel_schema(
    n_sessions: int = 30,
    n_client_hosts: int = 5,
    n_server_hosts: int = 3,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for TLS on non-standard ports.

    Characteristics:
    - TLS handshake patterns
    - Non-standard ports (not 443)
    - External destination
    - Encrypted payload after handshake
    """
    # TLS on unusual ports
    services = ["TLS_8443", "TLS_4433", "TLS_8080"]
    port_mapping = {"TLS_8443": 8443, "TLS_4433": 4433, "TLS_8080": 8080}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"tls_established": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # TLS handshake then encrypted data
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.35, context_updates={"tls_established": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.10),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="ENCTUN_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=500))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T00:00:00+00:00", "2025-03-01T23:59:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "60s"),
    )


def create_suspicious_protocol_anomaly_schema(
    n_sessions: int = 30,
    n_client_hosts: int = 5,
    n_server_hosts: int = 5,
    time_range: Optional[Tuple[str, str]] = None,
) -> DataSchema:
    """
    Create schema for protocol mismatch patterns.

    Characteristics:
    - HTTP-like traffic on non-HTTP ports
    - Protocol indicators don't match port
    - Could indicate tunneling or misconfiguration
    """
    # HTTP on unusual ports
    services = ["HTTP_8888", "HTTP_9000", "HTTP_3128"]
    port_mapping = {"HTTP_8888": 8888, "HTTP_9000": 9000, "HTTP_3128": 3128}

    tcp_state_column = Column(
        name="tcp_state",
        data_type="string",
        column_type=ColumnType.STATEFUL,
        column_category_type=ColumnCategoryType.MEASUREMENT,
        domain=Domain(
            type=DomainType.STATE_MACHINE,
            params=StateMachineParams(
                column_name="tcp_state",
                trigger_column_name="packet_type",
                initial_state="CLOSED",
                states=["CLOSED", "SYN_SENT", "SYN_RECEIVED", "ESTABLISHED", "FIN_WAIT", "CLOSE_WAIT", "TIME_WAIT"],
                terminal_states=["TIME_WAIT"],
                context_variables={"request_sent": False},
                transitions=[
                    Transition(trigger="SYN", source="CLOSED", dest="SYN_SENT", probability=1.0),
                    Transition(trigger="SYN_ACK", source="SYN_SENT", dest="SYN_RECEIVED", probability=1.0),
                    Transition(trigger="ACK", source="SYN_RECEIVED", dest="ESTABLISHED", probability=1.0),
                    # HTTP-like request/response
                    Transition(trigger="PSH_ACK_C2S", source="ESTABLISHED", dest="ESTABLISHED", probability=0.30, context_updates={"request_sent": True}),
                    Transition(trigger="PSH_ACK_S2C", source="ESTABLISHED", dest="ESTABLISHED", probability=0.40),
                    Transition(trigger="ACK", source="ESTABLISHED", dest="ESTABLISHED", probability=0.15),
                    Transition(trigger="FIN", source="ESTABLISHED", dest="FIN_WAIT", probability=0.15),
                    Transition(trigger="FIN_ACK", source="FIN_WAIT", dest="CLOSE_WAIT", probability=1.0),
                    Transition(trigger="ACK", source="CLOSE_WAIT", dest="TIME_WAIT", probability=1.0),
                ],
            ),
        ),
    )

    session_entity = Entity(
        name="tcp_session",
        cardinality=n_sessions,
        timestamp=Timestamp(column_name="packet_timestamp"),
        columns=[
            Column(name="session_id", data_type="string", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.ID, params=IDParams(template_str="PROTO_{id}"))),
            Column(name="fk_client_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_server_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="fk_service_id", data_type="string", column_type=ColumnType.FOREIGN_KEY, column_category_type=ColumnCategoryType.METADATA),
            Column(name="client_port", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.UNIFORM_DIST, params=UniformDistParams(lower=49152, upper=65535))),
            tcp_state_column,
            Column(name="packet_size_base", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.EXPONENTIAL_DIST, params=ExponentialDistParams(scale=400))),
            Column(name="window_size", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[32768, 65535], with_replacement=True))),
            Column(name="ttl", data_type="int64", column_type=ColumnType.INDEPENDENT, column_category_type=ColumnCategoryType.METADATA,
                   domain=Domain(type=DomainType.CATEGORICAL, params=CategoricalParams(values=[64, 128], with_replacement=True))),
        ],
    )

    t_start, t_end = time_range if time_range else ("2025-03-01T09:00:00+00:00", "2025-03-01T17:00:00+00:00")

    return DataSchema(
        entities=[
            _create_client_host_entity(n_client_hosts),
            _create_server_host_entity(n_server_hosts, external=True),
            _create_service_entity(services, port_mapping),
            session_entity,
        ],
        entity_relationships=_create_standard_relationships(),
        global_timestamp=_create_global_timestamp(t_start, t_end, "30s"),
    )


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEMPLATE_REGISTRY = {
    "normal": {
        "web_browsing": create_normal_web_browsing_schema,
        "api_calls": create_normal_api_calls_schema,
        "ssh_session": create_normal_ssh_session_schema,
        "file_transfer": create_normal_file_transfer_schema,
        "database_query": create_normal_database_query_schema,
        "email": create_normal_email_schema,
    },
    "anomalous": {
        "port_scan": create_anomaly_port_scan_schema,
        "syn_flood": create_anomaly_syn_flood_schema,
        "large_upload": create_anomaly_large_upload_schema,
        "beaconing": create_anomaly_beaconing_schema,
        "dns_tunnel": create_anomaly_dns_tunnel_schema,
        "slow_loris": create_anomaly_slow_loris_schema,
    },
    "suspicious": {
        "brute_force": create_suspicious_brute_force_schema,
        "lateral_movement": create_suspicious_lateral_movement_schema,
        "data_staging": create_suspicious_data_staging_schema,
        "encrypted_tunnel": create_suspicious_encrypted_tunnel_schema,
        "protocol_anomaly": create_suspicious_protocol_anomaly_schema,
    },
}


def get_template(category: str, template_name: str):
    """Get a template function by category and name."""
    if category not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown category: {category}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    if template_name not in TEMPLATE_REGISTRY[category]:
        raise ValueError(f"Unknown template: {template_name}. Available for {category}: {list(TEMPLATE_REGISTRY[category].keys())}")
    return TEMPLATE_REGISTRY[category][template_name]


def list_templates() -> dict[str, list[str]]:
    """List all available templates by category."""
    return {category: list(templates.keys()) for category, templates in TEMPLATE_REGISTRY.items()}
