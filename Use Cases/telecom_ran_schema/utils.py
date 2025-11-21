import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter, HourLocator
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
    SampleFromColumnParams,
    StateMachineParams,
    SumParams,
    TimeseriesParams,
    Timestamp,
    Transition,
    UniformDistParams,
)


def create_telecom_ran_schema(
    n_transport_links=6,
    n_core_nodes=16,
    n_cell_sites=100,
    global_start_time="2025-01-01T00:00:00Z",
    global_end_time="2025-01-03T00:00:00Z",
    global_time_interval="15min",
) -> DataSchema:
    """Create the Snowflake Demo telecom RAN network schema."""
    # ENTITY 1: transport_link
    transport_link = Entity(
        name="transport_link",
        cardinality=n_transport_links,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Device_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["RTR_001", "RTR_002", "RTR_003", "RTR_004"],
                        with_replacement=True,
                        seed=100,
                    ),
                ),
            ),
            Column(
                name="Interface_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["eth0", "eth1", "eth2", "eth3"],
                        with_replacement=True,
                        seed=101,
                    ),
                ),
            ),
            # Measurement columns (stateful)
            Column(
                name="Bandwidth_Utilization_Out",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=55.0,
                        min_value=20.0,
                        max_value=90.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.4,
                        noise_level=0.15,
                        spike_probability=0.02,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=102,
                    ),
                ),
            ),
            Column(
                name="Packet_Loss_Percent",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=0.5,
                        min_value=0.0,
                        max_value=2.0,
                        seasonality_type="none",
                        noise_level=0.3,
                        spike_probability=0.05,
                        spike_magnitude=0.5,
                        interval_minutes=15,
                        seed=103,
                    ),
                ),
            ),
            Column(
                name="Latency_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=15.0,
                        min_value=8.0,
                        max_value=25.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.25,
                        noise_level=0.2,
                        spike_probability=0.03,
                        spike_magnitude=0.4,
                        interval_minutes=15,
                        seed=104,
                    ),
                ),
            ),
            Column(
                name="Jitter_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=2.0,
                        min_value=0.5,
                        max_value=5.0,
                        seasonality_type="none",
                        noise_level=0.25,
                        spike_probability=0.04,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=105,
                    ),
                ),
            ),
        ],
    )

    # ENTITY 2: core_node
    core_node = Entity(
        name="core_node",
        cardinality=n_core_nodes,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Core_Node_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[
                            "MME_001",
                            "MME_002",
                            "AMF_001",
                            "SMF_001",
                            "UPF_001",
                            "UPF_002",
                        ],
                        with_replacement=False,
                        seed=300,
                    ),
                ),
            ),
            # Measurement columns (stateful)
            Column(
                name="MM_AttachedUEs",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=5000.0,
                        min_value=3000.0,
                        max_value=7500.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.3,
                        noise_level=0.15,
                        spike_probability=0.01,
                        spike_magnitude=0.2,
                        interval_minutes=15,
                        seed=302,
                    ),
                ),
            ),
            Column(
                name="SM_ActivePDUSessions",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=3000.0,
                        min_value=1800.0,
                        max_value=4500.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.3,
                        noise_level=0.15,
                        spike_probability=0.01,
                        spike_magnitude=0.2,
                        interval_minutes=15,
                        seed=303,
                    ),
                ),
            ),
            Column(
                name="CPU_Load",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=60.0,
                        min_value=30.0,
                        max_value=90.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.03,
                        spike_magnitude=0.25,
                        interval_minutes=15,
                        seed=304,
                    ),
                ),
            ),
        ],
    )

    # ENTITY 3: cell_site
    cell_site = Entity(
        name="cell_site",
        cardinality=n_cell_sites,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Cell_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="CELL_{id}"),
                ),
            ),
            # Base_Station_ID references RAN base station (eNodeB for 4G / gNodeB for 5G)
            # This is independent since we don't model base_station as a separate entity
            Column(
                name="Base_Station_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["eNB_001", "eNB_002", "eNB_003", "gNB_001", "gNB_002"],
                        with_replacement=True,
                        seed=200,
                    ),
                ),
            ),
            Column(
                name="Location_Lat",
                data_type="float64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.UNIFORM_DIST,
                    params=UniformDistParams(lower=40.0, upper=41.0, seed=202),
                ),
            ),
            Column(
                name="Location_Lon",
                data_type="float64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.UNIFORM_DIST,
                    params=UniformDistParams(lower=-80.0, upper=-79.0, seed=203),
                ),
            ),
            # Foreign keys to transport_link
            Column(
                name="Transport_Device_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            Column(
                name="Transport_Interface_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            # Measurement columns (stateful)
            Column(
                name="RRC_ConnEstabFail",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=5.0,  # ~5% failure rate
                        min_value=0.0,
                        max_value=20.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.02,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=211,
                    ),
                ),
            ),
            Column(
                name="RRC_ConnEstabSucc",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=95.0,
                        min_value=55.0,
                        max_value=145.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.01,
                        spike_magnitude=0.25,
                        interval_minutes=15,
                        seed=207,
                    ),
                ),
            ),
            Column(
                name="RRC_ConnEstabAtt",
                data_type="int64",
                column_type=ColumnType.DERIVED,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                derivation=Derivation(
                    function_type=DerivationFunctionType.SUM,
                    dependent_columns=["RRC_ConnEstabSucc", "RRC_ConnEstabFail"],
                    params=SumParams(),
                ),
            ),
            Column(
                name="ERAB_EstabInitSuccNbr_QCI",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=90.0,
                        min_value=50.0,
                        max_value=140.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.01,
                        spike_magnitude=0.25,
                        interval_minutes=15,
                        seed=208,
                    ),
                ),
            ),
            Column(
                name="DL_PRB_Utilization",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=50.0,
                        min_value=20.0,
                        max_value=85.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.4,
                        noise_level=0.18,
                        spike_probability=0.02,
                        spike_magnitude=0.2,
                        interval_minutes=15,
                        seed=209,
                    ),
                ),
            ),
            Column(
                name="Cell_Availability",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=99.5,
                        min_value=97.0,
                        max_value=100.0,
                        seasonality_type="none",
                        noise_level=0.05,
                        spike_probability=0.005,
                        spike_magnitude=0.5,
                        interval_minutes=15,
                        seed=210,
                    ),
                ),
            ),
        ],
    )

    # ENTITY RELATIONSHIPS
    relationships = [
        # cell_site -> transport_link
        # Cells connect to transport network via specific device/interface pairs
        EntityRelationship(
            from_entity="cell_site",
            to_entity="transport_link",
            relationship_type=EntityRelationshipType.MANY_TO_ONE,
            join_columns={
                "Transport_Device_ID": "Device_ID",
                "Transport_Interface_ID": "Interface_ID",
            },
        ),
    ]

    # GLOBAL TIMESTAMP CONFIGURATION
    global_timestamp = GlobalTimestamp(
        t_start=global_start_time,
        t_end=global_end_time,
        time_interval=global_time_interval,
    )

    return DataSchema(
        entities=[transport_link, core_node, cell_site],
        entity_relationships=relationships,
        global_timestamp=global_timestamp,
    )


def create_incident_telecom_ran_schema(
    n_transport_links=1,
    n_core_nodes=6,
    n_cell_sites=20,
    global_start_time="2025-01-03T00:00:00Z",
    global_end_time="2025-01-03T06:00:00Z",
    global_time_interval="15min",
) -> DataSchema:
    """Create the Snowflake Demo telecom RAN network schema for an incident."""
    # ENTITY 1: transport_link
    transport_link = Entity(
        name="transport_link",
        cardinality=n_transport_links,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Device_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["RTR_001"],
                        with_replacement=True,
                        seed=100,
                    ),
                ),
            ),
            Column(
                name="Interface_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["eth0"],
                        with_replacement=True,
                        seed=101,
                    ),
                ),
            ),
            # Measurement columns (stateful)
            Column(
                name="Bandwidth_Utilization_Out",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=55.0,
                        min_value=20.0,
                        max_value=90.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.4,
                        noise_level=0.15,
                        spike_probability=0.02,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=102,
                    ),
                ),
            ),
            Column(
                name="Packet_Loss_Percent",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=0.5,
                        min_value=0.0,
                        max_value=2.0,
                        seasonality_type="none",
                        noise_level=0.3,
                        spike_probability=0.05,
                        spike_magnitude=0.5,
                        interval_minutes=15,
                        seed=103,
                    ),
                ),
            ),
            Column(
                name="Latency_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=15.0,
                        min_value=8.0,
                        max_value=25.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.25,
                        noise_level=0.2,
                        spike_probability=0.03,
                        spike_magnitude=0.4,
                        interval_minutes=15,
                        seed=104,
                    ),
                ),
            ),
            Column(
                name="Jitter_ms",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=2.0,
                        min_value=0.5,
                        max_value=5.0,
                        seasonality_type="none",
                        noise_level=0.25,
                        spike_probability=0.04,
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=105,
                    ),
                ),
            ),
        ],
    )

    # ENTITY 2: core_node
    core_node = Entity(
        name="core_node",
        cardinality=n_core_nodes,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Core_Node_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=[
                            "MME_001",
                            "MME_002",
                            "AMF_001",
                            "SMF_001",
                            "UPF_001",
                            "UPF_002",
                        ],
                        with_replacement=False,
                        seed=300,
                    ),
                ),
            ),
            # Measurement columns (stateful)
            Column(
                name="MM_AttachedUEs",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=4000.0,  # Slightly lower (was 5000.0) - some UEs drop
                        min_value=3000.0,  # Same floor
                        max_value=6000.0,  # Reduced ceiling (was 7500.0)
                        seasonality_type="none",
                        noise_level=0.25,  # More volatility (was 0.15)
                        spike_probability=0.03,  # More instability
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=302,
                    ),
                ),
            ),
            Column(
                name="SM_ActivePDUSessions",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=3000.0,
                        min_value=1800.0,
                        max_value=4500.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.3,
                        noise_level=0.15,
                        spike_probability=0.01,
                        spike_magnitude=0.2,
                        interval_minutes=15,
                        seed=303,
                    ),
                ),
            ),
            Column(
                name="CPU_Load",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=75.0,  # Higher load (was 60.0)
                        min_value=65.0,  # Higher floor (was 30.0)
                        max_value=95.0,  # Near capacity (was 90.0)
                        seasonality_type="none",
                        noise_level=0.15,
                        spike_probability=0.05,  # More CPU spikes
                        spike_magnitude=0.3,
                        interval_minutes=15,
                        seed=304,
                    ),
                ),
            ),
        ],
    )

    # ENTITY 3: cell_site
    cell_site = Entity(
        name="cell_site",
        cardinality=n_cell_sites,
        timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
        columns=[
            # Metadata columns
            Column(
                name="Cell_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.ID,
                    params=IDParams(template_str="CELL_{id}"),
                ),
            ),
            # Base_Station_ID references RAN base station (eNodeB for 4G / gNodeB for 5G)
            # This is independent since we don't model base_station as a separate entity
            Column(
                name="Base_Station_ID",
                data_type="string",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.CATEGORICAL,
                    params=CategoricalParams(
                        values=["eNB_001", "eNB_002", "eNB_003", "gNB_001", "gNB_002"],
                        with_replacement=True,
                        seed=200,
                    ),
                ),
            ),
            Column(
                name="Location_Lat",
                data_type="float64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.UNIFORM_DIST,
                    params=UniformDistParams(lower=40.0, upper=41.0, seed=202),
                ),
            ),
            Column(
                name="Location_Lon",
                data_type="float64",
                column_type=ColumnType.INDEPENDENT,
                column_category_type=ColumnCategoryType.METADATA,
                domain=Domain(
                    type=DomainType.UNIFORM_DIST,
                    params=UniformDistParams(lower=-80.0, upper=-79.0, seed=203),
                ),
            ),
            # Foreign keys to transport_link
            Column(
                name="Transport_Device_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            Column(
                name="Transport_Interface_ID",
                data_type="string",
                column_type=ColumnType.FOREIGN_KEY,
                column_category_type=ColumnCategoryType.METADATA,
            ),
            # Measurement columns (stateful)
            Column(
                name="RRC_ConnEstabFail",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=15.0,  # 3x higher failures (was 5.0)
                        min_value=10.0,  # Higher floor (was 0.0)
                        max_value=30.0,  # Higher ceiling (was 20.0)
                        seasonality_type="none",
                        noise_level=0.25,
                        spike_probability=0.05,  # More failure spikes
                        spike_magnitude=0.4,
                        interval_minutes=15,
                        seed=211,
                    ),
                ),
            ),
            Column(
                name="RRC_ConnEstabSucc",
                data_type="int64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=95.0,
                        min_value=55.0,
                        max_value=145.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.01,
                        spike_magnitude=0.25,
                        interval_minutes=15,
                        seed=207,
                    ),
                ),
            ),
            Column(
                name="RRC_ConnEstabAtt",
                data_type="int64",
                column_type=ColumnType.DERIVED,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                derivation=Derivation(
                    function_type=DerivationFunctionType.SUM,
                    dependent_columns=["RRC_ConnEstabSucc", "RRC_ConnEstabFail"],
                    params=SumParams(),
                ),
            ),
            Column(
                name="ERAB_EstabInitSuccNbr_QCI",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=90.0,
                        min_value=50.0,
                        max_value=140.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.35,
                        noise_level=0.2,
                        spike_probability=0.01,
                        spike_magnitude=0.25,
                        interval_minutes=15,
                        seed=208,
                    ),
                ),
            ),
            Column(
                name="DL_PRB_Utilization",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=50.0,
                        min_value=20.0,
                        max_value=85.0,
                        seasonality_type="peak_offpeak",
                        peak_start_hour=8,
                        peak_end_hour=22,
                        seasonality_strength=0.4,
                        noise_level=0.18,
                        spike_probability=0.02,
                        spike_magnitude=0.2,
                        interval_minutes=15,
                        seed=209,
                    ),
                ),
            ),
            Column(
                name="Cell_Availability",
                data_type="float64",
                column_type=ColumnType.STATEFUL,
                column_category_type=ColumnCategoryType.MEASUREMENT,
                domain=Domain(
                    type=DomainType.TIMESERIES,
                    params=TimeseriesParams(
                        base_value=97.0,  # Degraded (was 99.5)
                        min_value=94.0,  # Lower floor (was 97.0)
                        max_value=99.0,  # Lower ceiling (was 100.0)
                        seasonality_type="none",
                        noise_level=0.08,  # More volatility
                        spike_probability=0.02,  # More availability dips
                        spike_magnitude=0.7,  # Larger dips
                        interval_minutes=15,
                        seed=210,
                    ),
                ),
            ),
        ],
    )

    # ENTITY RELATIONSHIPS
    relationships = [
        # cell_site -> transport_link
        # Cells connect to transport network via specific device/interface pairs
        EntityRelationship(
            from_entity="cell_site",
            to_entity="transport_link",
            relationship_type=EntityRelationshipType.MANY_TO_ONE,
            join_columns={
                "Transport_Device_ID": "Device_ID",
                "Transport_Interface_ID": "Interface_ID",
            },
        ),
    ]

    # GLOBAL TIMESTAMP CONFIGURATION
    global_timestamp = GlobalTimestamp(
        t_start=global_start_time,
        t_end=global_end_time,
        time_interval=global_time_interval,
    )

    return DataSchema(
        entities=[transport_link, core_node, cell_site],
        entity_relationships=relationships,
        global_timestamp=global_timestamp,
    )


def plot_metric_over_time(
    df,
    metric_col,
    timestamp_col,
    figsize=(12, 6),
    title=None,
    highlight_start=None,
    highlight_end=None,
    highlight_color="red",
):
    """
    Plot a metric over time from a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    metric_col : str
        Name of the metric column to plot
    timestamp_col : str
        Name of the timestamp column
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title (defaults to metric name)
    highlight_start : str or datetime, optional
        Start of highlight range (must provide both start and end)
    highlight_end : str or datetime, optional
        End of highlight range (must provide both start and end)
    highlight_color : str
        Color for highlighted time range (default: 'red')

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object (use fig.savefig() to save)
    """
    # Validate columns
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in dataframe")
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe")

    # Validate highlight parameters
    if (highlight_start is None) != (highlight_end is None):
        raise ValueError(
            "Must provide both highlight_start and highlight_end, or neither"
        )

    # Work on a copy and sort by timestamp
    plot_df = df[[timestamp_col, metric_col]].copy().sort_values(timestamp_col)
    plot_df[timestamp_col] = pd.to_datetime(plot_df[timestamp_col])

    # Validate highlight range if provided
    if highlight_start is not None:
        highlight_start = pd.to_datetime(highlight_start)
        highlight_end = pd.to_datetime(highlight_end)

        data_start = plot_df[timestamp_col].min()
        data_end = plot_df[timestamp_col].max()

        if highlight_start < data_start or highlight_end > data_end:
            raise ValueError(
                f"Highlight range [{highlight_start}, {highlight_end}] must be within "
                f"data range [{data_start}, {data_end}]"
            )
        if highlight_start >= highlight_end:
            raise ValueError(
                f"highlight_start ({highlight_start}) must be before highlight_end ({highlight_end})"
            )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with or without highlighting
    if highlight_start is not None:
        # Split data into segments
        before = plot_df[plot_df[timestamp_col] < highlight_start]
        highlight = plot_df[
            (plot_df[timestamp_col] >= highlight_start)
            & (plot_df[timestamp_col] <= highlight_end)
        ]
        after = plot_df[plot_df[timestamp_col] > highlight_end]

        # Plot each segment
        if not before.empty:
            ax.plot(
                before[timestamp_col],
                before[metric_col],
                linewidth=2.5,
                marker="o",
                markersize=4,
            )

        if not highlight.empty:
            ax.plot(
                highlight[timestamp_col],
                highlight[metric_col],
                linewidth=2.5,
                marker="o",
                markersize=4,
                color=highlight_color,
            )

        if not after.empty:
            ax.plot(
                after[timestamp_col],
                after[metric_col],
                linewidth=2.5,
                marker="o",
                markersize=4,
            )
    else:
        # Single plot without highlighting
        ax.plot(
            plot_df[timestamp_col],
            plot_df[metric_col],
            linewidth=2.5,
            marker="o",
            markersize=4,
        )

    # Show x axis timestamps every 12 hours
    ax.xaxis.set_major_locator(HourLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))

    # Formatting
    ax.set_xlabel(timestamp_col, fontsize=14, fontweight="bold")
    ax.set_ylabel(metric_col, fontsize=14, fontweight="bold")
    ax.set_title(title or f"{metric_col} Over Time", fontsize=16, fontweight="bold")

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    return fig


def plot_metrics_over_time(df, metric_cols, timestamp_col, figsize=(12, 6), title=None):
    """
    Plot multiple metrics over time from a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    metric_cols : list of str
        List of metric column names to plot
    timestamp_col : str
        Name of the timestamp column
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Plot title (defaults to "Metrics Over Time")

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object (use fig.savefig() to save)
    """
    # Validate columns
    for metric_col in metric_cols:
        if metric_col not in df.columns:
            raise ValueError(f"Column '{metric_col}' not found in dataframe")
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe")

    # Work on a copy and sort by timestamp
    plot_df = df[[timestamp_col] + metric_cols].copy().sort_values(timestamp_col)
    plot_df[timestamp_col] = pd.to_datetime(plot_df[timestamp_col])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each metric
    for metric_col in metric_cols:
        ax.plot(
            plot_df[timestamp_col], plot_df[metric_col], linewidth=2, label=metric_col
        )

    # Show x axis timestamps every 12 hours
    ax.xaxis.set_major_locator(HourLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d %H:%M"))

    # Formatting
    ax.set_xlabel(timestamp_col, fontsize=14, fontweight="bold")
    ax.set_ylabel("Value", fontsize=14, fontweight="bold")
    ax.set_title(title or "Metrics Over Time", fontsize=16, fontweight="bold")

    # Add legend
    ax.legend(loc="best", fontsize=11)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle="--")

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    return fig
