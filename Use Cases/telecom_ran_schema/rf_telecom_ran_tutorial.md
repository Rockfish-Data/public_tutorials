# Rockfish Entity Data Generator Tutorial: Telecom RAN Data

## Overview

This tutorial teaches you how to generate synthetic timeseries data using Rockfish's Entity Data Generator.

We'll walk through an example telecom RAN (Radio Access Network) schema to demonstrate core concepts.

**What you'll learn:**
- How to define entities with cardinalities and time ranges
- How to create entity relationships with foreign keys
- The timeseries data model (metadata vs measurement columns)
- How to configure columns using different domain types (Categorical, Uniform Distribution, Timeseries)
- How to control data types for your columns
- How to create derived columns with dependencies

## 1. Defining Entities and Schema Structure

Rockfish's Entity Data Generator creates synthetic data matching a `DataSchema`. 

In this tutorial, we'll generate data for a telecom RAN network `DataSchema` with three entities:
- **transport_link**: Network infrastructure devices
- **core_node**: Core network elements like MME, AMF, SMF, UPF
- **cell_site**: Radio access points serving end users

The exact schema definition can be found in the tutorial code [here](utils.py).

### 1.1 Creating Entities

An `Entity` represents a table in your synthetic dataset. Each entity has:
- `name`: Unique identifier (becomes the dataset name)
- `cardinality`: Number of unique entity instances
- `timestamp`: Timestamp column configuration for time-series data
- `columns`: List of column specifications (covered in Section 3)

Example from our telecom RAN schema:

```python
cell_site = Entity(
    name="cell_site",
    cardinality=100,
    timestamp=Timestamp(column_name="Timestamp", data_type="timestamp"),
    columns=[...]  # Cell_ID, Base_Station_ID, RRC metrics, etc.
)
```

This creates a `cell_site` entity with 100 unique cell instances.

### 1.2 Configuring Data Size

**Cardinality** determines how many unique instances of an entity exist. In our telecom RAN schema:
- **transport_link**: 6 network infrastructure devices
- **core_node**: 16 core network elements (MME, AMF, SMF, UPF)
- **cell_site**: 100 radio access points

**GlobalTimestamp** defines the time range and interval for all time-series data:

```python
global_timestamp = GlobalTimestamp(
    t_start="2025-01-01T00:00:00Z",
    t_end="2025-01-03T00:00:00Z",
    time_interval="15min",
)
```

**Parameters:**
- `t_start`: Start timestamp (ISO 8601 format)
- `t_end`: End timestamp (ISO 8601 format)
- `time_interval`: Time between measurements (e.g., `"1min"`, `"15min"`, `"1hour"`, `"1day"`)

**Total rows** = cardinality × number of timestamps. In our example, we generate data for 2 days at 15-minute intervals (193 timestamps, start and end inclusive):
- transport_link: 6 × 193 = 1,158 rows
- core_node: 16 × 193 = 3,088 rows
- cell_site: 100 × 193 = 19,300 rows

### 1.3 Entity Relationships

**Entity relationships** define how entities reference each other through foreign keys. 
Our schema has one relationship: many `cell_sites` connect to `transport_links`.

#### Step 1: Declare Foreign Key Columns

In the `cell_site` entity, declare foreign key columns that reference `transport_link` (see [Section 2.2](#22-column-types) for more on column types):

```python
# In cell_site entity columns:
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
```

**Note:** Foreign key columns do NOT have a `domain` or `derivation`. Their values are populated automatically by Rockfish.

#### Step 2: Define the Relationship

```python
relationships = [
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
```

**Parameters:**
- `from_entity`: Entity with foreign keys (`cell_site`)
- `to_entity`: Entity being referenced (`transport_link`)
- `relationship_type`: `MANY_TO_ONE`, `ONE_TO_ONE`, or `ONE_TO_MANY`
- `join_columns`: Maps foreign key columns to target columns (format: `{foreign_key: target_column}`)

In our example, each `cell_site` references a valid (`Device_ID`, `Interface_ID`) pair from `transport_link`.
Rockfish ensures referential integrity automatically.

## 2. The Timeseries Data Model

Rockfish's [timeseries data model](https://docs.rockfish.ai/data-models.html#time-series-data) organizes columns into two categories (metadata, measurement) that reflect 
how timeseries data is typically structured. 
We'll focus on **`cell_site`** as our primary example for the subsequent sections.

### 2.1 Column Categories: Metadata vs Measurement

**METADATA** (`ColumnCategoryType.METADATA`):
- Describes entity identity and attributes ("who", "what", "where")
- Examples: `Cell_ID`, `Base_Station_ID`, `Location_Lat`, `Transport_Device_ID`

**MEASUREMENT** (`ColumnCategoryType.MEASUREMENT`):
- Time-varying performance metrics ("how much", "how many" at each timestamp)
- Examples: `RRC_ConnEstabAtt`, `DL_PRB_Utilization`, `Cell_Availability`

### 2.2 Column Types

Each column has a `column_type` that determines how values are generated:

**INDEPENDENT** (`ColumnType.INDEPENDENT`):
- Generated once per entity instance; values don't change over time
- Typically, METADATA columns
- Example: `Base_Station_ID` sampled once per cell

**STATEFUL** (`ColumnType.STATEFUL`):
- Values evolve over time at each timestamp
- Typically, MEASUREMENT columns
- Example: `Cell_Availability` changes every 15 minutes

**DERIVED** (`ColumnType.DERIVED`):
- Computed from other columns using derivation functions
- Example: `RRC_ConnEstabAtt` = `RRC_ConnEstabSucc` + `RRC_ConnEstabFail`

**FOREIGN_KEY** (`ColumnType.FOREIGN_KEY`):
- References another entity; values populated automatically from relationships
- Always METADATA
- Example: `Transport_Device_ID` references `Device_ID` in transport_link

## 3. Configuring Individual Columns

### 3.1 Column Structure and Data Types

Every column has a standard structure:

```python
Column(
    name="column_name",
    data_type="string",  # string, int64, float64, timestamp
    column_type=ColumnType.INDEPENDENT,  # How values are generated
    column_category_type=ColumnCategoryType.METADATA,  # Metadata or Measurement
    domain=Domain(...)  # Domain specification (for INDEPENDENT/STATEFUL)
)
```

**Supported data types:**
- `"string"`: Text values (IDs, device names, status codes)
- `"int64"`: Integer values (counts, discrete measurements)
- `"float64"`: Floating-point numbers (percentages, ratios, continuous measurements)
- `"timestamp"`: Date/time values (used for entity timestamp columns)

The `domain` parameter specifies how to generate column values. 
In the following sections, we'll explore three main domain types used in `cell_site`.

### 3.2 Categorical Columns (CategoricalParams)

**Use case:** Sample from a fixed set of discrete values.

**Example from `cell_site`:**

```python
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
```

**Parameters:**
- `values`: List of possible values. You can use `Faker` or AI assistance to generate values! 
- `with_replacement`:
  - `True`: Values can be reused (multiple cells can share a base station)
  - `False`: Each value used at most once
- `seed`: Random seed for reproducibility

### 3.3 Uniform Distribution Columns (UniformDistParams)

**Use case:** Continuous numeric values uniformly distributed within a range.

**Example from `cell_site`:**

```python
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
```

**Parameters:**
- `lower`: Minimum value (inclusive)
- `upper`: Maximum value (inclusive)
- `seed`: Random seed for reproducibility

### 3.4 Timeseries Columns (TimeseriesParams)

**Use case:** Time-varying measurements with realistic temporal patterns (seasonality, noise, anomalies).

Timeseries columns use `ColumnType.STATEFUL`: they generate a sequence of values over time for each entity instance.

#### Example 1: RRC_ConnEstabFail (Connection Failures)

```python
Column(
    name="RRC_ConnEstabFail",
    data_type="int64",
    column_type=ColumnType.STATEFUL,
    column_category_type=ColumnCategoryType.MEASUREMENT,
    domain=Domain(
        type=DomainType.TIMESERIES,
        params=TimeseriesParams(
            base_value=5.0,  # Average failure rate (~5 per interval)
            min_value=0.0,  # Floor (no negative failures)
            max_value=20.0,  # Ceiling (max failures per interval)
            seasonality_type="peak_offpeak",  # Higher during peak hours
            peak_start_hour=8,  # Peak starts at 8 AM
            peak_end_hour=22,  # Peak ends at 10 PM
            seasonality_strength=0.35, # 35% variation due to time of day
            noise_level=0.2, # 20% random noise
            spike_probability=0.02, # 2% chance of spike per interval
            spike_magnitude=0.3, # Spikes are 30% above normal
            interval_minutes=15, # Matches global time_interval
            seed=211
        ),
    ),
),
```

**Parameters:**

**Core Values:**
- `base_value`: Baseline/average value
- `min_value`: Hard lower bound
- `max_value`: Hard upper bound

**Seasonality:**
- `seasonality_type`: `"peak_offpeak"` (step function), `"symmetric"` (sinusoidal), or `"none"`
- `peak_start_hour`, `peak_end_hour`: Peak period hours (0-23, for "peak_offpeak" only)
- `seasonality_strength`: Seasonal variation strength (0.0-1.0)

**Variation:**
- `noise_level`: Random noise as fraction of base_value (0.0-1.0)
- `spike_probability`: Chance of anomaly at each timestamp (0.0-1.0)
- `spike_magnitude`: Spike size as fraction of base_value (0.0-1.0+)

**Other Values:**
- `interval_minutes`: Must match `global_timestamp.time_interval`
- `seed`: Random seed for reproducibility

#### Example 2: Cell_Availability (High Availability)

```python
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
          seasonality_type="none",     # No daily pattern
          noise_level=0.05,            # Lower noise
          spike_probability=0.005,     # Rare anomalies
          spike_magnitude=0.5,
          interval_minutes=15,
          seed=210
        ),
    ),
),
```

## 4. Column Dependencies and Derived Columns

**Derived columns** compute their values from other columns using derivation functions. They:
- Use `ColumnType.DERIVED`
- Do NOT specify a `domain`
- Instead, specify a `derivation` with function type and dependent columns
- Are computed after all independent and stateful columns

### Example: RRC_ConnEstabAtt (Total Connection Attempts)

Total connection attempts = successful + failed connections. We derive this to ensure consistency:

```python
# Base columns (stateful)
Column(name="RRC_ConnEstabSucc", data_type="int64",
       column_type=ColumnType.STATEFUL, ...),
Column(name="RRC_ConnEstabFail", data_type="int64",
       column_type=ColumnType.STATEFUL, ...),

# Derived column
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
```

**Parameters:**
- `function_type`: Derivation function (e.g., `SUM`, `MULTIPLY`, `MAP_VALUES`)
- `dependent_columns`: Input column names (must exist in same entity)
- `params`: Function-specific parameters

**Result:** `RRC_ConnEstabAtt = RRC_ConnEstabSucc + RRC_ConnEstabFail` at each timestamp, ensuring consistency.

**Other derivation functions:**
- **MULTIPLY**: Element-wise multiplication (e.g., `total_cost = unit_price × quantity`)
- **SAMPLE_FROM_COLUMN**: Sample values based on conditions (e.g., error codes based on failure status)
- **MAP_VALUES**: Transform values using rules (e.g., "high"/"medium"/"low" from numeric thresholds)

See the [full API documentation](https://docs.rockfish.ai/sdk/actions-ent.html#rockfish.actions.ent.Derivation) for details on all derivation types.

## 5. Assembling the Complete Schema

Assemble entities, relationships, and global timestamp into a `DataSchema`:

```python
from rockfish.actions.ent import DataSchema

telecom_ran_schema = DataSchema(
    entities=[transport_link, core_node, cell_site],
    entity_relationships=relationships,
    global_timestamp=global_timestamp,
)
```

**Generating data:**

```python
import rockfish as rf
import rockfish.actions as ra

# Configure and run generation
config = ra.GenerateFromDataSchema.Config(schema=telecom_ran_schema)
generate = ra.GenerateFromDataSchema(config)

builder = rf.WorkflowBuilder()
builder.add(generate)
workflow = await builder.start(conn)
await workflow.wait(raise_on_failure=True)

# Access generated datasets
datasets = await workflow.datasets().collect()
for remote_ds in datasets:
    ds = await remote_ds.to_local(conn)
    df = ds.to_pandas()
    print(f"{ds.name()}: {len(df)} rows")
```

**Generated output** for our telecom RAN schema:
- transport_link: 1,158 rows
- core_node: 3,088 rows
- cell_site: 19,300 rows

## FAQ

### Entities & Relationships

**Q: How do I decide on cardinalities?**

Base them on your use case: match real-world scale, use smaller values (5-20) for quick iteration, and larger values (100-1000+) for the final data. 
Consider relationship ratios (e.g., 100 cells to 6 transport links = ~17 cells per link).

**Q: How do composite foreign keys work?**

Multiple columns together form the reference. In our schema, `(Transport_Device_ID, Transport_Interface_ID)` in `cell_site` references `(Device_ID, Interface_ID)` in `transport_link`.

### Columns & Data Types

**Q: When should I use INDEPENDENT vs STATEFUL?**

- **INDEPENDENT**: Values don't change over time (IDs, locations, device types)
- **STATEFUL**: Values evolve over time (bandwidth, CPU, counters)

**Q: What's the difference between METADATA and MEASUREMENT?**

- **METADATA**: Describes the entity (who/what/where)
- **MEASUREMENT**: Performance metrics that change over time

Refer to the [data models documentation page](https://docs.rockfish.ai/data-models.html) for an explanation of the data models that Rockfish supports. 

**Q: How do I choose the column `data_type`?**

- `string`: IDs, names, categorical text
- `int64`: Whole number counts, discrete values
- `float64`: Continuous measurements with decimals (percentages, utilization)
- `timestamp`: Date/time values (entity timestamps)

### Timeseries Configuration

**Q: How do I configure realistic timeseries patterns?**

Use `TimeseriesParams` to control seasonality, noise, and anomalies. See [Section 3.4](#34-timeseries-columns-timeseriesparams) for detailed parameter descriptions and examples.

### Derived Columns

**Q: Can derived columns depend on other derived columns?**

Yes. Rockfish automatically computes them in the correct dependency order.

**Q: What's the order of column generation?**

Rockfish automatically generates columns in the following order:
1. INDEPENDENT columns (IDs, locations)
2. FOREIGN_KEY columns (from relationships)
3. STATEFUL columns (timeseries measurements)
4. DERIVED columns (based on dependency order)

## Next Steps

**Experiment with the schema:**
- Modify cardinalities, timeseries parameters, or add new columns/entities
- Run the tutorial notebooks for baseline, scaled, and incident generation examples

**Learn more:**
- [Full API Documentation](https://docs.rockfish.ai/sdk/actions-ent.html)
