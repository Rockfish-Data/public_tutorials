#!/usr/bin/env python3
"""
Blended PCAP Dataset Generator

This script generates a week-long PCAP dataset with time-varying volumes
and proportions of normal/anomalous/suspicious traffic using Rockfish APIs.

Usage:
    python 02_generate_blended_pcap.py [--config CONFIG_PATH] [--dry-run]

Example:
    python 02_generate_blended_pcap.py --config configs/blended_generation_config.yaml
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import rockfish as rf
import rockfish.actions as ra
from dotenv import load_dotenv

from templates import TEMPLATE_REGISTRY, get_template
from utils.volume_calculator import VolumeCalculator, load_config


class BlendedPCAPGenerator:
    """Generates week-long PCAP datasets with configurable traffic mix."""

    def __init__(self, config_path: str, dry_run: bool = False):
        """
        Initialize the generator.

        Args:
            config_path: Path to the configuration YAML file
            dry_run: If True, only calculate volumes without generating data
        """
        self.config = load_config(config_path)
        self.dry_run = dry_run
        self.volume_calculator = VolumeCalculator(self.config)

        # Parse time range
        time_range = self.config.get("time_range", {})
        self.start_time = datetime.fromisoformat(
            time_range.get("start", "2025-03-01T00:00:00Z").replace("Z", "+00:00")
        )
        self.end_time = datetime.fromisoformat(
            time_range.get("end", "2025-03-07T23:59:59Z").replace("Z", "+00:00")
        )
        self.granularity = time_range.get("granularity", "1hour")

        # Output settings
        output_config = self.config.get("output", {})
        self.output_dir = Path(output_config.get("directory", "generated_data/week_simulation"))
        self.output_format = output_config.get("format", "csv")
        self.filename_prefix = output_config.get("filename_prefix", "pcap_week")

        # Generation settings
        gen_config = self.config.get("generation", {})
        self.seed = gen_config.get("seed", 42)
        self.batch_size = gen_config.get("batch_size", 100)

        # Labeling settings
        label_config = self.config.get("labeling", {})
        self.include_labels = label_config.get("include_labels", True)
        self.label_column = label_config.get("label_column", "traffic_category")
        self.template_column = label_config.get("template_column", "template_type")
        self.event_column = label_config.get("event_column", "scheduled_event")

        # Post-processing settings
        post_config = gen_config.get("post_processing", {})
        self.derive_direction = post_config.get("derive_direction", True)
        self.derive_tcp_flags = post_config.get("derive_tcp_flags", True)
        self.construct_ip_addresses = post_config.get("construct_ip_addresses", True)
        self.sort_by_timestamp = post_config.get("sort_by_timestamp", True)

        # Initialize connection (lazy)
        self._conn = None

    @property
    def conn(self):
        """Lazy initialization of Rockfish connection."""
        if self._conn is None:
            load_dotenv()
            self._conn = rf.Connection.from_env()
        return self._conn

    def print_summary(self):
        """Print a summary of what will be generated."""
        summary = self.volume_calculator.generate_summary(self.start_time, self.end_time)

        print("\n" + "=" * 60)
        print("BLENDED PCAP GENERATION SUMMARY")
        print("=" * 60)
        print(f"\nTime Range:")
        print(f"  Start: {summary['time_range']['start']}")
        print(f"  End:   {summary['time_range']['end']}")
        print(f"  Granularity: {self.granularity}")

        print(f"\nExpected Traffic Volume:")
        print(f"  Total Sessions: {summary['total_sessions']:,}")
        print(f"  Estimated Packets: ~{summary['total_sessions'] * 12:,} (avg 12 packets/session)")

        print(f"\nCategory Distribution:")
        for category, count in summary['category_totals'].items():
            pct = 100 * count / summary['total_sessions'] if summary['total_sessions'] > 0 else 0
            print(f"  {category}: {count:,} sessions ({pct:.1f}%)")

        print(f"\nHourly Volume Statistics:")
        print(f"  Min: {summary['hourly_stats']['min']:,} sessions/hour")
        print(f"  Max: {summary['hourly_stats']['max']:,} sessions/hour")
        print(f"  Avg: {summary['hourly_stats']['avg']:,.1f} sessions/hour")

        if summary['scheduled_events']:
            print(f"\nScheduled Events:")
            for event in summary['scheduled_events']:
                print(f"  - {event['name']}: {event['start']} to {event['end']}")

        print(f"\nOutput:")
        print(f"  Directory: {self.output_dir}")
        print(f"  Format: {self.output_format}")
        print("=" * 60 + "\n")

    async def generate_from_template(
        self,
        category: str,
        template_name: str,
        n_sessions: int,
        time_range: tuple[str, str],
    ) -> Optional[pd.DataFrame]:
        """
        Generate packets using a specific template.

        Args:
            category: Traffic category (normal, anomalous, suspicious)
            template_name: Name of the template to use
            n_sessions: Number of sessions to generate
            time_range: Tuple of (start_time, end_time) as ISO strings

        Returns:
            DataFrame with generated packets, or None if dry run
        """
        if self.dry_run:
            return None

        try:
            # Get template function
            template_fn = get_template(category, template_name)

            # Create schema with specified parameters
            schema = template_fn(
                n_sessions=n_sessions,
                time_range=time_range,
            )

            # Generate data
            config = ra.GenerateFromDataSchema.Config(
                schema=schema,
                upload_datasets=False,  # Don't upload intermediate results
            )
            generate = ra.GenerateFromDataSchema(config)

            builder = rf.WorkflowBuilder()
            builder.add(generate)
            workflow = await builder.start(self.conn)

            # Wait for completion
            async for log in workflow.logs():
                pass  # Just wait for completion

            # Retrieve session data
            datasets = await workflow.datasets().collect()
            for remote_ds in datasets:
                ds = await remote_ds.to_local(self.conn)
                if ds.name() == "tcp_session":
                    return ds.to_pandas()

            return None

        except Exception as e:
            print(f"  Warning: Failed to generate {category}/{template_name}: {e}")
            return None

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to generated data.

        Args:
            df: Raw generated DataFrame

        Returns:
            Processed DataFrame
        """
        if df is None or len(df) == 0:
            return df

        # Derive direction from packet_type
        if self.derive_direction and "packet_type" in df.columns:
            direction_map = {
                "SYN": "C2S",
                "SYN_ACK": "S2C",
                "ACK": "C2S",
                "PSH_ACK_C2S": "C2S",
                "PSH_ACK_S2C": "S2C",
                "FIN": "C2S",
                "FIN_ACK": "S2C",
                "RST": "S2C",
            }
            df["direction"] = df["packet_type"].map(direction_map).fillna("C2S")

        # Derive TCP flags from packet_type
        if self.derive_tcp_flags and "packet_type" in df.columns:
            tcp_flags_map = {
                "SYN": "SYN",
                "SYN_ACK": "SYN,ACK",
                "ACK": "ACK",
                "PSH_ACK_C2S": "PSH,ACK",
                "PSH_ACK_S2C": "PSH,ACK",
                "FIN": "FIN",
                "FIN_ACK": "FIN,ACK",
                "RST": "RST",
            }
            df["tcp_flags"] = df["packet_type"].map(tcp_flags_map).fillna("ACK")

        # Derive packet size category
        if "packet_type" in df.columns:
            packet_size_category_map = {
                "SYN": "control",
                "SYN_ACK": "control",
                "ACK": "control",
                "FIN": "control",
                "FIN_ACK": "control",
                "RST": "control",
                "PSH_ACK_C2S": "data",
                "PSH_ACK_S2C": "data",
            }
            df["packet_size_category"] = (
                df["packet_type"].map(packet_size_category_map).fillna("control")
            )

        # Calculate actual packet sizes
        if "packet_size_base" in df.columns and "packet_size_category" in df.columns:
            def calc_size(row):
                if row["packet_size_category"] == "control":
                    return np.random.randint(40, 60)
                else:
                    return min(max(int(row["packet_size_base"]), 64), 1500)

            df["packet_size"] = df.apply(calc_size, axis=1)

        return df

    async def generate(self) -> pd.DataFrame:
        """
        Main generation loop.

        Returns:
            Complete DataFrame with all generated packets
        """
        self.print_summary()

        if self.dry_run:
            print("DRY RUN: No data will be generated.")
            return pd.DataFrame()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        np.random.seed(self.seed)

        all_packets = []
        total_chunks = 0
        processed_chunks = 0

        # Count total chunks first
        for _ in self.volume_calculator.iterate_time_chunks(
            self.start_time, self.end_time, self.granularity
        ):
            total_chunks += 1

        print(f"Generating data for {total_chunks} time chunks...")

        # Generate data for each time chunk
        for chunk_start, chunk_end, volume, proportions, template_weights in (
            self.volume_calculator.iterate_time_chunks(
                self.start_time, self.end_time, self.granularity
            )
        ):
            processed_chunks += 1
            print(f"\n[{processed_chunks}/{total_chunks}] Processing {chunk_start.isoformat()}")
            print(f"  Volume: {volume} sessions")

            # Check for active event
            event = self.volume_calculator.get_active_event(chunk_start)
            event_name = event["name"] if event else None
            if event_name:
                print(f"  Active event: {event_name}")

            # Time range for this chunk
            time_range = (chunk_start.isoformat(), chunk_end.isoformat())

            # Allocate sessions across categories and templates
            allocation = self.volume_calculator.allocate_sessions(
                volume, proportions, template_weights
            )

            # Generate sessions for each category and template
            for category, templates in allocation.items():
                for template_name, n_sessions in templates.items():
                    if n_sessions == 0:
                        continue

                    print(f"  Generating {n_sessions} {category}/{template_name} sessions...")

                    # Generate in batches if needed
                    remaining = n_sessions
                    while remaining > 0:
                        batch_size = min(remaining, self.batch_size)
                        packets_df = await self.generate_from_template(
                            category, template_name, batch_size, time_range
                        )

                        if packets_df is not None and len(packets_df) > 0:
                            # Add labels
                            if self.include_labels:
                                packets_df[self.label_column] = category
                                packets_df[self.template_column] = template_name
                                packets_df[self.event_column] = event_name

                            # Post-process
                            packets_df = self.post_process(packets_df)
                            all_packets.append(packets_df)

                        remaining -= batch_size

        if not all_packets:
            print("\nNo packets generated.")
            return pd.DataFrame()

        # Combine all packets
        print("\nCombining all packets...")
        result_df = pd.concat(all_packets, ignore_index=True)

        # Sort by timestamp if requested
        if self.sort_by_timestamp and "packet_timestamp" in result_df.columns:
            result_df = result_df.sort_values("packet_timestamp").reset_index(drop=True)

        print(f"\nTotal packets generated: {len(result_df):,}")
        print(f"Total sessions: {result_df['session_id'].nunique():,}")

        # Save to file
        output_path = self.output_dir / f"{self.filename_prefix}.{self.output_format}"
        if self.output_format == "parquet":
            result_df.to_parquet(output_path, index=False)
        else:
            result_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

        # Print category summary
        print("\nTraffic Category Summary:")
        print(result_df[self.label_column].value_counts())

        return result_df


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate blended PCAP dataset with time-varying traffic"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/blended_generation_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate volumes without generating data",
    )
    args = parser.parse_args()

    # Find config relative to script or use absolute path
    config_path = args.config
    if not os.path.isabs(config_path):
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / config_path

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    generator = BlendedPCAPGenerator(str(config_path), dry_run=args.dry_run)
    await generator.generate()


if __name__ == "__main__":
    asyncio.run(main())
