#!/usr/bin/env python3
"""
Main entry point for the TV Command Synthesis Pipeline.

This script provides a command-line interface for running the complete
TV command synthesis pipeline or individual phases.
"""

import argparse
import sys
import json
from pathlib import Path

from src.pipeline.pipeline_executor import PipelineExecutor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TV Command Synthesis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --run-all

  # Run specific phases
  python main.py --phases phase1 phase2

  # Run with custom configuration
  python main.py --config configs/custom_config.yaml --run-all

  # Run Phase 4 with custom mode (multiple SNR ranges)
  python main.py --phases phase4 --custom-mode

  # Validate setup
  python main.py --validate

  # Show status
  python main.py --status
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file (default: configs/pipeline_config.yaml)'
    )

    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run the complete pipeline (all phases)'
    )

    parser.add_argument(
        '--phases',
        nargs='+',
        choices=['phase1', 'phase2', 'phase3', 'phase4'],
        help='Run specific phases'
    )

    parser.add_argument(
        '--custom-mode',
        action='store_true',
        help='Enable custom mode for Phase 4 (multiple SNR ranges)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate pipeline setup and data availability'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status and configuration'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Initialize pipeline
    try:
        pipeline = PipelineExecutor(args.config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    # Handle different commands
    if args.validate:
        print("Validating pipeline setup...")
        validation = pipeline.validate_setup()

        if validation['valid']:
            print("✅ Pipeline setup is valid")
        else:
            print("❌ Pipeline setup has issues:")
            for issue in validation['issues']:
                print(f"  - {issue}")

        if validation['warnings']:
            print("⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")

        return

    if args.status:
        print("Pipeline Status:")
        status = pipeline.get_pipeline_status()
        print(json.dumps(status, indent=2, default=str))
        return

    # Override output directory if specified
    if args.output_dir:
        pipeline.config.set('data.output_dir', args.output_dir)

    # Run pipeline
    if args.run_all:
        print("Running complete pipeline...")
        results = pipeline.run_full_pipeline(
            custom_mode_phase4=args.custom_mode
        )
    elif args.phases:
        print(f"Running phases: {args.phases}")
        results = pipeline.run_full_pipeline(
            phases=args.phases,
            custom_mode_phase4=args.custom_mode
        )
    else:
        parser.print_help()
        return

    # Print results
    print("\n" + "="*50)
    print("PIPELINE RESULTS")
    print("="*50)

    if results['success']:
        print("✅ Pipeline completed successfully!")
    else:
        print("❌ Pipeline failed!")
        for error in results['errors']:
            print(f"  Error: {error}")

    print(f"\nTotal execution time: {results['total_execution_time']:.2f} seconds")

    print("\nPhase Results:")
    for phase, result in results['phases'].items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['execution_time']:.2f}s"
        records = result.get('output_records', 'N/A')
        print(f"  {status} {phase}: {records} records in {time_str}")

        if not result['success']:
            print(f"    Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()