#!/usr/bin/env python3
"""
Runner script that loads YAML configuration files and runs main.py with those parameters.
Usage: python runner.py --f scripts/celeba/sum/script.yaml
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def yaml_to_args(config):
    """Convert YAML config dictionary to command-line arguments list."""
    args = []
    
    # Aliases: map YAML keys to their canonical argument names
    # Only needed for keys that don't follow the standard snake_case -> --snake_case pattern
    aliases = {
        'agg': 'aggregator',
        'wd': 'weight_decay',
    }
    
    def get_arg_name(key):
        """Convert YAML key to command-line argument name."""
        # First check if it's an alias
        canonical_key = aliases.get(key, key)
        # Convert to --argument_name format
        return f'--{canonical_key}'
    
    for key, value in config.items():
        arg_name = get_arg_name(key)
        
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                args.append(arg_name)
        # Handle list arguments
        elif isinstance(value, list):
            args.append(arg_name)
            args.extend([str(v) for v in value])
        # Handle None values (skip them)
        elif value is None:
            continue
        # Handle regular arguments
        else:
            args.append(arg_name)
            args.append(str(value))
    
    return args


def run_single_config(config_file):
    """Run a single YAML configuration file."""
    # Load YAML config
    try:
        config = load_yaml_config(config_file)
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {e}", file=sys.stderr)
        return False
    
    # Convert to command-line arguments
    cmd_args = yaml_to_args(config)
    
    # Build command
    cmd = [sys.executable, 'main.py'] + cmd_args
    
    # Print command for transparency
    print(f"\n{'=' * 80}", flush=True)
    print(f"Running: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 80}\n", flush=True)
    
    # Run main.py
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py with {config_file}: {e}", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted by user while running {config_file}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run main.py with YAML configuration file(s)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single YAML file
  python runner.py --f scripts/cifar100/aligned_mtl/mse.yaml
  
  # Run multiple YAML files
  python runner.py --f file1.yaml --f file2.yaml --f file3.yaml
  
  # Run from a file list (one path per line)
  python runner.py --file-list configs.txt
        """
    )
    parser.add_argument(
        '--f',
        dest='config_files',
        type=str,
        action='append',
        help='Path to YAML configuration file (can be specified multiple times)'
    )
    parser.add_argument(
        '--file-list',
        dest='file_list',
        type=str,
        help='Path to a text file containing YAML file paths (one per line)'
    )
    
    args = parser.parse_args()
    
    # Collect all config files to run
    config_files = []
    
    # Add files from --f arguments
    if args.config_files:
        config_files.extend(args.config_files)
    
    # Add files from --file-list
    if args.file_list:
        file_list_path = Path(args.file_list)
        if not file_list_path.exists():
            print(f"Error: File list not found: {file_list_path}", file=sys.stderr)
            sys.exit(1)
        
        with open(file_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    config_files.append(line)
    
    # Validate that at least one config file is provided
    if not config_files:
        parser.error("At least one configuration file must be provided via --f or --file-list")
    
    # Run each configuration file
    print(f"Found {len(config_files)} configuration file(s) to run:\n")
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file}")
    print()
    
    failed_files = []
    for i, config_file in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] Processing: {config_file}")
        success = run_single_config(config_file)
        if not success:
            failed_files.append(config_file)
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"Summary: {len(config_files) - len(failed_files)}/{len(config_files)} configuration(s) completed successfully")
    if failed_files:
        print(f"Failed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
        sys.exit(1)
    else:
        print("All configurations completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()

