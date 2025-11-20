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


def main():
    parser = argparse.ArgumentParser(
        description='Run main.py with YAML configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--f',
        dest='config_file',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    # Load YAML config
    try:
        config = load_yaml_config(args.config_file)
    except Exception as e:
        print(f"Error loading configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert to command-line arguments
    cmd_args = yaml_to_args(config)
    
    # Build command
    cmd = [sys.executable, 'main.py'] + cmd_args
    
    # Print command for transparency
    print(f"Running: {' '.join(cmd)}", flush=True)
    print("-" * 80, flush=True)
    
    # Run main.py
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == '__main__':
    main()

