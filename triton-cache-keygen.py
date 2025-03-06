from __future__ import annotations
import hashlib
import json
import os
import sys
import logging
import argparse
from typing import Dict, Set, Tuple, Any
from pathlib import Path

import triton
from triton import __version__

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate Triton cache key')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed components of the cache key')
    return parser.parse_args()

def get_cache_invalidating_env_vars() -> Dict[str, str]:
    """Get the Triton's official list of cache-invalidating environment variables."""
    try:
        from triton._C.libtriton import get_cache_invalidating_env_vars as _get_env_vars
        return _get_env_vars()
    except ImportError as e:
        logger.warning("Failed to import Triton's C library: %s", e)
        return {}

def get_current_target() -> str:
    """Get current target information from Triton's driver."""
    try:
        from triton.runtime.driver import driver
        target = driver.active.get_current_target()
        return f"{target.backend}-{target.arch}-{target.warp_size}"
    except Exception as e:
        logger.error("Failed to get current target: %s", e)
        return "unknown-backend-0-0"

def get_env_vars_for_cache() -> Dict[str]:
    """Collect all environment variables that affect the cache key."""
    core_vars = get_cache_invalidating_env_vars()
    
    return core_vars

#def triton_base_encoding(key: str) -> str:
#    # In early versions of Triton, the hash is directly used in the path name.
#    # Later, the hash is converted to base64 before being used in the path name.
#    # Later, the base64 convertion was replaced to the base32
#    #
#    # This code tries to import _base64 and falls back to _base32 if _base64 is unavailable.
#    #
#    # To handle this, try to import the to-base64-conversion function.
#    # If it exists, use it; otherwise, try using _base32; if both are unavailable, use the hash directly.
#    try:
#        from triton.runtime.cache import _base64
#        return _base64(key)
#    except Exception as e:
#        try:
#            from triton.runtime.cache import _base32
#
#            return _base32(key)
#        except Exception as e:
#            return key

def generate_triton_cache_key(source_hash: str | None = None) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a Triton cache key
    
    Returns:
        Tuple of (base{32|64}-encoded key, components dictionary)
    """
    components = {}
    
    try:
        # Get Triton installation fingerprint
        components['triton_key'] = triton.compiler.compiler.triton_key()
        logger.debug("Triton key components: %s", components['triton_key'])

        # Source content
        source_content = source_hash or "dummy_source_content"
        components['source'] = {
            'content': source_content,
            'hash': hashlib.sha256(source_content.encode()).hexdigest()
        }

        # Backend information
        backend_info = get_current_target()
        components['backend'] = {
            'info': backend_info,
            'hash': hashlib.sha256(backend_info.encode()).hexdigest()
        }

        # Compilation options
        options = {
            "num_warps": 4,
            "num_stages": 3,
            "debug": os.getenv("TRITON_DEBUG", "0") == "1"
        }
        components['options'] = {
            'values': options,
            'hash': hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
        }

        # Environment variables
        env_vars = get_env_vars_for_cache()
        components['environment'] = {
            'variables': env_vars,
            'hash': hashlib.sha256(json.dumps(sorted(env_vars.items()), sort_keys=True).encode()).hexdigest()
        }

        # Combine all components
        key_components = (
            f"{components['triton_key']}-"
            f"{components['source']['hash']}-"
            f"{components['backend']['hash']}-"
            f"{components['options']['hash']}-"
            f"{components['environment']['hash']}"
        )
        components['final_composite'] = key_components
        
        final_hash = hashlib.sha256(key_components.encode()).hexdigest()
        return final_hash, components

    except Exception as e:
        logger.error("Failed to generate cache key: %s", e)
        raise

def format_component(component: Any) -> str:
    """Format components for human-readable output"""
    if isinstance(component, dict):
        return json.dumps(component, indent=2, sort_keys=True)
    if isinstance(component, (list, tuple)):
        return '\n'.join(component)
    return str(component)

if __name__ == '__main__':
    args = parse_args()
    
    try:
        logger.info("Generating Triton cache key...")
        cache_key, components = generate_triton_cache_key()
        
        if args.verbose:
            print("\nCache Key Components Breakdown:")
            print(f"{'-'*40}")
            
            print("\n1. Triton Installation Fingerprint:")
            print(format_component(components['triton_key']))
            
            print("\n2. Source Code:")
            print(format_component(components['source']))
            
            print("\n3. Backend Configuration:")
            print(format_component(components['backend']))
            
            print("\n4. Compilation Options:")
            print(format_component(components['options']))
            
            print("\n5. Environment Variables:")
            print(format_component(components['environment']['variables']))
            
            print("\n6. Final Composite String:")
            print(format_component(components['final_composite']))
            
            print(f"\n{'='*40}")
            print("Final Cache Key:")
        
        print(f"\n{cache_key}")
        sys.exit(0)
    except Exception as e:
        logger.critical("Critical error during cache key generation: %s", e)
        sys.exit(1)
