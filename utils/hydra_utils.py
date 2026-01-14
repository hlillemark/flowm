from typing import Optional, List, Tuple
import os
import sys
from omegaconf import OmegaConf
from pathlib import Path
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, bold_red
from omegaconf import DictConfig, OmegaConf

def validate_config_against_template(cfg: DictConfig, template_path: str = "configurations/full_template.yaml") -> bool:
    """
    Validate that the resolved config contains all keys from the template.
    
    Args:
        cfg: The resolved Hydra configuration
        template_path: Path to the template YAML file
        
    Returns:
        bool: True if validation passes, False otherwise
        
    Raises:
        ValueError: If validation fails with details about missing keys
    """

    # NOTE:  Need to fill in the missing keys here
    missing_keys_tolerance = [
        "algorithm.tasks.prediction.block_ssm",
        "algorithm.backbone.attn_chunk_size",
        "algorithm.backbone.b_h_list",
        "algorithm.backbone.b_w_list",
        "algorithm.backbone.d_state",
        "algorithm.backbone.expand",
        "algorithm.backbone.headdim",
        "algorithm.backbone.share_child",
        "algorithm.vae.pretrained_kwargs.depth_latent_mean",
        "algorithm.vae.pretrained_kwargs.depth_latent_std",
        "algorithm.varlen_context.prob",
        "algorithm.model_weights.rgb_pretrained.",
        "tags"
    ]
    
    extra_keys_tolerance = [
        "algorithm.model_weights.rgb_pretrained.",
        "tags",
        "algorithm.model_weights",
        "algorithm.vae.pretrained_dir",
        "experiment.throughput"
    ]

    def get_all_keys(config_dict, prefix=""):
        """Recursively extract all keys from a nested dictionary."""
        keys = set()
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                current_key = f"{prefix}.{key}" if prefix else key
                keys.add(current_key)
                if isinstance(value, dict):
                    keys.update(get_all_keys(value, current_key))
        return keys
    
    try:
        # Load template configuration
        template_cfg = OmegaConf.load(template_path)
        
        # Convert both configs to containers for comparison
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        template_config = OmegaConf.to_container(template_cfg, resolve=True)
        
        # Get all keys from both configs
        resolved_keys = get_all_keys(resolved_config)
        template_keys = get_all_keys(template_config)
        
        # Find missing keys
        missing_keys = template_keys - resolved_keys
        extra_keys = resolved_keys - template_keys

        toleranted_missing_keys = []
        tolerated_extra_keys = []
        for tolerance_key in missing_keys_tolerance:
            for key in missing_keys:
                if key.startswith(tolerance_key):
                    toleranted_missing_keys.append(key)
        for tolerance_key in extra_keys_tolerance:
            for key in extra_keys:
                if key.startswith(tolerance_key):
                    tolerated_extra_keys.append(key)
                    
        missing_keys = missing_keys - set(toleranted_missing_keys)
        extra_keys = extra_keys - set(tolerated_extra_keys)
        
        if missing_keys or extra_keys:
            error_msg = "Configuration validation failed:\n"
            if missing_keys:
                error_msg += f"  Missing keys (present in template but not in resolved config):\n"
                for key in sorted(missing_keys):
                    error_msg += f"    - {key}\n"
            if extra_keys:
                error_msg += f"  Extra keys (present in resolved config but not in template):\n"
                for key in sorted(extra_keys):
                    error_msg += f"    - {key}\n"
            
            rank_zero_print(bold_red("❌ Config validation failed!"))
            rank_zero_print(error_msg)
            return False
        
        rank_zero_print(cyan("✅ Configuration validation passed - all template keys are present"))
        return True
        
    except Exception as e:
        rank_zero_print(bold_red(f"❌ Error during config validation: {str(e)}"))
        return False



def get_available_shortcodes(config_path: str) -> List[str]:
    """
    Get a list of all available shortcodes in the system.
    
    Args:
        config_path: Path to the configurations directory
        
    Returns:
        List of available shortcode names
    """
    shortcode_dir = Path(config_path) / "shortcode"
    available_shortcodes = []
    
    if not shortcode_dir.exists():
        return available_shortcodes
    
    # Find all yaml files in shortcode directory
    for yaml_file in shortcode_dir.rglob("*.yaml"):
        # Get relative path from shortcode directory
        relative_path = yaml_file.relative_to(shortcode_dir)
        
        # Convert path to shortcode format
        if relative_path.name == "base.yaml":
            # For base.yaml files, use the parent directory path
            shortcode = str(relative_path.parent)
        else:
            # For other files, remove .yaml extension
            shortcode = str(relative_path.with_suffix(''))
        
        # Skip empty shortcodes and deprecated ones (optional)
        if shortcode and shortcode != ".":
            available_shortcodes.append(shortcode)
    
    return sorted(available_shortcodes)


def validate_shortcode(shortcode: str, config_path: str) -> Tuple[bool, str]:
    """
    Validate if a shortcode is valid and exists.
    
    Args:
        shortcode: The shortcode to validate (without @ prefix)
        config_path: Path to the configurations directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not shortcode:
        return False, "Shortcode cannot be empty"
    
    # Check for invalid characters
    if any(char in shortcode for char in ['@', ' ', '\t', '\n']):
        return False, f"Shortcode '{shortcode}' contains invalid characters"
    
    base_path = f"{config_path}/shortcode/{shortcode}/base.yaml"
    default_path = f"{config_path}/shortcode/{shortcode}.yaml"
    
    # Check if either base.yaml or direct .yaml exists
    if os.path.exists(base_path) or os.path.exists(default_path):
        return True, ""
    
    # If not found, provide helpful error message with suggestions
    available_shortcodes = get_available_shortcodes(config_path)
    
    # Find similar shortcodes for suggestions
    suggestions = []
    shortcode_lower = shortcode.lower()
    
    # First, look for exact substring matches
    for available in available_shortcodes:
        if shortcode_lower in available.lower() or available.lower() in shortcode_lower:
            suggestions.append(available)
    
    # If no substring matches, look for shortcodes with similar parts
    if not suggestions:
        shortcode_parts = shortcode_lower.split('/')
        for available in available_shortcodes:
            available_parts = available.lower().split('/')
            # Check if they share common path parts
            common_parts = set(shortcode_parts) & set(available_parts)
            if len(common_parts) >= min(2, len(shortcode_parts)):
                suggestions.append(available)
    
    # Limit to reasonable number of suggestions
    suggestions = suggestions[:10]
    
    error_msg = f"Shortcode '@{shortcode}' not found."
    
    if suggestions:
        error_msg += f"\n\nDid you mean one of these?\n"
        for suggestion in suggestions[:5]:  # Show top 5 suggestions
            error_msg += f"  @{suggestion}\n"
    else:
        error_msg += f"\n\nAvailable shortcodes include:\n"
        for shortcode_name in available_shortcodes[:10]:  # Show first 10
            error_msg += f"  @{shortcode_name}\n"
        if len(available_shortcodes) > 10:
            error_msg += f"  ... and {len(available_shortcodes) - 10} more\n"
    
    return False, error_msg


def _dict_to_str(d: dict) -> str:
    """
    Convert a dictionary to a string without quotes.
    """
    output = "{"
    for key, value in d.items():
        if value is None:
            value = "null"
        output += (
            f"{key}: {_dict_to_str(value) if isinstance(value, dict) else value}, "
        )
    output = output[:-2] + "}"
    return output


def _yaml_to_cli(
    yaml_path: str,
    prefix: Optional[str] = None,
) -> list[str]:
    """
    Convert a yaml file to a list of command line arguments.
    """
    cfg = OmegaConf.load(yaml_path)
    cli = []
    for key, value in OmegaConf.to_container(cfg).items():
        if value is None:
            value = "null"
        cli.append(
            f"++{prefix + '.' if prefix else ''}{key}={_dict_to_str(value) if isinstance(value, dict) else value}"
        )
    return cli


def validate_tags_param(tags_str: str) -> Tuple[bool, str, List[str]]:
    """
    Validate and parse tags parameter.
    
    Args:
        tags_str: Tags string from command line (e.g., '["dfot", "baseline"]' or 'dfot,baseline')
        
    Returns:
        Tuple of (is_valid, error_message, parsed_tags_list)
    """
    if not tags_str.strip():
        return False, "Tags cannot be empty", []
    
    try:
        # Try to parse as JSON list first
        if tags_str.strip().startswith('[') and tags_str.strip().endswith(']'):
            import json
            tags_list = json.loads(tags_str)
            if not isinstance(tags_list, list):
                return False, "Tags must be a list", []
            
            # Validate each tag
            for tag in tags_list:
                if not isinstance(tag, str):
                    return False, f"All tags must be strings, got: {type(tag).__name__}", []
                if not tag.strip():
                    return False, "Tags cannot be empty strings", []
                # Check for invalid characters
                if any(char in tag for char in [' ', '\t', '\n', '"', "'"]):
                    return False, f"Tag '{tag}' contains invalid characters (spaces, quotes, etc.)", []
            
            return True, "", [tag.strip() for tag in tags_list]
        
        # Otherwise, try comma-separated format
        else:
            tags_list = [tag.strip() for tag in tags_str.split(',')]
            
            # Validate each tag
            for tag in tags_list:
                if not tag:
                    return False, "Tags cannot be empty", []
                # Check for invalid characters
                if any(char in tag for char in [' ', '\t', '\n', '"', "'"]):
                    return False, f"Tag '{tag}' contains invalid characters (spaces, quotes, etc.)", []
            
            return True, "", tags_list
            
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format for tags: {e}", []
    except Exception as e:
        return False, f"Error parsing tags: {e}", []


def validate_shortcode_params(argv: list[str], config_path: str) -> None:
    """
    Validate shortcode and tags parameters in command line arguments.
    
    Args:
        argv: Command line arguments
        config_path: Path to the configurations directory
        
    Raises:
        ValueError: If any parameter is invalid
    """
    # Import here to avoid circular imports

    for arg in argv:
        if arg.startswith("shortcode="):
            shortcode_path = arg.split("=", 1)[1]  # Use split with maxsplit=1 to handle paths with = signs
            
            # Validate the shortcode path
            is_valid, error_msg = validate_shortcode(shortcode_path, config_path)
            if not is_valid:
                rank_zero_print(f"\n{bold_red('ERROR:')} Invalid shortcode parameter!")
                rank_zero_print(f"Command: {' '.join(argv)}")
                rank_zero_print(f"Invalid shortcode: {shortcode_path}")
                rank_zero_print(f"\n{error_msg}")
                sys.exit(1)
            else:
                rank_zero_print(cyan(f" =========== ✓ Shortcode validation passed: @{shortcode_path} =========== "))
        
        elif arg.startswith("tags=") or arg.startswith("+tags="):
            tags_str = arg.split("=", 1)[1]
            
            # Validate the tags
            is_valid, error_msg, tags_list = validate_tags_param(tags_str)
            if not is_valid:
                rank_zero_print(f"\n{bold_red('ERROR:')} Invalid tags parameter!")
                rank_zero_print(f"Command: {' '.join(argv)}")
                rank_zero_print(f"Invalid tags: {tags_str}")
                rank_zero_print(f"\n{error_msg}")
                rank_zero_print(f"\nValid formats:")
                rank_zero_print(f"  - JSON list: +tags='[\"dfot\", \"baseline\"]'")
                rank_zero_print(f"  - Comma-separated: +tags='dfot,baseline'")
                rank_zero_print(f"  - Note: Use +tags= (not tags=) for Hydra compatibility")
                sys.exit(1)
            else:
                rank_zero_print(cyan(f" =========== ✓ Tags validation passed: {tags_list} =========== "))


def unwrap_shortcuts(
    argv: list[str],
    config_path: str,
    config_name: str,
) -> list[str]:
    """
    Unwrap shortcuts by replacing them with commands from corresponding yaml files.
    All shortcuts should be in the form of `@shortcut_name`.
    Example:
    - @latent -> unwrap configurations/shortcut/latent/base.yaml and configurations/shortcut/latent/dataset_name.yaml
    - @mit_vision/h100 -> unwrap configurations/shortcut/mit_vision/h100.yaml
    """
    # find the default dataset
    defaults = OmegaConf.load(f"{config_path}/{config_name}.yaml").defaults
    dataset = next(default.dataset for default in defaults if "dataset" in default)
    # check if dataset is overridden
    for arg in argv:
        if arg.startswith("dataset="):
            dataset = arg.split("=")[1]

    if dataset is None:
        raise ValueError("Dataset name is not provided.")

    new_argv = []
    for arg in argv:
        if arg.startswith("@"):
            shortcut = arg[1:]
            
            # Validate the shortcode first
            is_valid, error_msg = validate_shortcode(shortcut, config_path)
            if not is_valid:
                raise ValueError(error_msg)
            
            base_path = f"{config_path}/shortcut/{shortcut}/base.yaml"

            if os.path.exists(base_path):
                new_argv += _yaml_to_cli(base_path)
                dataset_path = f"{config_path}/shortcut/{shortcut}/{dataset}.yaml"
                if os.path.exists(dataset_path):
                    new_argv += _yaml_to_cli(dataset_path)
            else:
                default_path = f"{config_path}/shortcut/{shortcut}.yaml"
                if os.path.exists(default_path):
                    new_argv += _yaml_to_cli(default_path)
                else:
                    # This should not happen if validation passed, but keep as fallback
                    raise ValueError(f"Shortcut @{shortcut} not found.")
        elif arg.startswith("algorithm/backbone="):
            # this is a workaround to enable overriding the backbone in the command line
            # otherwise, the backbone could be re-overridden by
            # the backbone cfgs in dataset-experiment dependent cfgs
            new_argv += override_backbone(arg[19:])
        else:
            new_argv.append(arg)

    return new_argv


def override_backbone(name: str) -> list[str]:
    """
    Override the backbone with the specified name.
    """
    return ["algorithm.backbone=null"] + _yaml_to_cli(
        f"configurations/algorithm/backbone/{name}.yaml", prefix="algorithm.backbone"
    )


def list_available_shortcodes(config_path: str = "configurations") -> None:
    """
    Print all available shortcodes in a formatted way.
    
    Args:
        config_path: Path to the configurations directory
    """
    available_shortcodes = get_available_shortcodes(config_path)
    
    if not available_shortcodes:
        rank_zero_print("No shortcodes found.")
        return
    
    rank_zero_print(f"Available shortcodes ({len(available_shortcodes)} total):")
    rank_zero_print("=" * 50)
    
    # Group shortcodes by category
    categories = {}
    for shortcode in available_shortcodes:
        parts = shortcode.split('/')
        category = parts[0] if len(parts) > 1 else "root"
        
        if category not in categories:
            categories[category] = []
        categories[category].append(shortcode)
    
    for category, shortcodes in sorted(categories.items()):
        rank_zero_print(f"\n{category.upper()}:")
        for shortcode in sorted(shortcodes):
            rank_zero_print(f"  @{shortcode}")


if __name__ == "__main__":
    # When run directly, list all available shortcodes
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--list-shortcodes":
        list_available_shortcodes()
    else:
        list_available_shortcodes()
