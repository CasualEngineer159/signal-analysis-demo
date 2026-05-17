import json

class ConfigManager:
    @staticmethod
    def save_to_file(filepath: str, ui_settings_dict: dict, components_list: list, spectral_flux_settings: dict) -> None:
        """
        Saves the UI settings and component configurations to a JSON file.
        """
        config = {
            'ui_settings': ui_settings_dict,
            'components': components_list,
            'spectral_flux': spectral_flux_settings
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_from_file(filepath: str) -> tuple[dict, list, dict]:
        """
        Loads the UI settings and component configurations from a JSON file.
        Returns a tuple of (ui_settings_dict, components_list, spectral_flux_settings).
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is invalid, return default values for all parts.
            return {}, [], {'rectify': False}

        # Use .get() to safely access keys that may not exist, providing a default value.
        ui_settings = config.get('ui_settings', {})
        components = config.get('components', [])
        spectral_flux_settings = config.get('spectral_flux', {'rectify': False})
        
        # This function will now always return three items, preventing the unpack error.
        return ui_settings, components, spectral_flux_settings
