import json

class ConfigManager:
    @staticmethod
    def save_to_file(filepath: str, ui_settings_dict: dict, components_list: list) -> None:
        """
        Saves the UI settings and component configurations to a JSON file.
        """
        config = {
            'ui_settings': ui_settings_dict,
            'components': components_list
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load_from_file(filepath: str) -> tuple[dict, list]:
        """
        Loads the UI settings and component configurations from a JSON file.
        Returns a tuple of (ui_settings_dict, components_list).
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        ui_settings = config.get('ui_settings', {})
        components = config.get('components', [])
        
        return ui_settings, components
