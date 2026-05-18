
# Change Report

This report details the evolution of the project from a basic signal generator to a more robust and user-friendly tool. The changes are broken down by file.

---

### `models/base_model.py`

This file defines the foundational abstract classes for all signal and anomaly components.

*   **Initial State**: The file started with `BaseComponentModel`, `SignalComponentModel`, and `AnomalyComponentModel`, establishing the core `generate`, `to_dict`, and `from_dict` methods.
*   **Enhancements**:
    *   A `get_anomaly_times` method was added to the `BaseComponentModel` to provide a standardized way for components to report ground truth events.
    *   The `generate` method in `SignalComponentModel` was modified to use an exclusive end index (`side='left'`) when applying signals. This was a key change to prevent an issue where two adjacent signals would both affect the same single data point, causing an unwanted spike in amplitude.

---

### `models/signal_models.py`

This file contains the data models for all primary signal sources.

*   **Initial State**: The file defined the basic signal generation logic for `SineModel`, `CosineModel`, `SquareModel`, `ChirpModel`, and `SineVaryingFreqModel`.
*   **Enhancements**:
    *   **Time Windowing**: The `_generate_signal` methods for `ChirpModel` and `SineVaryingFreqModel` were significantly updated to respect the `start_time` and `end_time` parameters inherited from the base class. This ensures that signals with internal time-based logic (like a chirp's duration or a frequency change time) behave correctly within their assigned window.
    *   The `get_anomaly_times` method was implemented for `SineVaryingFreqModel` to report the exact time of the frequency change.

---

### `components/signals.py`

This file manages the UI controllers for the signal source components.

*   **Initial State**: Each controller created UI sliders for its model's basic parameters (e.g., amplitude, frequency).
*   **Enhancements**:
    *   **Time Control UI**: "Start Time" and "End Time" sliders were added to every signal controller, allowing users to define the active window for each signal component.
    *   **Dynamic Slider Ranges**: The code was updated to ensure that these new time-based sliders, along with the existing "Change Time" slider, are correctly added to the `config_widgets` list. This allows their maximum range to be dynamically updated when the global signal duration changes.

---

### `ui_pipeline_list.py`

This file handles the main component pipeline UI.

*   **Initial State**: The panel contained the pipeline listbox and buttons for adding, removing, and reordering components. Parameter editing was handled directly in the main window.
*   **Enhancements**:
    *   **Settings Button**: A "Settings" button was added to the button bar.
    *   **Double-Click Event**: A `<Double-1>` event binding was added to the listbox.
    *   **Popup Logic**: Both the new button and the double-click event now trigger the `on_open_settings` callback, signaling the main application to open the new settings popup.

---

### `ui_settings_popup.py`

*   **New File**: This file was created to encapsulate all logic for the component settings popup window, promoting better code structure.
*   **Key Features**:
    *   **Modal Window**: It creates a `Toplevel` window that is modal (`grab_set()`), preventing interaction with the main window until it is closed.
    *   **Dynamic Sizing & Centering**: The popup automatically calculates the required height to fit its content and positions itself in the center of the main application window.
    *   **Live Preview**: It contains a dedicated Matplotlib canvas that shows a live preview of the selected component's output, which updates in real-time as parameters are changed.
    *   **Callback Management**: It temporarily overrides the selected controller's update callback to ensure that both the main plot and the popup's preview plot are updated simultaneously.

---

### `main.py`

This is the main application entry point.

*   **Initial State**: The main window was responsible for displaying everything, including the component parameters and the preview plot, leading to a cluttered UI.
*   **Refactoring**:
    *   The "Parameters" and "Component Preview" frames were removed from the main layout.
    *   The `open_settings_popup` method was added. This method now instantiates the new `SettingsPopup` class from `ui_settings_popup.py`.
    *   The application now manages the lifecycle of the popup, ensuring only one is open at a time and that the main plot is refreshed when it closes.
