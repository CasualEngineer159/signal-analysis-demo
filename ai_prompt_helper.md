# AI Prompt Helper

This file provides a summary of the project structure and the contents of each file to assist with AI prompting.

## Project Overview

The project is a Signal Analysis Tool built with Python and Tkinter. It allows users to generate signals, add anomalies, and perform analysis such as FFT and STFT. The application is structured into several modules, including a main application file, utility functions, analysis functions, component controllers, data models, and UI panels.

## File Summaries

### `main.py`

*   **Purpose:** The main entry point of the application.
*   **Key Components:**
    *   `SignalGeneratorApp`: The main application class that initializes the UI and manages the overall application state.
    *   `COMPONENT_MAP`: A dictionary that maps component names to their corresponding model and controller classes.
    *   Initializes the main window, panels, and plot manager.
    *   Handles saving and loading of configurations.

### `utils.py`

*   **Purpose:** Provides utility classes and functions for the application.
*   **Key Components:**
    *   `ScrollableFrame`: A custom Tkinter frame that supports scrolling.
    *   `RoundedStringVar`: A custom `tk.StringVar` that rounds the displayed value.
    *   `create_slider_entry`: A function that creates a composite widget with a label, a slider, and a text entry.

### `analysis.py`

*   **Purpose:** Contains functions for signal analysis.
*   **Key Functions:**
    *   `perform_fft`: Performs a Fast Fourier Transform on a given signal.
    *   `find_fft_peaks`: Finds peaks in the FFT spectrum.
    *   `perform_stft`: Performs a Short-Time Fourier Transform on a given signal.
    *   `calculate_spectral_flux`: Calculates the spectral flux of an STFT result and finds anomalies.
    *   `get_matched_pairs`: Matches ground truth times to predicted times.
    *   `evaluate_detection`: Evaluates anomaly detection performance.

### `components/signals.py`

*   **Purpose:** Defines the UI controllers for the various signal types.
*   **Key Classes:**
    *   `SineController`, `CosineController`, `SquareController`, `ChirpController`, `SineVaryingFreqController`: These classes are responsible for creating the UI elements (sliders, entries) for configuring the parameters of their respective signal models.

### `components/anomalies.py`

*   **Purpose:** Defines the UI controllers for the various signal anomalies.
*   **Key Classes:**
    *   `GaussianNoiseController`, `ImpulseNoiseController`, `AmplitudeJumpController`, `BiasController`, `DriftController`, `DropoutController`, `SaturationController`, `OutlierController`, `TimeDelayController`: These classes create the UI for configuring the parameters of the anomaly models.

### `models/signal_models.py`

*   **Purpose:** Defines the data models for the different signal types.
*   **Key Classes:**
    *   `SineModel`, `CosineModel`, `SquareModel`, `ChirpModel`, `SineVaryingFreqModel`: These dataclasses define the parameters and generation logic for each signal type.

### `models/anomaly_models.py`

*   **Purpose:** Defines the data models for the different anomaly types.
*   **Key Classes:**
    *   `GaussianNoiseModel`, `ImpulseNoiseModel`, `AmplitudeJumpModel`, `BiasModel`, `DriftModel`, `DropoutModel`, `SaturationModel`, `OutlierModel`, `TimeDelayModel`: These dataclasses define the parameters and application logic for each anomaly.

### `plot_manager.py`

*   **Purpose:** Manages the creation and updating of the various plots and results tables in the UI.
*   **Key Components:**
    *   `PlotManager`: A class that initializes and manages the Matplotlib plots for the time-domain signal, STFT, spectral flux, and FFT. It also manages the tables for displaying FFT peaks and detection results.

### `config_manager.py`

*   **Purpose:** Handles saving and loading the application's configuration.
*   **Key Methods:**
    *   `save_to_file`: Saves the UI settings and component configurations to a JSON file.
    *   `load_from_file`: Loads the UI settings and component configurations from a JSON file.

### `signal_engine.py`

*   **Purpose:** Responsible for generating the signal data and running the analysis pipeline.
*   **Key Functions:**
    *   `generate_pipeline_data`: Takes the list of controllers and settings, generates the signal, applies anomalies, performs STFT and FFT, calculates spectral flux, and evaluates detection performance.

### `ui_pipeline_list.py`

*   **Purpose:** Manages the UI for the component pipeline.
*   **Key Components:**
    *   `PipelineListPanel`: A class that creates and manages the listbox for the execution pipeline, allowing users to add, remove, and reorder components.

### `ui_settings_panel.py`

*   **Purpose:** Manages the UI for the global settings.
*   **Key Components:**
    *   `SettingsPanel`: A class that creates and manages the UI for global time settings, STFT settings, and spectral flux settings.
