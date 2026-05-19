# Signal Analysis Tool

A Signal Analysis Tool built with Python and Tkinter. It allows users to generate signals, add anomalies, and perform advanced signal analysis such as FFT and STFT.

## Demo Capabilities

* **Signal Generation**: Generate various signals including Sine, Cosine, Square, Chirp, and Varying Frequency Sine, with adjustable time windows.
* **Anomaly Insertion**: Inject a wide range of anomalies into signals such as Gaussian Noise, White Noise, Impulse Noise, Amplitude Jumps, Bias, Drift, Dropouts, Saturation, Outliers, and Time Delays.
* **Signal Analysis**:
  * **FFT**: Perform Fast Fourier Transform and find peaks in the spectrum.
  * **STFT**: Perform Short-Time Fourier Transform for time-frequency analysis.
  * **Spectral Flux**: Calculate spectral flux and detect anomalies automatically.
* **Performance Evaluation**: Evaluate detection performance by matching predicted anomaly times against ground truth events.
* **Interactive UI**: Build an execution pipeline of signals and anomalies, configure parameters dynamically.
* **Save/Load Configurations**: Save and load execution pipelines and UI settings to/from JSON files.

## File Structure

* `main.py` - Main entry point of the application that initializes the UI and manages overall state.
* `core/`
  * `analysis.py` - Functions for FFT, STFT, spectral flux calculation, and anomaly detection evaluation.
  * `config_manager.py` - Handles saving and loading the application's configuration.
  * `plot_manager.py` - Manages the Matplotlib plots and results tables in the UI.
  * `signal_engine.py` - Responsible for generating the signal data and running the analysis pipeline.
  * `utils.py` - Utility classes and functions (e.g., custom UI widgets).
* `components/`
  * `signals.py` - UI controllers for the various signal types.
  * `anomalies.py` - UI controllers for configuring the parameters of the anomaly models.
  * `base.py` - Base classes for the component UI controllers.
* `models/`
  * `base_model.py` - Foundational abstract classes establishing the core methods for all models.
  * `signal_models.py` - Data models defining parameters and logic for different signal types.
  * `anomaly_models.py` - Data models defining parameters and logic for different anomaly types.
* `ui/`
  * `ui_pipeline_list.py` - Manages the UI for the component execution pipeline.
  * `ui_settings_panel.py` - Manages the UI for the global application settings.
  * `ui_settings_popup.py` - Modal popup window to adjust component settings with a live preview canvas.
* `configs/` - Contains saved configuration and demo state files (JSON).
* `data/` - Contains data sets used for tests or examples (CSV).
* `tests/` - Contains automated tests for analysis, models, and the signal engine.
* `docs/` - Contains project documentation, helper files, and specifications.
