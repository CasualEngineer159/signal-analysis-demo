import tkinter as tk
from tkinter import ttk

class PipelineListPanel:
    def __init__(self, parent, controllers, on_pipeline_changed, on_open_settings, component_map, signal_types, on_add_component):
        self.controllers = controllers
        self.on_pipeline_changed = on_pipeline_changed
        self.on_open_settings = on_open_settings
        self.component_map = component_map
        self.signal_types = signal_types
        self.on_add_component = on_add_component
        
        self.selected_controller = None
        self._drag_data = {"item_index": None}

        self.setup_component_panels(parent)

    def setup_component_panels(self, parent):
        # Execution Pipeline
        pipeline_frame = ttk.LabelFrame(parent, text="Execution Pipeline")
        pipeline_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.pipeline_listbox = tk.Listbox(pipeline_frame, height=6, exportselection=False)
        self.pipeline_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.pipeline_listbox.bind('<<ListboxSelect>>', self.on_component_select)
        self.pipeline_listbox.bind('<Double-1>', self.open_settings_popup)
        
        # Drag and Drop bindings
        self.pipeline_listbox.bind('<Button-1>', self.on_drag_start)
        self.pipeline_listbox.bind('<B1-Motion>', self.on_drag_motion)
        self.pipeline_listbox.bind('<ButtonRelease-1>', self.on_drag_release)
        
        btn_frame = ttk.Frame(pipeline_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Adjusted button layout and widths
        ttk.Button(btn_frame, text="Up", command=self.move_component_up).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="Down", command=self.move_component_down).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
        ttk.Button(btn_frame, text="Settings", command=self.open_settings_popup).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
        ttk.Button(btn_frame, text="Remove", command=self.remove_component).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Add Signal
        sig_frame = ttk.LabelFrame(parent, text="Add Signal")
        sig_frame.pack(fill=tk.X, pady=(5, 5))
        sig_btn_frame = ttk.Frame(sig_frame)
        sig_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        sig_classes = {k:v for k,v in self.component_map.items() if k in self.signal_types}
        sig_var = tk.StringVar(value=list(sig_classes.keys())[0])
        ttk.Combobox(sig_btn_frame, textvariable=sig_var, values=list(sig_classes.keys()), state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sig_btn_frame, text="Add", command=lambda: self.on_add_component(sig_var.get())).pack(side=tk.LEFT, padx=(5,0))

        # Add Anomaly
        anom_frame = ttk.LabelFrame(parent, text="Add Anomaly & Noise")
        anom_frame.pack(fill=tk.X, pady=(5, 5))
        anom_btn_frame = ttk.Frame(anom_frame)
        anom_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        anom_classes = {k:v for k,v in self.component_map.items() if k not in self.signal_types}
        anom_var = tk.StringVar(value=list(anom_classes.keys())[0])
        ttk.Combobox(anom_btn_frame, textvariable=anom_var, values=list(anom_classes.keys()), state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(anom_btn_frame, text="Add", command=lambda: self.on_add_component(anom_var.get())).pack(side=tk.LEFT, padx=(5,0))

    def open_settings_popup(self, event=None):
        """Calls the callback to open the settings window if an item is selected."""
        if self.pipeline_listbox.curselection():
            self.on_open_settings()

    def on_drag_start(self, event):
        """Record the item's starting index."""
        index = self.pipeline_listbox.nearest(event.y)
        if index >= 0:
            self._drag_data["item_index"] = index

    def on_drag_motion(self, event):
        """Move the item visually in the listbox as it is dragged."""
        current_index = self._drag_data["item_index"]
        if current_index is None:
            return
            
        new_index = self.pipeline_listbox.nearest(event.y)
        
        if new_index != current_index and new_index >= 0 and new_index < self.pipeline_listbox.size():
            # Update Listbox visually
            text = self.pipeline_listbox.get(current_index)
            self.pipeline_listbox.delete(current_index)
            self.pipeline_listbox.insert(new_index, text)
            
            # Update controllers array to match visual order
            controller = self.controllers.pop(current_index)
            self.controllers.insert(new_index, controller)
            
            # Maintain selection
            self.pipeline_listbox.selection_clear(0, tk.END)
            self.pipeline_listbox.selection_set(new_index)
                        
            self._drag_data["item_index"] = new_index

    def on_drag_release(self, event):
        """Finalize drag and trigger an update."""
        if self._drag_data["item_index"] is not None:
            self._drag_data["item_index"] = None
            self.on_pipeline_changed()

    def remove_component(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        self.pipeline_listbox.delete(idx)
        self.controllers.pop(idx)
        self.selected_controller = None
        # Select the next item or the last one if the removed item was at the end
        if self.pipeline_listbox.size() > 0:
            new_selection_idx = min(idx, self.pipeline_listbox.size() - 1)
            self.pipeline_listbox.selection_set(new_selection_idx)
            self.on_component_select()
        self.on_pipeline_changed()

    def move_component_up(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == 0: return
        
        self.controllers[idx - 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx - 1]
        
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx - 1, text)
        self.pipeline_listbox.selection_set(idx - 1)
        
        self.on_pipeline_changed()

    def move_component_down(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == len(self.controllers) - 1: return
        
        self.controllers[idx + 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx + 1]
        
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx + 1, text)
        self.pipeline_listbox.selection_set(idx + 1)
        
        self.on_pipeline_changed()

    def on_component_select(self, event=None):
        """Updates the currently selected controller based on listbox selection."""
        idxs = self.pipeline_listbox.curselection()
        if not idxs:
            self.selected_controller = None
            return
        idx = idxs[0]
        self.selected_controller = self.controllers[idx]

    def add_to_listbox(self, controller):
        self.pipeline_listbox.insert(tk.END, str(controller))
        
    def clear_listbox(self):
        self.pipeline_listbox.delete(0, tk.END)
        self.selected_controller = None
