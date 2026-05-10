import tkinter as tk
from tkinter import ttk

class PipelineListPanel:
    def __init__(self, parent, controllers, on_pipeline_changed, config_panel, component_map, signal_types, on_add_component):
        self.controllers = controllers
        self.on_pipeline_changed = on_pipeline_changed
        self.config_panel = config_panel
        self.component_map = component_map
        self.signal_types = signal_types
        self.on_add_component = on_add_component
        
        self.selected_controller = None
        self.config_frame = None
        self._drag_data = {"item_index": None}

        self.setup_component_panels(parent)

    def setup_component_panels(self, parent):
        # Execution Pipeline
        pipeline_frame = ttk.LabelFrame(parent, text="Execution Pipeline")
        pipeline_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.pipeline_listbox = tk.Listbox(pipeline_frame, height=6, exportselection=False)
        self.pipeline_listbox.pack(fill=tk.X, padx=5, pady=5)
        self.pipeline_listbox.bind('<<ListboxSelect>>', self.on_component_select)
        
        # Drag and Drop bindings
        self.pipeline_listbox.bind('<Button-1>', self.on_drag_start)
        self.pipeline_listbox.bind('<B1-Motion>', self.on_drag_motion)
        self.pipeline_listbox.bind('<ButtonRelease-1>', self.on_drag_release)
        
        btn_frame = ttk.Frame(pipeline_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Button(btn_frame, text="Move Up", command=self.move_component_up).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(btn_frame, text="Move Down", command=self.move_component_down).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 2))
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
            
            # Maintain selection if it was the selected item
            if self.selected_controller and self.selected_controller.id == controller.id:
                self.pipeline_listbox.selection_clear(0, tk.END)
                self.pipeline_listbox.selection_set(new_index)
            else:
                # If we moved an item past the currently selected one, we need to fix the visual selection
                if self.selected_controller:
                    try:
                        sel_idx = self.controllers.index(self.selected_controller)
                        self.pipeline_listbox.selection_clear(0, tk.END)
                        self.pipeline_listbox.selection_set(sel_idx)
                    except ValueError:
                        pass
                        
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
        controller_to_remove = self.controllers.pop(idx)
        if self.selected_controller and self.selected_controller.id == controller_to_remove.id:
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None
        self.on_pipeline_changed()

    def move_component_up(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == 0: return # Already at top
        
        # Swap in controllers
        self.controllers[idx - 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx - 1]
        
        # Swap in listbox
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx - 1, text)
        self.pipeline_listbox.selection_set(idx - 1)
        
        self.on_pipeline_changed()

    def move_component_down(self):
        idxs = self.pipeline_listbox.curselection()
        if not idxs: return
        idx = idxs[0]
        if idx == len(self.controllers) - 1: return # Already at bottom
        
        # Swap in controllers
        self.controllers[idx + 1], self.controllers[idx] = self.controllers[idx], self.controllers[idx + 1]
        
        # Swap in listbox
        text = self.pipeline_listbox.get(idx)
        self.pipeline_listbox.delete(idx)
        self.pipeline_listbox.insert(idx + 1, text)
        self.pipeline_listbox.selection_set(idx + 1)
        
        self.on_pipeline_changed()

    def on_component_select(self, event=None):
        idxs = self.pipeline_listbox.curselection()
        if not idxs:
            if self.config_frame: self.config_frame.destroy()
            self.selected_controller = None
            self.on_pipeline_changed()
            return
        idx = idxs[0]
        self.selected_controller = self.controllers[idx]
        if self.config_frame: self.config_frame.destroy()
        self.config_frame = ttk.Frame(self.config_panel)
        self.config_frame.pack(fill=tk.X, padx=5, pady=5)
        if self.selected_controller:
            self.selected_controller.get_config_frame(self.config_frame)
        self.on_pipeline_changed()

    def add_to_listbox(self, controller):
        self.pipeline_listbox.insert(tk.END, str(controller))
        
    def clear_listbox(self):
        self.pipeline_listbox.delete(0, tk.END)
        if self.config_frame: self.config_frame.destroy()
        self.selected_controller = None
