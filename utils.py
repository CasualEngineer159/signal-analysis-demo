import tkinter as tk
from tkinter import ttk

class ScrollableFrame(ttk.Frame):
    """
    A custom ttk.Frame that is scrollable with a mousewheel or a scrollbar.
    Widgets are placed into the `self.scrollable_frame`.
    """
    def __init__(self, container, h_scroll=False, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        if h_scroll:
            h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=canvas.xview)
            canvas.configure(xscrollcommand=h_scrollbar.set)
            h_scrollbar.pack(side="bottom", fill="x")
            
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        self.window_id = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set)
        
        if not h_scroll:
            canvas.bind("<Configure>", lambda e: canvas.itemconfig(self.window_id, width=e.width))

        # Bind mousewheel scrolling
        self.bind_all("<MouseWheel>", lambda event: self._on_mousewheel(event, canvas, h_scroll))

        # Pack scrollbars before canvas so they aren't clipped if the frame gets too small
        v_scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

    def _on_mousewheel(self, event, canvas, h_scroll):
        # On Windows, the delta is usually a multiple of 120
        # If Shift is held, scroll horizontally
        if event.state & 0x0001 and h_scroll:  # Shift key mask
            canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class RoundedStringVar(tk.StringVar):
    """
    A custom tkinter StringVar that automatically rounds the displayed value
    to two decimal places when it's not being actively edited by the user.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_editing = False

    def set(self, value):
        try:
            if not self.is_editing:
                value = f"{float(value):.2f}"
        except (ValueError, TypeError):
            pass
        super().set(value)

def create_slider_entry(parent, label, var, from_, to, command):
    """
    Creates a composite widget with a label, a slider, and a text entry.
    """
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5, padx=5)
    ttk.Label(frame, text=label).pack(side=tk.TOP, anchor=tk.W)

    string_var = RoundedStringVar(value=f"{var.get():.2f}")

    def update_var_from_string(*args):
        try:
            # FIX: Replace comma with period to handle different locales
            value_str = string_var.get().replace(',', '.')
            var.set(float(value_str))
        except (ValueError, TypeError):
            pass

    def update_string_from_var(*args):
        if not string_var.is_editing:
            string_var.set(var.get())

    var.trace_add("write", update_string_from_var)
    string_var.trace_add("write", update_var_from_string)

    entry = ttk.Entry(frame, textvariable=string_var, width=10)
    entry.pack(side=tk.RIGHT, padx=(5, 0))

    def on_entry_focus_in(event):
        string_var.is_editing = True

    def on_entry_focus_out(event):
        string_var.is_editing = False
        string_var.set(var.get())
        command()

    entry.bind("<FocusIn>", on_entry_focus_in)
    entry.bind("<FocusOut>", on_entry_focus_out)
    entry.bind("<Return>", lambda e: command())

    slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL, variable=var, command=command)
    slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
    
    frame.slider = slider
    return frame
