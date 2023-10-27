import tkinter as tk
from tkinter import ttk, Menu
import sv_ttk


class SliderContainer(ttk.Frame):
    def __init__(self, master, label_text, **slider_params):
        super().__init__(master)
        self.callback = None

        if "callback" in slider_params.keys():
            self.callback = slider_params["callback"]
            del slider_params["callback"]

        self.pack(pady=10, fill=tk.X, expand=True)

        label = ttk.Label(self, text=label_text)
        label.pack(side=tk.LEFT, padx=10)

        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill=tk.X, expand=True, padx=10)

        min_label = ttk.Label(slider_frame, text=f"{slider_params['from_']}")
        min_label.grid(row=0, column=0, sticky="w")

        max_label = ttk.Label(slider_frame, text=f"{slider_params['to']}")
        max_label.grid(row=0, column=2, sticky="e")

        value_label = ttk.Label(slider_frame, text="")
        value_label.grid(row=0, column=1, sticky="nsew")

        slider = ttk.Scale(slider_frame, **slider_params)
        slider.grid(row=1, column=0, columnspan=3, sticky="nsew")

        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=0)
        slider_frame.columnconfigure(2, weight=1)

        self.slider = slider
        self.value_label = value_label
        self.slider.bind("<Motion>", self.__motion_callback)

    def bind_update(self, callback):
        self.callback = callback

    def get_value(self):
        return self.slider.get()

    def __motion_callback(self, event):
        current_val = self.slider.get()
        self.value_label["text"] = f"{current_val:.2f}"
        if self.callback and callable(self.callback):
            self.callback(current_val)


class RadioContainer(ttk.Frame):
    def __init__(self, master, options, variable, **kwargs):
        super().__init__(master)
        self.variable = variable
        for text, value in options:
            radio = ttk.Radiobutton(
                self, text=text, variable=self.variable, value=value, **kwargs
            )
            radio.pack(fill=tk.X, pady=5)

    def get_value(self):
        """Return the currently selected radio value"""
        return self.variable.get()


class MenuContainer(Menu):
    def __init__(self, master, menu_items=None, **kwargs):
        super().__init__(master, **kwargs)

        if menu_items:
            for label, command in menu_items:
                self.add_command(label=label, command=command)


class MainWindow(tk.Tk):
    def __init__(self, title, image_viewer):
        super().__init__()
        self.title(title)
        self.image_viewer = image_viewer

    def set_menu_container(self, menu_container):
        self.menu_container = menu_container
        self.config(menu=self.menu_container)

    def add_container(self, container):
        container.pack(fill=tk.BOTH, expand=True)

    def run(self):
        sv_ttk.set_theme("dark")

        self.mainloop()
