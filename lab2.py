import numpy as np
import tkinter as tk
from tkinter import ttk
import threading

from image_viewer import ImageViewer
from gui_elements import SliderContainer, RadioContainer, MenuContainer, MainWindow
import common



class TransformationContainer(ttk.Frame):
    TRANSFORMATIONS = {
        "none": lambda self, img: img,
        "negative": lambda self, img: 256 - 1 - img,
        "logarithmic": lambda self, img: self.logarithmic_transformation(img),
        "gamma": lambda self, img: self.gamma_transformation(img),
    }

    def __init__(self, master, image_viewer, **kwargs):
        super().__init__(master, **kwargs)
        self.image_viewer = image_viewer
        self.init_ui()

    def init_ui(self):
        self._init_sliders()
        self._init_radios()

    def _init_sliders(self):
        self.c_log_slider = SliderContainer(
            self,
            "c value for Logarithmic:",
            from_=-5,
            to=5,
            orient=tk.HORIZONTAL,
            callback=lambda val: self.apply_transformation_if("logarithmic"),
        )
        self.c_log_slider.pack(pady=10, fill=tk.X, expand=True)

        self.c_gamma_slider = SliderContainer(
            self,
            "c value for Gamma:",
            from_=0.01,
            to=1,
            orient=tk.HORIZONTAL,
            callback=lambda val: self.apply_transformation_if("gamma"),
        )
        self.c_gamma_slider.pack(pady=10, fill=tk.X, expand=True)

        self.gamma_slider = SliderContainer(
            self,
            "Gamma value:",
            from_=0.1,
            to=15,
            orient=tk.HORIZONTAL,
            callback=lambda val: self.apply_transformation_if("gamma"),
        )
        self.gamma_slider.pack(pady=10, fill=tk.X, expand=True)

    def _init_radios(self):
        self.transformation_var = tk.StringVar(value="none")
        self.radio_container = RadioContainer(
            self,
            [
                ("No transformation", "none"),
                ("Apply Negative Transformation", "negative"),
                ("Apply Logarithmic Transformation", "logarithmic"),
                ("Apply Gamma Transformation", "gamma"),
            ],
            self.transformation_var,
            command=self.apply_transformation,
        )
        self.radio_container.pack(pady=5, fill=tk.X)

    def logarithmic_transformation(self, img):
        c = self.c_log_slider.get_value()
        img_value = img.astype(np.float32)
        transformed_img = 255 / np.log(1 + pow(10, c) * 255) * np.log(1 + pow(10, c) * img_value)
        return np.clip(transformed_img, 0, 255).astype(np.uint8)

    def gamma_transformation(self, img):
        c_gamma = self.c_gamma_slider.get_value()
        gamma = self.gamma_slider.get_value()
        transformed_img = c_gamma * (img.astype(np.float32) / 255) ** gamma
        return np.clip(transformed_img * 255, 0, 255).astype(np.uint8)

    def apply_transformation_if(self, transformation_type):
        if self.transformation_var.get() == transformation_type:
            self.apply_transformation()

    def apply_transformation(self):
        if not hasattr(self.master, "image") or self.master.image is None:
            return
        transformation_type = self.transformation_var.get()
        new_image = self.TRANSFORMATIONS[transformation_type](self, self.master.image)
        self.image_viewer.set_image(new_image)


if __name__ == "__main__":
    viewer = ImageViewer()
    main_window = MainWindow("ToneMapping GUI", viewer)

    menu = MenuContainer(
        main_window,
        [
            ["Open Image", lambda: common.open_image(main_window)],
            ["Save Image", lambda: common.save_image(main_window)],
        ],
    )

    main_window.set_menu_container(menu)

    transformation_container = TransformationContainer(main_window, viewer)
    main_window.add_container(transformation_container)

    viewer_thread = threading.Thread(target=viewer.run)
    viewer_thread.start()

    main_window.run()
    viewer_thread.join()
