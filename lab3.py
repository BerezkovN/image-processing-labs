import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)  # Required for embedding matplotlib figure

from image_viewer import ImageViewer
from gui_elements import SliderContainer, RadioContainer, MenuContainer, MainWindow
import common


class HistogramContainer(ttk.Frame):
    def __init__(self, master, image_viewer_original, image_viewer_equalized, **kwargs):
        super().__init__(master, **kwargs)
        self.image_viewer_original = image_viewer_original
        self.image_viewer_equalized = image_viewer_equalized
        self.init_ui()

    def init_ui(self):
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.canvas_frame_equalized = ttk.Frame(self)
        self.canvas_frame_equalized.pack(
            fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10
        )

        self.btn_update_histogram = ttk.Button(
            self, text="Equalize Histogram", command=self.update_histogram
        )
        self.btn_update_histogram.pack(pady=20)

    def update_histogram(self):
        image = self.image_viewer_original.get_roi()
        if image is not None:
            gray_image = common.simple_cvtColorBGRtoGray(image)
            equalized_image = common.equalizeHist(gray_image)
            self.image_viewer_equalized.set_image(equalized_image)

            data = [gray_image.flatten()]
            labels = ["Gray"]
            colors = ["gray"]
            self.plot_histogram(data, labels, colors, self.canvas_frame)

            data_eq = [equalized_image.flatten()]
            self.plot_histogram(data_eq, labels, colors, self.canvas_frame_equalized)

    def plot_histogram(self, data, labels, colors, frame):
        fig, ax = plt.subplots()

        for d, label, color in zip(data, labels, colors):
            ax.hist(d, bins=256, label=label, alpha=0.6, color=color, range=[0, 255])
        ax.legend()
        ax.set_title("Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")

        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    viewer_original = ImageViewer("Original Image")
    viewer_equalized = ImageViewer("Equalized Image")

    main_window = MainWindow("Histogram GUI", viewer_original)
    main_window.minsize(640, 480)

    menu = MenuContainer(
        main_window,
        [
            ["Open Image", lambda: common.open_image(main_window)],
            ["Save Image", lambda: common.save_image_viewer(viewer_equalized)],
        ],
    )

    main_window.set_menu_container(menu)

    histogram_container = HistogramContainer(
        main_window, viewer_original, viewer_equalized
    )
    main_window.add_container(histogram_container)

    viewer_thread_original = threading.Thread(target=viewer_original.run)
    viewer_thread_original.start()

    viewer_thread_equalized = threading.Thread(target=viewer_equalized.run)
    viewer_thread_equalized.start()

    main_window.run()

    viewer_thread_original.join()
    viewer_thread_equalized.join()
