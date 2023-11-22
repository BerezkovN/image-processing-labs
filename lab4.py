import numpy as np
import tkinter as tk
import math
from tkinter import ttk, messagebox, simpledialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_viewer import ImageViewer
from gui_elements import SliderContainer, RadioContainer, MenuContainer, MainWindow
import common
from common import fft2, fftshift, ifftshift, ifft2


class FourierContainer(ttk.Frame):
    def __init__(self, master, image_viewer_original, image_viewer_transformed, image_viewer_filter, **kwargs):
        super().__init__(master, **kwargs)
        self.image_viewer_original = image_viewer_original
        self.image_viewer_transformed = image_viewer_transformed
        self.image_viewer_filter = image_viewer_filter
        self.init_ui()

    def init_ui(self):
        self.canvas_frame_original = ttk.Frame(self)
        self.canvas_frame_original.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.canvas_frame_transformed = ttk.Frame(self)
        self.canvas_frame_transformed.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.canvas_frame_filter = ttk.Frame(self)
        self.canvas_frame_filter.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.btn_fourier_transform = ttk.Button(self, text="Compute Fourier Transform", command=self.compute_transform)
        self.btn_fourier_transform.pack(pady=20)

        # Listbox for Filters
        self.listbox = tk.Listbox(self)
        self.listbox.pack(pady=20)
        self.listbox.bind("<<ListboxSelect>>", self.on_filter_select)
        self.listbox.insert(tk.END, "НЧ Ідельний")
        self.listbox.insert(tk.END, "ВЧ Ідеальний")
        self.listbox.insert(tk.END, "НЧ фільтра Баттерворта")
        self.listbox.insert(tk.END, "ВЧ фільтра Баттерворта")
        self.listbox.insert(tk.END, "НЧ фільтра Гауса")
        self.listbox.insert(tk.END, "ВЧ фільтра Гауса")


    def on_filter_select(self, event):
        index = self.listbox.curselection()

        if self.image_viewer_original.get_image() is None:
            width = 100
            height = 100
        else:
            width = self.image_viewer_original.width()
            height = self.image_viewer_original.height()

        if not index:
            index = 0;
            return;
        else:
            index = index[0]

        # Sample frequency filtering using a mask
        rows, cols = width, height
        crow, ccol = rows // 2, cols // 2
        self.mask = np.ones((rows, cols), np.double)

        if index == 0:

            radius = int(simpledialog.askinteger("", "Радіус:"))

            for indY in range(self.mask.shape[1]):
                for indX in range(self.mask.shape[0]):
                    value = self.D(indX, indY, width, height)
                    self.mask[indY][indX] = value <= radius if 1 else 0

        elif index == 1:
            radius = int(simpledialog.askinteger("", "Радіус:"))

            for indY in range(self.mask.shape[1]):
                for indX in range(self.mask.shape[0]):
                    value = self.D(indX, indY, width, height)
                    self.mask[indY][indX] = value >= radius if 1 else 0

        elif index == 2 or index == 3:
            radius = int(simpledialog.askinteger("", "Радіус:"))
            powOrder = int(simpledialog.askinteger("", "Порядок степені:"))

            for indY in range(self.mask.shape[1]):
                for indX in range(self.mask.shape[0]):
                    value = self.D(indX, indY, width, height)
                    self.mask[indY][indX] = 1/ (1 + pow(value / radius, powOrder * 2))

            if index == 3:
                self.mask = 1 - self.mask

        elif index == 4 or index == 5:
            radius = int(simpledialog.askinteger("", "Ширина Гаусівської кривої:"))

            for indY in range(self.mask.shape[1]):
                for indX in range(self.mask.shape[0]):
                    value = self.D(indX, indY, width, height)
                    self.mask[indY][indX] = math.exp(-(value**2) / (2*(radius ** 2)))

            if index == 5:
                self.mask = 1 - self.mask

        self.image_viewer_filter.set_image(self.mask * 255)
    
    def D(self, u, v, M, N):
        return math.sqrt((u - N / 2)**2 + (v - N / 2)**2) 

    def compute_transform(self):
        image = self.image_viewer_original.get_roi()
        if image is not None:
            gray_image = common.simple_cvtColorBGRtoGray(image)
            f = fft2(gray_image)
            fshift = fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            
            if self.mask is None:
                self.on_filter_select(None)

            # Apply mask and inverse DFT
            fshift = fshift * self.mask
            f_ishift = ifftshift(fshift)
            img_back = ifft2(f_ishift)
            img_back = np.abs(img_back)

            self.image_viewer_transformed.set_image(img_back.astype(np.uint8))
            self.plot_image(magnitude_spectrum, 'Fourier Spectrum', self.canvas_frame_transformed)

    def plot_image(self, image, title, frame):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":

    plt.style.use('dark_background')

    viewer_original = ImageViewer("Original Image")
    viewer_transformed = ImageViewer("Transformed Image") 
    viewer_filter = ImageViewer("Filter Image")

    main_window = MainWindow("Fourier Transform GUI", viewer_original)
    main_window.minsize(640, 480)

    menu = MenuContainer(
        main_window,
        [
            ["Open Image", lambda: common.open_image(main_window)],
            ["Save Image", lambda: common.save_image_viewer(viewer_transformed)],
        ],
    )

    main_window.set_menu_container(menu)

    fourier_container = FourierContainer(main_window, viewer_original, viewer_transformed, viewer_filter)
    main_window.add_container(fourier_container)

    viewer_thread_original = threading.Thread(target=viewer_original.run)
    viewer_thread_original.start()

    viewer_thread_transformed = threading.Thread(target=viewer_transformed.run)
    viewer_thread_transformed.start()

    viewer_thread_filter = threading.Thread(target=viewer_filter.run)
    viewer_thread_filter.start()

    main_window.run()

    viewer_thread_original.join()
    viewer_thread_transformed.join()
    viewer_thread_filter.join()
