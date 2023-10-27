import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_viewer import ImageViewer
from gui_elements import SliderContainer, RadioContainer, MenuContainer, MainWindow
import common

class FourierContainer(ttk.Frame):
    def __init__(self, master, image_viewer_original, image_viewer_transformed, **kwargs):
        super().__init__(master, **kwargs)
        self.image_viewer_original = image_viewer_original
        self.image_viewer_transformed = image_viewer_transformed
        self.init_ui()

    def init_ui(self):
        self.canvas_frame_original = ttk.Frame(self)
        self.canvas_frame_original.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.canvas_frame_transformed = ttk.Frame(self)
        self.canvas_frame_transformed.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=10)

        self.btn_fourier_transform = ttk.Button(self, text="Compute Fourier Transform", command=self.compute_transform)
        self.btn_fourier_transform.pack(pady=20)

    def compute_transform(self):
        image = self.image_viewer_original.get_roi()
        if image is not None:
            gray_image = common.simple_cvtColorBGRtoGray(image)
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            
            # Sample frequency filtering using a mask
            rows, cols = gray_image.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 10  # Radius of the mask
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 > r*r
            mask[mask_area] = 0

            # Apply mask and inverse DFT
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
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

    fourier_container = FourierContainer(main_window, viewer_original, viewer_transformed)
    main_window.add_container(fourier_container)

    viewer_thread_original = threading.Thread(target=viewer_original.run)
    viewer_thread_original.start()

    viewer_thread_transformed = threading.Thread(target=viewer_transformed.run)
    viewer_thread_transformed.start()

    main_window.run()

    viewer_thread_original.join()
    viewer_thread_transformed.join()
