import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
from image_viewer import ImageViewer
from gui_elements import SliderContainer, RadioContainer, MenuContainer, MainWindow
import common



class Filter:
    def __init__(self, name, matrix):
        self.name = name
        if isinstance(matrix, np.ndarray):
            self.matrix = matrix
        else:
            self.matrix = np.array(matrix)
        sum = np.sum(self.matrix)
        if sum != 0 and sum != 1:
            self.matrix = self.matrix / sum

    def __str__(self):
        return self.name


available_filters = [
    Filter("No filter", [[1]]),
    Filter("Gaussian 3x3", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])),
    Filter(
        "Gaussian 5x5",
        np.array(
            [
                [2, 7, 12, 7, 2],
                [7, 31, 52, 31, 7],
                [12, 52, 127, 52, 12],
                [7, 31, 52, 31, 7],
                [2, 7, 12, 7, 2],
            ]
        ),
    ),
    Filter("Laplacian 3x3", [[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    Filter(
        "Laplacian 5x5",
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ],
    ),
    Filter("Prewitt Horizontal", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    Filter("Prewitt Vertical", [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    Filter("Sobel Horizontal", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    Filter("Sobel Vertical", [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    Filter("Laplace High Pass Filter (3x3)", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    Filter(
        "Laplace High Pass Filter (5x5)",
        [
            [-1, -3, -4, -3, -1],
            [-3, 0, 6, 0, -3],
            [-4, 6, 20, 6, -4],
            [-3, 0, 6, 0, -3],
            [-1, -3, -4, -3, -1],
        ],
    ),
    Filter("Hipass", [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
    Filter("Edge detection", [[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
    Filter("Sharpen", [[-1, -1, -1], [-1, 16, -1], [-1, -1, -1]]),
    Filter("Softening", [[2, 2, 2], [2, 0, 2], [2, 2, 2]]),
]

current_filter_index = 0

class FilterContainer(ttk.Frame):
    def __init__(self, master, image_viewer, **kwargs):
        super().__init__(master, **kwargs)
        self.image_viewer = image_viewer
        self.prefiltered_image = None
        self.grayscale = False
        self.saved_path = None
        self.init_ui()

    def init_ui(self):
        self.pack(fill="both", expand=True, padx=10, pady=10)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.grid_rowconfigure(0, weight=0)  # grayscale_button
        self.grid_rowconfigure(1, weight=1)  # listbox
        self.grid_rowconfigure(2, weight=0)  # buttons
        self.grid_rowconfigure(3, weight=1)  # matrix_frame

        # Convert to Grayscale Button
        self.grayscale_button = ttk.Button(
            self,
            text="Convert to Grayscale",
            command=self.switch_colorscheme,
        )
        self.grayscale_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)

        # Listbox for Filters
        self.listbox = tk.Listbox(self)
        for filt in available_filters:
            self.listbox.insert(tk.END, filt.name)
        self.listbox.bind("<<ListboxSelect>>", self.on_filter_select)
        self.listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=10)

        # Edit & Custom Filter Buttons in the same row
        self.edit_button = ttk.Button(
            self,
            text="Edit Filter",
            command=self.edit_filter,
        )
        self.edit_button.grid(row=2, column=0, sticky="ew", pady=10, padx=5)

        self.custom_filter_button = ttk.Button(
            self,
            text="Custom Filter",
            command=self.custom_filter,
        )
        self.custom_filter_button.grid(row=2, column=1, sticky="ew", pady=10, padx=5)

        # Matrix Frame
        self.matrix_frame = ttk.Frame(self)
        self.matrix_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky="ew")

        self.cells = []

    def notify_image_changed(self):
        if self.grayscale:
            self.grayscale = not self.grayscale
            self.switch_colorscheme()
        else:
            self.prefiltered_image = self.master.image.copy()
            self.push_current_filter()

    def switch_colorscheme(self):
        self.grayscale = not self.grayscale

        if not self.grayscale and self.master.image is not None:
            self.prefiltered_image = self.master.image.copy()

        elif self.grayscale and self.master.image is not None:
            self.prefiltered_image = common.simple_cvtColorBGRtoGray(self.master.image)
        self.push_current_filter()

    def push_current_filter(self):
        index = self.listbox.curselection()
        if index:
            selected_filter = available_filters[index[0]]
            self.filtered_image = self.apply_filter(selected_filter.matrix)
            self.push_image()
        else:
            self.filtered_image = self.prefiltered_image
            self.push_image()

    def apply_filter(self, matrix):
        if self.prefiltered_image is not None:
            return common.filter2D(self.prefiltered_image, matrix)
        return None

    def push_image(self):
        self.image_viewer.set_image(self.filtered_image)

    def live_filter_update(self, event):
        try:
            temp_matrix = []
            for row in self.cells:
                new_row = []
                for cell in row:
                    value = float(cell.get())
                    new_row.append(value)
                temp_matrix.append(new_row)
            temp_matrix = np.array(temp_matrix)
            self.filtered_image = self.apply_filter(temp_matrix)
            self.push_image()
        except ValueError:
            pass  # If there's an error, probably because of incomplete input, so just pass

    def on_filter_select(self, event):
        index = self.listbox.curselection()
        if index:
            selected_filter = available_filters[index[0]]
            current_filter = index[0]
            self.filtered_image = self.apply_filter(selected_filter.matrix)
            self.display_matrix(selected_filter.matrix)
            self.push_image()

    def create_matrix_viewer(self, rows, cols):
        for row in self.cells:
            for cell in row:
                cell.destroy()
        self.cells = []

        for i in range(rows):
            row_cells = []
            for j in range(cols):
                cell = tk.Entry(self.matrix_frame, width=8, justify="center")
                cell.grid(row=i, column=j, padx=2, pady=2)
                cell.bind("<KeyRelease>", self.live_filter_update)
                row_cells.append(cell)
            self.cells.append(row_cells)

    def display_matrix(self, matrix):
        rows, cols = matrix.shape
        self.create_matrix_viewer(rows, cols)

        for i in range(rows):
            for j in range(cols):
                value = round(matrix[i, j], 4)
                self.cells[i][j].insert(0, f"{value}")

        self.update()

    def edit_filter(self):

        try:
            new_matrix = []
            for row in self.cells:
                new_row = []
                for cell in row:
                    value = float(cell.get())
                    new_row.append(value)
                new_matrix.append(new_row)
            new_matrix = np.array(new_matrix)
            available_filters[current_filter_index].matrix = new_matrix
            messagebox.showinfo("Success", "Filter matrix updated successfully!")
            self.filtered_image = self.apply_filter(new_matrix)
            self.push_image()
        except ValueError:
            messagebox.showerror("Error", "Invalid value in matrix!")

    def custom_filter(self):
        try:
            matrix_size = simpledialog.askinteger(
                "Custom Filter", "Enter matrix size:"
            )
            new_matrix = np.zeros((matrix_size, matrix_size))

            new_filter = Filter("Custom", new_matrix)
            available_filters.append(new_filter)
            self.listbox.insert(
                tk.END, new_filter.name
            )  # Add the new filter to the listbox

            self.display_matrix(new_matrix)
        except:
            messagebox.showerror("Error", "Invalid value in matrix!")


if __name__ == "__main__":
    viewer = ImageViewer()
    main_window = MainWindow("ToneMapping GUI", viewer)

    filter_container = FilterContainer(main_window, viewer)
    main_window.add_container(filter_container)

    def open_image_cb():
        if common.open_image(main_window):
            filter_container.notify_image_changed()

    menu = MenuContainer(
        main_window,
        [
            ["Open Image", open_image_cb],
            ["Save Image", lambda: common.save_image(main_window)],
        ],
    )

    main_window.set_menu_container(menu)

    viewer_thread = threading.Thread(target=viewer.run)
    viewer_thread.start()

    main_window.run()
    viewer_thread.join()
