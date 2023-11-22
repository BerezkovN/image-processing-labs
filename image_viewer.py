import cv2
from time import sleep


MAXIMUM_WIDTH = 1600
MAXIMUM_HEIGHT = 900
ZOOM_IN_FACTOR = 1.1
ZOOM_OUT_FACTOR = 0.9


class ImageViewer:
    def __init__(self, window_name="Image", image=None):
        self.image = cv2.imread(image) if image else None
        self.roi_center = (
            [self.image.shape[1] // 2, self.image.shape[0] // 2] if image else [0, 0]
        )
        self.scale = 1.0
        self.window_name = window_name
        self.drag_start = None
        self.last_update = 0
        # 50 fps
        self.update_interval = 0.02
        self.dead = False
        self.__update_window_size()

    def set_image(self, image):
        if image is None:
            return
        self.image = image
        if (
            self.roi_center is None
            or self.roi_center[0] > self.image.shape[1]
            or self.roi_center[1] > self.image.shape[0]
        ):
            self.roi_center = (
                [self.image.shape[1] // 2, self.image.shape[0] // 2]
                if self.image is not None
                else [0, 0]
            )

    def run(self):
        while self.image is None:
            sleep(0.1)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.__mouse_events)
        self.__show_image()
        self.__update_window_size()
        try:
            while cv2.getWindowProperty(self.window_name, 0) >= 0 and not self.dead:
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - self.last_update > self.update_interval:
                    self.__show_image()
                    self.last_update = current_time
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
        except:
            pass
        
        self.dead = True

    def width(self):
        return self.image.shape[0]; 

    def height(self):
        return self.image.shape[1];

    def get_image(self):
        return self.image;

    def get_roi(self):
        H, W = self.image.shape[:2]
        roi_W, roi_H = int(W / self.scale), int(H / self.scale)
        roi_W = max(roi_W, 1)
        roi_H = max(roi_H, 1)
        x1 = int(self.roi_center[0] - roi_W / 2)
        y1 = int(self.roi_center[1] - roi_H / 2)
        x2 = int(self.roi_center[0] + roi_W / 2)
        y2 = int(self.roi_center[1] + roi_H / 2)

        # Clamping x1, y1, x2, y2 to ensure they are within the image's boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Adjusting the ROI width and height if they get clamped
        if x2 - x1 < roi_W:
            if x1 == 0:  # ROI is at the left edge
                x2 = x1 + roi_W
            else:  # ROI is at the right edge
                x1 = x2 - roi_W

        if y2 - y1 < roi_H:
            if y1 == 0:  # ROI is at the top edge
                y2 = y1 + roi_H
            else:  # ROI is at the bottom edge
                y1 = y2 - roi_H
        return self.image[y1:y2, x1:x2]

    def __show_image(self):
        roi = self.get_roi()
        cv2.imshow(self.window_name, roi)

    def __update_window_size(self):
        if self.image is None:
            return
        self.aspect_ratio = self.image.shape[0] / self.image.shape[1]
        is_wide = self.image.shape[0] > self.image.shape[1]
        rect = cv2.getWindowImageRect(self.window_name)
        width, height = rect[2:]
        if is_wide:
            if width > MAXIMUM_WIDTH:
                width = MAXIMUM_WIDTH
            cv2.resizeWindow(self.window_name, width, int(width * self.aspect_ratio))
        else:
            if height > MAXIMUM_HEIGHT:
                height = MAXIMUM_HEIGHT
            cv2.resizeWindow(self.window_name, int(height / self.aspect_ratio), height)

    def __mouse_events(self, event, x, y, flags, param):
        self.__update_window_size()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            dx, dy = self.drag_start[0] - x, self.drag_start[1] - y
            self.roi_center[0] += dx
            self.roi_center[1] += dy
            self.drag_start = (x, y)

            H, W = self.image.shape[:2]
            roi_W, roi_H = int(W / self.scale), int(H / self.scale)

            # Clamp the roi_center to ensure it's within valid bounds
            self.roi_center[0] = max(roi_W / 2, min(W - roi_W / 2, self.roi_center[0]))
            self.roi_center[1] = max(roi_H / 2, min(H - roi_H / 2, self.roi_center[1]))

            # Update display only if the sufficient time has passed
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.last_update > self.update_interval:
                self.__show_image()
                self.last_update = current_time

        elif event == cv2.EVENT_MOUSEWHEEL:
            self.scale *= ZOOM_IN_FACTOR if flags > 0 else ZOOM_OUT_FACTOR
            if self.scale < 1.0:
                self.scale = 1.0
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if current_time - self.last_update > self.update_interval:
            self.__show_image()
            self.last_update = current_time
