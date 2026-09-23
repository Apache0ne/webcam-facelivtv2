"""Capture continuously while inference consumes only the latest camera frame."""
import threading
import time


class LatestCamera:
    def __init__(self, opener, stop):
        self.stop = stop
        self.closed = threading.Event()
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.fps = 0.0
        self.captured_at = 0.0
        self.frame_time = 0.0
        self.error = None
        self.thread = threading.Thread(target=self._capture, args=(opener,), daemon=True)
        self.thread.start()

    def _capture(self, opener):
        camera = None
        try:
            camera = opener()
            previous = None
            while not self.closed.is_set() and not self.stop.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Webcam stopped returning frames")
                now = time.perf_counter()
                with self.condition:
                    if previous is not None and now > previous:
                        current = 1 / (now - previous)
                        self.fps = current if self.fps == 0 else self.fps * 0.9 + current * 0.1
                    self.frame = frame
                    self.captured_at = now
                    self.sequence += 1
                    self.condition.notify_all()
                previous = now
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            if camera is not None:
                camera.release()
            self.closed.set()
            with self.condition:
                self.condition.notify_all()

    def read(self):
        with self.condition:
            while self.sequence == self.consumed and not self.error and not self.closed.is_set() and not self.stop.is_set():
                self.condition.wait(timeout=0.1)
            if self.error:
                raise RuntimeError(str(self.error)) from self.error
            if self.closed.is_set() or self.stop.is_set():
                return False, None
            self.consumed = self.sequence
            self.frame_time = self.captured_at
            return True, self.frame

    def release(self):
        self.closed.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Camera driver did not finish closing within five seconds")
