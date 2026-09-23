"""SCRFD-10G -> aligned crops -> native FaceLiVT -> local JSON attendance."""
import argparse
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from webcam.database import AttendanceLog, EnrollmentDB, Tracker
from webcam.detector import SCRFD, align_faces_cuda
from webcam.native import NativeRuntime
from webcam.capture import LatestCamera
from webcam.video_gpu import GPUVideoDevice, OverlayState

ROOT = Path(__file__).resolve().parent
DETECTOR_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"


def fingerprint(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def make_detector():
    path = ROOT / "models/scrfd_10g_bnkps.onnx"
    if not path.exists() or fingerprint(path) != DETECTOR_SHA256:
        raise RuntimeError("SCRFD-10G model missing or checksum mismatch; see WEBCAM.md")
    return SCRFD(path)


class DataLock:
    def __init__(self, directory):
        import msvcrt
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / "app.lock").open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.file.close()
            raise RuntimeError("An attendance app is already using this database")

    def close(self):
        import msvcrt
        self.file.seek(0)
        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()


def publish(channel, value):
    try:
        channel.put_nowait(value)
    except queue.Full:
        try:
            channel.get_nowait()
        except queue.Empty:
            pass
        channel.put_nowait(value)


def worker(args, commands, results, stop):
    camera = runtime = attendance = lock = None
    try:
        cv2.setNumThreads(4)
        lock = DataLock(args.data)
        sha = fingerprint(ROOT / "facelivtv2-l.fp16.flvt")
        db = EnrollmentDB(args.data / "enrollments.json", sha)
        attendance = AttendanceLog(args.data, sha)
        detector = make_detector()
        runtime = NativeRuntime(ROOT, max_batch=8)
        runtime.infer(np.zeros((1, 112, 112, 3), np.uint8))
        overlay_state = OverlayState()
        camera = LatestCamera(lambda: GPUVideoDevice(ROOT, args.camera, args.gpu_window, overlay_state), stop)
        tracker = Tracker()
        threshold = args.threshold
        enrollment = None
        message = "Enter your name, then click Enroll while you are the only face in view."
        last_tick = time.perf_counter()
        last_flush, last_log = last_tick, last_tick
        fps = 0.0
        logging.info("Camera %s starting; SCRFD-10G CUDA, FaceLiVT native CUDA; enrolled=%s", args.camera, len(db.people))
        while not stop.is_set():
            while True:
                try:
                    command, value = commands.get_nowait()
                except queue.Empty:
                    break
                if command == "enroll":
                    try:
                        name = db.validate_name(value)
                        enrollment = {"name": name, "samples": [], "started": time.perf_counter(), "last": 0.0}
                        message = f"Enrolling {name}: face forward and gently vary your pose."
                    except ValueError as error:
                        message = str(error)
                elif command == "cancel":
                    enrollment = None
                    message = "Enrollment cancelled."
                elif command == "threshold":
                    threshold = float(value)
                    tracker = Tracker()
                    message = f"Match threshold set to {threshold:.2f}."
            ok, frame = camera.read()
            if stop.is_set():
                break
            if not ok or frame is None:
                raise RuntimeError("Webcam stopped returning frames")
            started = time.perf_counter()
            detected = detector.detect(frame)
            faces = detected[:8]
            detector_ms = (time.perf_counter() - started) * 1000
            crops = align_faces_cuda(frame, faces) if faces else None
            crop_preview = None
            inference_ms = 0.0
            embeddings = []
            if faces:
                started = time.perf_counter()
                embeddings = runtime.infer(crops)
                inference_ms = (time.perf_counter() - started) * 1000
                crop_preview = crops[0].cpu().numpy()
            now = time.perf_counter()
            timestamp = datetime.now().astimezone()
            attendance.rollover(timestamp)
            tracks = tracker.update([face.box for face in faces], now)
            if enrollment is not None:
                if now - enrollment["started"] > 30:
                    message, enrollment = "Enrollment timed out. Try again with a clear, well-lit face.", None
                elif len(detected) != 1:
                    enrollment["samples"].clear()
                    message = "Enrollment needs exactly one face in view."
                else:
                    face, crop, embedding = faces[0], crop_preview, embeddings[0]
                    eye_distance = np.linalg.norm(face.landmarks[0] - face.landmarks[1])
                    sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    if min(face.box[2:] - face.box[:2]) < 80 or eye_distance < 24 or sharpness < 25:
                        message = "Move closer and hold still in good light to enroll."
                    elif enrollment["samples"] and float(enrollment["samples"][0] @ embedding) < 0.65:
                        enrollment["samples"].clear()
                        message = "Face changed during enrollment; restarting capture."
                    elif now - enrollment["last"] >= 0.2:
                        enrollment["samples"].append(embedding.copy())
                        enrollment["last"] = now
                        message = f"Capturing {enrollment['name']}: {len(enrollment['samples'])}/8 samples"
                        if len(enrollment["samples"]) == 8:
                            person = db.enroll(enrollment["name"], enrollment["samples"])
                            message = f"Enrolled {person['name']}. Recognition is now active."
                            logging.info("Enrollment saved; total enrolled=%s", len(db.people))
                            enrollment = None
                            tracker = Tracker()
                            tracks = tracker.update([face.box for face in faces], now)
            overlays = []
            for face, embedding, track in zip(faces, embeddings, tracks):
                person, score = db.match(embedding, threshold)
                confirmed = tracker.confirm(track, person, now)
                if confirmed and enrollment is None:
                    attendance.observe(person, score, timestamp)
                label = "Unknown" if person is None else person["name"] + ("" if confirmed else " (checking)")
                color = (0.39, 0.82, 0.35) if confirmed else (0.94, 0.71, 0.16)
                overlays.append({"box": face.box, "landmarks": face.landmarks,
                                 "color": color, "label": f"{label}  {score:.2f}"})
            overlay_state.update(overlays)
            elapsed = now - last_tick
            fps = 1 / elapsed if fps == 0 else fps * 0.85 + 0.15 / elapsed
            last_tick = now
            if now - last_flush >= 2:
                attendance.flush()
                last_flush = now
            if now - last_log >= 10:
                logging.info("Live CUDA: faces=%s enrolled=%s attendance=%s processed_fps=%.1f camera_fps=%.1f detect=%.1fms embed=%.2fms age=%.1fms",
                             len(detected), len(db.people), len(attendance.rows), fps, camera.fps,
                             detector_ms, inference_ms, (now-camera.frame_time)*1000)
                last_log = now
            publish(results, {"crop": crop_preview,
                              "embedding": embeddings[0] if len(embeddings) else None,
                              "people": [p["name"] for p in db.people],
                              "attendance": [dict(row) for row in attendance.rows.values()],
                              "message": message, "enrolling": enrollment is not None,
                              "metrics": f"{fps:.1f} processed FPS | Camera {camera.fps:.1f} FPS | {len(detected)} faces\nSCRFD GPU {detector_ms:.1f} ms | FaceLiVT GPU {inference_ms:.2f} ms | Frame age {(now-camera.frame_time)*1000:.1f} ms\nVideo: GPU surfaces / NVIDIA CUDA-D3D11 preview"})
    except Exception as error:
        logging.exception("Webcam stopped")
        publish(results, {"error": str(error)})
    finally:
        try:
            if camera is not None:
                camera.release()
        except Exception:
            logging.exception("Camera cleanup failed")
        try:
            if runtime is not None:
                runtime.close()
        except Exception:
            logging.exception("Runtime cleanup failed")
        try:
            if attendance is not None:
                attendance.flush()
        except Exception as error:
            logging.exception("Failed to save attendance")
            publish(results, {"error": f"Attendance save failed: {error}"})
        finally:
            if lock is not None:
                lock.close()
        logging.info("Camera released; runtime closed")


class App:
    def __init__(self, args):
        self.root = tk.Tk()
        self.root.title("FaceLiVT | SCRFD-10G Live Attendance")
        self.root.geometry("1120x850")
        self.root.minsize(1000, 760)
        self.root.configure(bg="#101721")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101721")
        style.configure("TLabel", background="#101721", foreground="#e2eaf2", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22))
        style.configure("Muted.TLabel", foreground="#93a8bd")
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Treeview", background="#172230", fieldbackground="#172230", foreground="#e2eaf2", rowheight=28)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Live attendance", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="SCRFD-10G CUDA + five-point alignment + FaceLiVT native CUDA", style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
        body = ttk.Frame(outer)
        body.pack(fill="x")
        left = ttk.Frame(body)
        left.pack(side="left", anchor="n")
        self.preview = tk.Frame(left, bg="#080d14", width=640, height=480)
        self.preview.pack_propagate(False)
        self.preview.pack()
        self.metrics = ttk.Label(left, text="Loading...", style="Muted.TLabel")
        self.metrics.pack(anchor="w", pady=8)
        right = ttk.Frame(body, padding=(20, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(right, text="Enroll a person", font=("Segoe UI Semibold", 15)).pack(anchor="w")
        ttk.Label(right, text="One person in view. Capture 8 clear samples.", style="Muted.TLabel").pack(anchor="w", pady=(4, 8))
        self.name = ttk.Entry(right, font=("Segoe UI", 12))
        self.name.pack(fill="x")
        self.name.bind("<Return>", lambda _: self.enroll())
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        self.enroll_button = ttk.Button(buttons, text="Enroll", command=self.enroll, state="disabled")
        self.enroll_button.pack(side="left")
        ttk.Button(buttons, text="Cancel", command=lambda: self.commands.put(("cancel", None))).pack(side="left", padx=8)
        self.status = ttk.Label(right, text="Starting...", wraplength=340)
        self.status.pack(anchor="w", pady=(0, 10))
        self.crop = tk.Label(right, bg="#172230")
        self.crop.pack(anchor="w")
        self.crop_image = ImageTk.PhotoImage(Image.new("RGB", (112, 112), "#172230"))
        self.crop.configure(image=self.crop_image)
        ttk.Label(right, text="Aligned 112 x 112 model input", style="Muted.TLabel").pack(anchor="w")
        self.signature = tk.Canvas(right, height=55, width=320, bg="#172230", highlightthickness=0)
        self.signature.pack(anchor="w", pady=(8, 0))
        ttk.Label(right, text="Embedding activity: first 64 of 512 values", style="Muted.TLabel").pack(anchor="w")
        self.enrolled = ttk.Label(right, text="Enrolled: none", wraplength=340, style="Muted.TLabel")
        self.enrolled.pack(anchor="w", pady=10)
        options = ttk.Frame(right)
        options.pack(anchor="w")
        ttk.Label(options, text="Match threshold").pack(side="left")
        self.threshold = tk.StringVar(value=f"{args.threshold:.2f}")
        ttk.Spinbox(options, from_=0.1, to=0.99, increment=0.01, textvariable=self.threshold, width=5).pack(side="left", padx=6)
        ttk.Button(options, text="Apply", command=self.set_threshold).pack(side="left")
        ttk.Label(right, text="Trial threshold; accuracy is not yet calibrated.", style="Muted.TLabel").pack(anchor="w", pady=5)
        ttk.Label(outer, text="Today's attendance", font=("Segoe UI Semibold", 15)).pack(anchor="w", pady=(8, 6))
        table_frame = ttk.Frame(outer)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=("name", "first", "last", "state", "score"), show="headings", height=5)
        for column, label, width in (("name", "Person", 240), ("first", "First seen", 160), ("last", "Last seen", 160), ("state", "Status", 180), ("score", "Best similarity", 140)):
            self.table.heading(column, text=label)
            self.table.column(column, width=width)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        ttk.Label(outer, text="Local processing. JSON stores enrollments and attendance; webcam images are not saved.", style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        self.commands, self.results, self.stop = queue.Queue(), queue.Queue(maxsize=1), threading.Event()
        self.root.update_idletasks()
        args.gpu_window = self.preview.winfo_id()
        self.thread = threading.Thread(target=worker, args=(args, self.commands, self.results, self.stop), daemon=True)
        self.thread.start()
        self.closing = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(30, self.poll)

    def enroll(self):
        if not self.name.get().strip():
            self.status.configure(text="Enter a name first.")
            return
        self.commands.put(("enroll", self.name.get()))

    def set_threshold(self):
        try:
            value = float(self.threshold.get())
            if not 0.1 <= value <= 0.99:
                raise ValueError
            self.commands.put(("threshold", value))
        except ValueError:
            self.status.configure(text="Threshold must be between 0.10 and 0.99.")

    def poll(self):
        if self.closing:
            return
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            self.root.after(30, self.poll)
            return
        if "error" in result:
            self.status.configure(text=result["error"])
            self.metrics.configure(text="Stopped")
            self.enroll_button.configure(state="disabled")
            messagebox.showerror("Webcam attendance", result["error"], parent=self.root)
            return
        self.metrics.configure(text=result["metrics"])
        self.status.configure(text=result["message"])
        self.enroll_button.configure(state="disabled" if result["enrolling"] else "normal")
        crop = result["crop"] if result["crop"] is not None else np.zeros((112, 112, 3), np.uint8)
        self.crop_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        self.crop.configure(image=self.crop_image)
        self.signature.delete("all")
        self.signature.create_line(0, 27, 320, 27, fill="#53667a")
        if result["embedding"] is not None:
            for index, value in enumerate(result["embedding"][:64]):
                y = 27 - float(np.clip(value, -0.2, 0.2)) * 120
                self.signature.create_line(index*5+2, 27, index*5+2, y, fill="#54d6bd", width=3)
        self.enrolled.configure(text="Enrolled: " + (", ".join(result["people"]) or "none"))
        now = datetime.now().astimezone()
        rows = {row["person_id"]: row for row in result["attendance"]}
        for item in self.table.get_children():
            if item not in rows:
                self.table.delete(item)
        for identity, row in rows.items():
            first, last = datetime.fromisoformat(row["first_seen"]), datetime.fromisoformat(row["last_seen"])
            state = "In view" if (now - last).total_seconds() <= 2 else "Seen today"
            values = (row["name"], first.strftime("%H:%M:%S"), last.strftime("%H:%M:%S"), state, f"{row['best_cosine']:.3f}")
            if self.table.exists(identity):
                self.table.item(identity, values=values)
            else:
                self.table.insert("", "end", iid=identity, values=values)
        self.root.after(30, self.poll)

    def close(self):
        self.closing = True
        self.stop.set()
        self.status.configure(text="Releasing camera and saving attendance...")
        self.enroll_button.configure(state="disabled")
        self.wait_closed()

    def wait_closed(self):
        if self.thread.is_alive():
            self.root.after(100, self.wait_closed)
        else:
            self.root.destroy()


def video_self_test():
    """Draw CUDA-generated pixels without opening a camera or enrollment DB."""
    import torch
    root = tk.Tk()
    root.title("NVIDIA CUDA / Direct3D 11 preview test - no camera")
    panel = tk.Frame(root,width=640,height=480,bg="black")
    panel.pack()
    root.update_idletasks()
    video = GPUVideoDevice(ROOT,0,panel.winfo_id(),preview_only=True)
    frame = torch.zeros((480,640,4),dtype=torch.uint8,device="cuda:0")
    frame[:,:,0] = torch.arange(640,device="cuda:0").remainder(256).to(torch.uint8)
    frame[:,:,1] = torch.arange(480,device="cuda:0")[:,None].remainder(256).to(torch.uint8)
    frame[:,:,2] = 80
    frame[:,:,3] = 255
    def render():
        video.test_frame(frame)
        root.after(33,render)
    def close():
        video.release()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW",close)
    root.after(10,render)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--video-self-test", action="store_true", help="Show a CUDA-generated gradient through D3D11, without opening the camera")
    args = parser.parse_args()
    if not 0.1 <= args.threshold <= 0.99:
        parser.error("threshold must be between 0.10 and 0.99")
    if args.video_self_test:
        video_self_test()
        return
    args.data.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.data / "webcam.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    app = App(args)
    app.root.mainloop()


if __name__ == "__main__":
    main()
