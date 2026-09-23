import ctypes as ct
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import threading
import time

import numpy as np

from webcam.database import AttendanceLog, EnrollmentDB, Tracker, normalized
from webcam.detector import SCRFD, TEMPLATE, nms, similarity_transform, Face, align_face, align_faces_cuda
from webcam.native import NativeRuntime
from webcam.capture import LatestCamera

ROOT = Path(__file__).resolve().parents[1]


def basis(index):
    v = np.zeros(512, np.float32)
    v[index] = 1
    return v


class TestEnrollment(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / "build")
        self.path = Path(self.directory.name) / "enrollments.json"
        self.db = EnrollmentDB(self.path, "test-model-hash")

    def tearDown(self):
        self.directory.cleanup()

    def test_enrollment_persists_and_unknown_is_rejected(self):
        person = self.db.enroll("Test Person", [basis(0)] * 8)
        reopened = EnrollmentDB(self.path, "test-model-hash")
        self.assertEqual(reopened.match(basis(0))[0]["id"], person["id"])
        self.assertIsNone(reopened.match(basis(1))[0])
        self.assertEqual(len(reopened.people), 1)

    def test_similar_people_require_margin(self):
        self.db.enroll("Person A", [basis(0)] * 8)
        self.db.enroll("Person B", [normalized(basis(0) + 0.05 * basis(1))] * 8)
        self.assertIsNone(self.db.match(basis(0))[0])

    def test_duplicates_and_bad_samples_do_not_modify_file(self):
        self.db.enroll("Test Person", [basis(0)] * 8)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.db.enroll(" test person ", [basis(0)] * 8)
        with self.assertRaises(ValueError):
            self.db.enroll("New", [np.full(512, np.nan)] * 8)
        with self.assertRaises(ValueError):
            self.db.enroll("New", [basis(1)] * 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_model_mismatch_and_corruption_never_reset_database(self):
        with self.assertRaises(ValueError):
            EnrollmentDB(self.path, "different-model")
        self.path.write_text("broken json", encoding="utf-8")
        with self.assertRaises(ValueError):
            EnrollmentDB(self.path, "test-model-hash")
        self.assertEqual(self.path.read_text(), "broken json")

    def test_daily_deduplication_and_midnight_rollover(self):
        log = AttendanceLog(Path(self.directory.name), "test-model-hash")
        now = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
        person = {"id": "one", "name": "Test Person"}
        log.observe(person, 0.8, now)
        log.observe(person, 0.9, now)
        old_path = log.path
        log.observe(person, 0.85, now + timedelta(seconds=2))
        log.flush()
        old = json.loads(old_path.read_text())
        new = json.loads(log.path.read_text())
        self.assertEqual(len(old["attendance"]), 1)
        self.assertEqual(old["attendance"][0]["observations"], 2)
        self.assertEqual(new["attendance"][0]["observations"], 1)
        self.assertNotEqual(old_path, log.path)


class TestTracking(unittest.TestCase):
    def test_confirmations_reset_on_unknown_and_missing_face(self):
        tracker = Tracker()
        box = np.array([10, 10, 100, 100])
        person = {"id": "one"}
        for i in range(4):
            track = tracker.update([box], i * 0.2)[0]
            confirmed = tracker.confirm(track, person, i * 0.2)
        self.assertTrue(confirmed)
        self.assertFalse(tracker.confirm(track, None, 0.8))
        self.assertFalse(tracker.confirm(track, person, 1.0))
        tracker.update([], 1.1)
        track = tracker.update([box], 1.2)[0]
        self.assertFalse(tracker.confirm(track, person, 1.2))
        self.assertEqual(track["hits"], 1)

    def test_tracks_expire_and_two_faces_do_not_share_track(self):
        tracker = Tracker()
        boxes = [np.array([0, 0, 80, 80]), np.array([90, 0, 170, 80])]
        tracks = tracker.update(boxes, 0)
        ids = {track["id"] for track in tracks}
        self.assertEqual(len(ids), 2)
        later = tracker.update(boxes, 2)
        self.assertFalse(ids & {track["id"] for track in later})


class TestDetection(unittest.TestCase):
    def test_no_cpu_fallback_when_cuda_is_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                SCRFD(ROOT / "models/scrfd_10g_bnkps.onnx")

    def test_alignment_recovers_rotation_translation_and_scale(self):
        angle = 0.25
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        source = TEMPLATE @ rotation.T * 1.7 + [90, 40]
        matrix = similarity_transform(source)
        actual = np.column_stack((source, np.ones(5))) @ matrix.T
        np.testing.assert_allclose(actual, TEMPLATE, atol=1e-4)
        with self.assertRaises(ValueError):
            similarity_transform(np.zeros((5, 2)))

    def test_nms_keeps_distinct_faces(self):
        boxes = np.array([[0, 0, 100, 100], [1, 1, 99, 99], [200, 200, 300, 300]])
        self.assertEqual(nms(boxes, np.array([0.9, 0.8, 0.7])), [0, 2])

    @unittest.skipUnless((ROOT / "models/scrfd_10g_bnkps.onnx").exists(), "SCRFD model absent")
    def test_actual_model_loads_and_blank_frame_has_no_faces(self):
        detector = SCRFD(ROOT / "models/scrfd_10g_bnkps.onnx")
        self.assertEqual(detector.detect(np.zeros((480, 640, 3), np.uint8)), [])
        self.assertEqual(detector.session.get_providers()[0], "CUDAExecutionProvider")
        self.assertEqual(detector.session.get_session_options().get_session_config_entry("session.disable_cpu_ep_fallback"), "1")
        # Constant BGR input verifies GPU RGB conversion, normalization, and padding.
        frame = np.full((480, 640, 3), [10, 80, 200], dtype=np.uint8)
        detector.detect(frame)
        blob = detector.input.cpu().numpy()
        np.testing.assert_allclose(blob[0, :, 0, 0], (np.array([200, 80, 10]) - 127.5) / 128)
        np.testing.assert_allclose(blob[0, :, 500, 0], -127.5 / 128)
        # Compare with an independent CPU execution of the original ONNX graph.
        # OpenCV DNN differs materially on this graph, so it is not our oracle.
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        reference = ort.InferenceSession(str(ROOT / "models/scrfd_10g_bnkps.onnx"),
                                        sess_options=options, providers=["CPUExecutionProvider"])
        outputs = reference.run(None,{reference.get_inputs()[0].name:blob})
        for output in outputs:
            actual = detector.outputs[output.shape].cpu().numpy()
            np.testing.assert_allclose(actual, output, rtol=1e-4, atol=5e-5)

    def test_cuda_alignment_preserves_opencv_crop_contract(self):
        import torch
        source = np.random.default_rng(31).integers(0,256,(480,640,3),dtype=np.uint8)
        landmarks = TEMPLATE * 2.2 + [100,40]
        face = Face(np.array([100,40,340,300]),landmarks,0.9)
        expected = align_face(source,face)
        actual = align_faces_cuda(torch.as_tensor(source,device="cuda"),[face])[0].cpu().numpy()
        difference = np.abs(actual.astype(float)-expected.astype(float))
        self.assertLess(float(difference.mean()),0.2)
        self.assertLessEqual(float(difference.max()),8)


class TestCapture(unittest.TestCase):
    def test_capture_overwrites_stale_frames_and_releases_device(self):
        ready = threading.Event()
        finish = threading.Event()
        stop = threading.Event()

        class FakeCamera:
            index = 0
            released = False

            def read(self):
                self.index += 1
                if self.index == 6:
                    ready.set()
                    finish.wait(timeout=3)
                return True, np.array([self.index])

            def release(self):
                self.released = True

        fake = FakeCamera()
        camera = LatestCamera(lambda: fake, stop)
        try:
            self.assertTrue(ready.wait(timeout=3))
            ok, frame = camera.read()
            self.assertTrue(ok)
            self.assertEqual(frame[0], 5)  # No queue of the four older frames.
        finally:
            stop.set()
            finish.set()
            camera.release()
        self.assertTrue(fake.released)
        self.assertFalse(camera.thread.is_alive())

    def test_camera_errors_reach_consumer(self):
        def fail():
            raise RuntimeError("camera unavailable")
        camera = LatestCamera(fail, threading.Event())
        try:
            with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
                camera.read()
        finally:
            camera.release()


@unittest.skipUnless((ROOT / "build/Release/facelivt_native.dll").exists(), "Native webcam DLL has not been built")
class TestNative(unittest.TestCase):
    def test_dll_matches_cli_and_batch_matches_single(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            face = Path(directory) / "face.bgr"
            output = Path(directory) / "output.f32"
            crop = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
            crop.tofile(face)
            subprocess.run([str(ROOT / "build/Release/facelivt_embed.exe"),
                            str(ROOT / "facelivtv2-l.fp16.flvt"),
                            str(ROOT / "generated/kernels/kernel_manifest.tsv"), str(face), str(output)], check=True)
            expected = np.fromfile(output, dtype=np.float32)
            with NativeRuntime(ROOT) as runtime:
                single = runtime.infer(crop[None])[0]
                batch = runtime.infer(np.stack([crop, crop]))
                np.testing.assert_allclose(single, expected, atol=1e-6)
                np.testing.assert_allclose(batch, np.stack([single, single]), atol=1e-5)
                import torch
                device = runtime.infer(torch.as_tensor(np.stack([crop,crop]),device="cuda"))
                np.testing.assert_allclose(device,batch,atol=1e-6)
                error = ct.create_string_buffer(512)
                target = np.zeros((1, 512), np.float32)
                status = runtime.lib.flvt_infer(runtime.handle, crop.ctypes.data, 1, 1,
                                               target.ctypes.data, target.size, error, len(error))
                self.assertEqual(status, 1)
                self.assertIn(b"expected", error.value)


if __name__ == "__main__":
    unittest.main()
