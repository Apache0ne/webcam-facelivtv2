"""Enrollment storage and attendance decisions, independent of camera/UI code."""
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np


def normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("An embedding must contain 512 finite values")
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("Embedding cannot be zero")
    return vector / length


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EnrollmentDB:
    def __init__(self, path, model_sha256):
        self.path, self.model_sha256 = Path(path), model_sha256
        self.people = []
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1 or data.get("model_sha256") != model_sha256:
                raise ValueError("Enrollment database uses a different schema or recognition model")
            self.people = data["people"]
            ids, names = set(), set()
            for person in self.people:
                key = person["name"].casefold()
                if person["id"] in ids or key in names:
                    raise ValueError("Duplicate IDs or names in enrollment database")
                ids.add(person["id"])
                names.add(key)
                normalized(person["embedding"])
        else:
            self.save()

    def save(self):
        atomic_json(self.path, {"schema_version": 1, "model": "facelivtv2_l",
                              "model_sha256": self.model_sha256, "embedding_size": 512,
                              "alignment": "five-point-112", "people": self.people})

    def validate_name(self, name):
        name = name.strip()
        if not name or len(name) > 60 or any(ord(c) < 32 for c in name):
            raise ValueError("Enter a name between 1 and 60 characters")
        if any(p["name"].casefold() == name.casefold() for p in self.people):
            raise ValueError("That name is already enrolled; use a distinct name")
        return name

    def enroll(self, name, samples):
        name = self.validate_name(name)
        if len(samples) < 5:
            raise ValueError("Enrollment requires at least five samples")
        vector = normalized(np.mean([normalized(s) for s in samples], axis=0))
        person = {"id": uuid.uuid4().hex, "name": name,
                  "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  "samples": len(samples), "embedding": vector.tolist()}
        self.people.append(person)
        try:
            self.save()
        except Exception:
            self.people.pop()
            raise
        return person

    def match(self, embedding, threshold=0.55, margin=0.08):
        if not self.people:
            return None, 0.0
        scores = np.array([normalized(p["embedding"]) @ embedding for p in self.people])
        order = np.argsort(-scores)
        best = int(order[0])
        score = float(scores[best])
        runner_up = float(scores[order[1]]) if len(order) > 1 else -1.0
        if score < threshold or score - runner_up < margin:
            return None, score
        return self.people[best], score


def iou(a, b):
    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    area = float(np.prod(np.maximum(0, a[2:] - a[:2])) + np.prod(np.maximum(0, b[2:] - b[:2])))
    return intersection / max(area - intersection, 1e-8)


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, now):
        self.tracks = {key: value for key, value in self.tracks.items() if now - value["seen"] < 1.0}
        available = set(self.tracks)
        result = []
        for box in boxes:
            best = max(available, key=lambda key: iou(box, self.tracks[key]["box"]), default=None)
            if best is None or iou(box, self.tracks[best]["box"]) < 0.3:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"id": best, "candidate": None, "hits": 0, "since": now}
            else:
                available.remove(best)
            track = self.tracks[best]
            track.update(box=np.array(box), seen=now)
            result.append(track)
        for key in available:
            self.tracks[key].update(candidate=None, hits=0)
        return result

    @staticmethod
    def confirm(track, person, now):
        identity = person["id"] if person else None
        if identity is None or track["candidate"] != identity:
            track.update(candidate=identity, hits=1 if identity else 0, since=now)
        else:
            track["hits"] += 1
        return identity is not None and track["hits"] >= 4 and now - track["since"] >= 0.4


class AttendanceLog:
    def __init__(self, directory, model_sha256):
        self.directory, self.model_sha256 = Path(directory), model_sha256
        self.day = None
        self.rows = {}
        self.dirty = False
        self.rollover(datetime.now().astimezone())

    def rollover(self, now):
        day = now.date().isoformat()
        if day == self.day:
            return
        self.flush()
        path = self.directory / f"attendance-{day}.json"
        rows = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1 or data.get("model_sha256") != self.model_sha256:
                raise ValueError("Attendance file uses a different schema or recognition model")
            rows = {row["person_id"]: row for row in data["attendance"]}
        self.day, self.path, self.rows = day, path, rows
        self.dirty = not path.exists()
        self.flush()

    def observe(self, person, score, now):
        self.rollover(now)
        timestamp = now.isoformat(timespec="seconds")
        row = self.rows.setdefault(person["id"], {"person_id": person["id"], "name": person["name"],
                                                 "first_seen": timestamp, "last_seen": timestamp,
                                                 "observations": 0, "best_cosine": float(score)})
        row["last_seen"] = timestamp
        row["observations"] += 1
        row["best_cosine"] = max(row["best_cosine"], float(score))
        self.dirty = True

    def flush(self):
        if self.dirty:
            atomic_json(self.path, {"schema_version": 1, "date": self.day,
                                   "model_sha256": self.model_sha256,
                                   "attendance": list(self.rows.values())})
            self.dirty = False
