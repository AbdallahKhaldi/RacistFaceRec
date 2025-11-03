import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

try:
    import face_recognition  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency warning
    raise SystemExit(
        "face_recognition package is required. Install with `pip install face_recognition`."
    ) from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency warning
    raise SystemExit(
        "Pillow package is required. Install with `pip install pillow`."
    ) from exc


class GroupClassifier:
    """Machine learning classifier to predict group membership for unknown faces."""

    def __init__(self):
        self.classifier = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.min_samples_per_group = 2

    def train(self, encodings: List[np.ndarray], groups: List[str]) -> bool:
        """Train the classifier on known face encodings and their groups."""
        if len(encodings) < self.min_samples_per_group * 2:
            # Need at least 2 samples per group to train
            self.is_trained = False
            return False

        try:
            X = np.array(encodings)
            y = self.label_encoder.fit_transform(groups)

            # Train the classifier
            self.classifier.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Training error: {e}")
            self.is_trained = False
            return False

    def predict(self, encoding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Predict which group an unknown face belongs to with confidence."""
        if not self.is_trained:
            return None

        try:
            # Get prediction probabilities
            proba = self.classifier.predict_proba([encoding])[0]
            predicted_class = np.argmax(proba)
            confidence = float(proba[predicted_class])
            predicted_group = self.label_encoder.inverse_transform([predicted_class])[0]

            return predicted_group, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


class FaceDatabase:
    """Persisted storage for known face encodings."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._storage_path_parent = self.storage_path.parent
        self._entries: List[Dict[str, np.ndarray]] = []
        self._storage_path_parent.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def entries(self) -> List[Dict[str, np.ndarray]]:
        return self._entries

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._entries = []
            return
        with self.storage_path.open("r", encoding="utf-8") as infile:
            raw_entries = json.load(infile)
        self._entries = [
            {
                "name": item["name"],
                "encoding": np.array(item["encoding"]),
                "group": item.get("group", "A")  # Default to group A for backward compatibility
            }
            for item in raw_entries
        ]

    def save(self) -> None:
        payload = [
            {
                "name": entry["name"],
                "encoding": entry["encoding"].tolist(),
                "group": entry.get("group", "A")
            }
            for entry in self._entries
        ]
        with self.storage_path.open("w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, indent=2)

    def add_encoding(self, name: str, encoding: np.ndarray, group: str = "A") -> None:
        self._entries.append({"name": name, "encoding": encoding, "group": group})
        self.save()

    def clear(self) -> None:
        self._entries.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()


BASE_DIR = Path(__file__).resolve().parent.parent
ENCODINGS_PATH = BASE_DIR / "data" / "encodings.json"

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "frontend"),
    static_url_path="",
)

database = FaceDatabase(ENCODINGS_PATH)
group_classifier = GroupClassifier()


def train_group_classifier() -> bool:
    """Train the classifier on current database entries."""
    if len(database.entries) < 4:  # Need at least 2 from each group
        return False

    encodings = [entry["encoding"] for entry in database.entries]
    groups = [entry.get("group", "A") for entry in database.entries]

    # Check if we have both groups represented
    unique_groups = set(groups)
    if len(unique_groups) < 2:
        return False

    return group_classifier.train(encodings, groups)


def decode_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL into an RGB numpy array."""
    if "," not in data_url:
        raise ValueError("Invalid image payload.")
    _, base64_data = data_url.split(",", 1)
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def extract_face_encoding(image_array: np.ndarray) -> np.ndarray:
    """Extract the first face encoding from the provided image array."""
    face_locations = face_recognition.face_locations(image_array, model="hog")
    if not face_locations:
        raise ValueError("No face detected in the provided image.")
    encodings = face_recognition.face_encodings(image_array, face_locations)
    if not encodings:
        raise ValueError("Unable to extract face encoding.")
    if len(encodings) > 1:
        raise ValueError("Multiple faces detected; please provide a single face.")
    return encodings[0]


def match_encoding(encoding: np.ndarray, tolerance: float = 0.45) -> Optional[Tuple[str, float, str]]:
    """Attempt to match the provided encoding against known faces."""
    if not database.entries:
        return None
    known_encodings = [entry["encoding"] for entry in database.entries]
    known_names = [entry["name"] for entry in database.entries]
    known_groups = [entry.get("group", "A") for entry in database.entries]
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_index = int(np.argmin(distances))
    if distances[best_index] <= tolerance:
        return known_names[best_index], float(distances[best_index]), known_groups[best_index]
    return None


@app.route("/")
def index() -> str:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/register", methods=["POST"])
def register_face():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload."}), 400
    name = payload.get("name")
    image_data = payload.get("image")
    group = payload.get("group", "A")  # Default to group A

    if not name or not image_data:
        return jsonify({"error": "Both 'name' and 'image' fields are required."}), 400

    # Validate group
    if group not in ["A", "J"]:
        return jsonify({"error": "Group must be either 'A' or 'J'."}), 400

    try:
        image_array = decode_image(image_data)
        encoding = extract_face_encoding(image_array)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    database.add_encoding(name.strip(), encoding, group)

    # Retrain classifier with new data
    classifier_trained = train_group_classifier()

    return jsonify({
        "status": "registered",
        "name": name.strip(),
        "group": group,
        "classifier_trained": classifier_trained
    }), 201


@app.route("/api/register_batch", methods=["POST"])
def register_batch():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload."}), 400

    faces = payload.get("faces", [])
    if not faces:
        return jsonify({"error": "Field 'faces' with array of face data is required."}), 400

    results = []
    for idx, face_data in enumerate(faces):
        name = face_data.get("name")
        image_data = face_data.get("image")
        group = face_data.get("group", "A")

        if not name or not image_data:
            results.append({
                "index": idx,
                "name": name,
                "status": "failed",
                "error": "Missing name or image data"
            })
            continue

        if group not in ["A", "J"]:
            results.append({
                "index": idx,
                "name": name,
                "status": "failed",
                "error": "Group must be either 'A' or 'J'"
            })
            continue

        try:
            image_array = decode_image(image_data)
            encoding = extract_face_encoding(image_array)
            database.add_encoding(name.strip(), encoding, group)
            results.append({
                "index": idx,
                "name": name.strip(),
                "group": group,
                "status": "registered"
            })
        except ValueError as exc:
            results.append({
                "index": idx,
                "name": name,
                "status": "failed",
                "error": str(exc)
            })

    successful = sum(1 for r in results if r["status"] == "registered")
    failed = len(results) - successful

    # Retrain classifier after batch upload
    classifier_trained = train_group_classifier()

    return jsonify({
        "status": "completed",
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "classifier_trained": classifier_trained,
        "results": results
    }), 200


@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload."}), 400
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "Field 'image' is required."}), 400
    try:
        image_array = decode_image(image_data)
        encoding = extract_face_encoding(image_array)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # First try exact match
    match = match_encoding(encoding)
    if match:
        name, distance, group = match
        return jsonify({
            "status": "recognized",
            "name": name,
            "distance": distance,
            "group": group,
            "match_type": "exact"
        }), 200

    # If no exact match, try ML prediction
    prediction = group_classifier.predict(encoding)
    if prediction:
        predicted_group, confidence = prediction
        return jsonify({
            "status": "predicted",
            "group": predicted_group,
            "confidence": confidence,
            "match_type": "predicted",
            "message": f"Unknown face - predicted as Group {predicted_group}"
        }), 200

    # Neither exact match nor prediction possible
    return jsonify({
        "status": "unknown",
        "message": "Face not recognized and not enough data to predict group"
    }), 200


@app.route("/api/faces", methods=["GET"])
def list_faces():
    faces_with_groups = [
        {"name": entry["name"], "group": entry.get("group", "A")}
        for entry in database.entries
    ]
    # Sort by name
    faces_with_groups.sort(key=lambda x: x["name"])
    return jsonify({"faces": faces_with_groups})


@app.route("/api/faces", methods=["DELETE"])
def clear_faces():
    database.clear()
    return jsonify({"status": "cleared"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
