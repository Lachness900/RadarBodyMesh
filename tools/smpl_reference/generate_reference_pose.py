"""Export the approved mmYoga poses as local SMPL reference assets.

The T-pose uses the canonical template directly. The remaining presets are
generated with the official ``smplx`` linear blend skinning implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pose_presets import POSE_LABELS, get_pose_axis_angles, get_pose_translation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "OneDrive" / "SMPL_model"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "smpl_reference"


class _SparseMatrixPlaceholder:
    """Load a SciPy sparse payload without importing SciPy.

    The provided SMPL pickle stores ``J_regressor`` as a SciPy CSC matrix.
    It is restored to a dense NumPy matrix only when a posed mesh needs it.
    """

    def __new__(cls, *_args: Any, **_kwargs: Any) -> "_SparseMatrixPlaceholder":
        return super().__new__(cls)

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _TrustedSmplUnpickler(pickle.Unpickler):
    """Restrict pickle globals to the NumPy and sparse types used by SMPL."""

    _NUMPY_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.numeric", "_frombuffer"),
    }
    _SPARSE_GLOBALS = {
        ("scipy.sparse._csc", "csc_matrix"),
        ("scipy.sparse._csr", "csr_matrix"),
        ("scipy.sparse.csc", "csc_matrix"),
        ("scipy.sparse.csr", "csr_matrix"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self._SPARSE_GLOBALS:
            return _SparseMatrixPlaceholder
        if (module, name) in self._NUMPY_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Unsupported global in SMPL pickle: {module}.{name}"
        )


def load_smpl_data(model_path: Path) -> dict[str, Any]:
    """Load one trusted SMPL data dictionary without importing SciPy."""

    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL model not found: {model_path}")

    with model_path.open("rb") as source:
        model = _TrustedSmplUnpickler(source, encoding="latin1").load()

    if not isinstance(model, dict):
        raise ValueError("Expected the SMPL pickle root to be a dictionary")
    required = {
        "J_regressor",
        "f",
        "kintree_table",
        "posedirs",
        "shapedirs",
        "v_template",
        "weights",
    }
    missing = required.difference(model)
    if missing:
        raise ValueError(f"SMPL pickle is missing: {', '.join(sorted(missing))}")
    return model


def _sparse_placeholder_to_dense(
    sparse: _SparseMatrixPlaceholder,
) -> NDArray[np.float64]:
    """Restore the inspected CSC/CSR payload as a regular NumPy matrix."""

    state = getattr(sparse, "state", None)
    if not isinstance(state, dict):
        raise ValueError("SMPL sparse matrix state is missing")

    matrix_format = state.get("format")
    shape = tuple(state.get("_shape", ()))
    indptr = np.asarray(state.get("indptr"), dtype=np.int64)
    indices = np.asarray(state.get("indices"), dtype=np.int64)
    values = np.asarray(state.get("data"), dtype=np.float64)
    if len(shape) != 2 or len(indices) != len(values):
        raise ValueError("SMPL sparse matrix state is invalid")

    dense = np.zeros(shape, dtype=np.float64)
    if matrix_format == "csc":
        for column in range(shape[1]):
            start, end = int(indptr[column]), int(indptr[column + 1])
            dense[indices[start:end], column] = values[start:end]
        return dense
    if matrix_format == "csr":
        for row in range(shape[0]):
            start, end = int(indptr[row]), int(indptr[row + 1])
            dense[row, indices[start:end]] = values[start:end]
        return dense
    raise ValueError(f"Unsupported SMPL sparse matrix format: {matrix_format}")


def generate_pose_mesh(
    model: dict[str, Any],
    pose_label: str,
) -> tuple[NDArray[np.float64], NDArray[np.uint32]]:
    """Generate one mesh, using official smplx LBS for non-zero poses."""

    vertices = np.asarray(model["v_template"], dtype=np.float64)
    faces = np.asarray(model["f"], dtype=np.uint32)

    if pose_label != "t_pose":
        try:
            import torch
            from smplx.lbs import lbs
        except ImportError as error:
            raise RuntimeError(
                "Non-T presets require torch and smplx. "
                "Install the project requirements first."
            ) from error

        joint_regressor = model["J_regressor"]
        if isinstance(joint_regressor, _SparseMatrixPlaceholder):
            joint_regressor = _sparse_placeholder_to_dense(joint_regressor)
        joint_regressor = np.asarray(joint_regressor, dtype=np.float32)

        shapedirs = np.asarray(model["shapedirs"], dtype=np.float32)
        shapedirs = shapedirs[:, :, :10]
        posedirs = np.asarray(model["posedirs"], dtype=np.float32)
        posedirs = posedirs.reshape((-1, posedirs.shape[-1])).T
        parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
        parents[0] = -1

        skinning_weights = np.asarray(model["weights"], dtype=np.float32)
        full_pose = get_pose_axis_angles(pose_label).reshape((1, -1))
        with torch.no_grad():
            posed_vertices, _joints = lbs(
                betas=torch.zeros((1, shapedirs.shape[-1]), dtype=torch.float32),
                pose=torch.as_tensor(full_pose, dtype=torch.float32),
                v_template=torch.as_tensor(vertices, dtype=torch.float32),
                shapedirs=torch.as_tensor(shapedirs, dtype=torch.float32),
                posedirs=torch.as_tensor(posedirs, dtype=torch.float32),
                J_regressor=torch.as_tensor(joint_regressor, dtype=torch.float32),
                parents=torch.as_tensor(parents, dtype=torch.long),
                lbs_weights=torch.as_tensor(skinning_weights),
                pose2rot=True,
            )
        vertices = posed_vertices[0].cpu().numpy().astype(np.float64)
        vertices = vertices + get_pose_translation(pose_label)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected vertices shaped (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces shaped (M, 3), got {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("SMPL template contains non-finite vertices")
    if faces.size == 0 or int(faces.max()) >= len(vertices):
        raise ValueError("SMPL faces reference an invalid vertex index")

    return vertices, faces


def compute_vertex_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Compute smooth per-vertex normals from the triangle topology."""

    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )

    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths > 0, lengths, 1.0)
    return normals.astype("<f4")


def _append_aligned(buffer: bytearray, payload: bytes) -> tuple[int, int]:
    """Append binary data with glTF's required four-byte alignment."""

    offset = len(buffer)
    buffer.extend(payload)
    payload_length = len(payload)
    buffer.extend(b"\x00" * ((-len(buffer)) % 4))
    return offset, payload_length


def write_glb(
    output_path: Path,
    vertices: NDArray[np.float64],
    faces: NDArray[np.uint32],
    source_model_name: str,
    pose_label: str,
) -> None:
    """Write a dependency-free glTF 2.0 binary containing the static mesh."""

    positions = np.asarray(vertices, dtype="<f4")
    normals = compute_vertex_normals(vertices, faces)

    max_index = int(faces.max())
    if max_index <= np.iinfo(np.uint16).max:
        indices = np.asarray(faces.reshape(-1), dtype="<u2")
        index_component_type = 5123
    else:
        indices = np.asarray(faces.reshape(-1), dtype="<u4")
        index_component_type = 5125

    binary = bytearray()
    position_offset, position_length = _append_aligned(binary, positions.tobytes())
    normal_offset, normal_length = _append_aligned(binary, normals.tobytes())
    index_offset, index_length = _append_aligned(binary, indices.tobytes())

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "mmYoga SMPL reference pose exporter",
        },
        "scene": 0,
        "scenes": [{"name": "SMPL Reference Pose", "nodes": [0]}],
        "nodes": [
            {
                "name": f"{pose_label} Reference",
                "mesh": 0,
                "extras": {
                    "poseLabel": pose_label,
                    "referenceOnly": True,
                    "sourceModel": source_model_name,
                },
            }
        ],
        "meshes": [
            {
                "name": "SMPL Template Mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "Reference Pose Material",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.10, 0.64, 0.66, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.78,
                },
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": position_length,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": normal_offset,
                "byteLength": normal_length,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_length,
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": index_component_type,
                "count": int(indices.size),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
    }

    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary_chunk = bytes(binary)

    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total_length))
        output.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        output.write(json_chunk)
        output.write(struct.pack("<I4s", len(binary_chunk), b"BIN\x00"))
        output.write(binary_chunk)


def validate_glb(output_path: Path) -> dict[str, Any]:
    """Validate the generated GLB container and return its JSON document."""

    payload = output_path.read_bytes()
    if len(payload) < 20:
        raise ValueError("Generated GLB is too small")

    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or total_length != len(payload):
        raise ValueError("Generated GLB header is invalid")

    json_length, json_type = struct.unpack_from("<I4s", payload, 12)
    if json_type != b"JSON":
        raise ValueError("Generated GLB is missing its JSON chunk")

    json_start = 20
    json_end = json_start + json_length
    document = json.loads(payload[json_start:json_end].decode("utf-8"))

    binary_length, binary_type = struct.unpack_from("<I4s", payload, json_end)
    if binary_type != b"BIN\x00":
        raise ValueError("Generated GLB is missing its binary chunk")
    if json_end + 8 + binary_length != len(payload):
        raise ValueError("Generated GLB chunk lengths are inconsistent")

    return document


def render_preview(
    preview_path: Path,
    vertices: NDArray[np.float64],
    faces: NDArray[np.uint32],
    pose_label: str,
) -> None:
    """Render front and side views for human approval of the reference pose."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Matplotlib expects Z to be vertical. SMPL uses Y-up, so plot X/Z/Y.
    display_vertices = vertices[:, [0, 2, 1]]
    triangles = display_vertices[faces]
    mins = display_vertices.min(axis=0)
    maxs = display_vertices.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    centers = (mins + maxs) / 2

    fig = plt.figure(figsize=(9, 6), facecolor="#f4f7f8")
    for index, (title, azimuth) in enumerate(
        (("Front", -90), ("Side", 0)),
        start=1,
    ):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        mesh = Poly3DCollection(
            triangles,
            facecolor="#1aa3a8",
            edgecolor="none",
            alpha=1.0,
        )
        axis.add_collection3d(mesh)
        axis.set_xlim(centers[0] - spans[0] / 2, centers[0] + spans[0] / 2)
        axis.set_ylim(centers[1] - spans[1] / 2, centers[1] + spans[1] / 2)
        axis.set_zlim(centers[2] - spans[2] / 2, centers[2] + spans[2] / 2)
        axis.set_box_aspect(spans)
        axis.view_init(elev=0, azim=azimuth)
        axis.set_title(title, fontsize=13, color="#34484b", pad=10)
        axis.set_axis_off()

    title = pose_label.replace("_", " ").title()
    fig.suptitle(f"SMPL {title} Reference", fontsize=16, color="#172027")
    fig.tight_layout()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def file_sha256(path: Path) -> str:
    """Return a reproducible content hash for a local source or output file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    report_path: Path,
    model_path: Path,
    output_path: Path,
    vertices: NDArray[np.float64],
    faces: NDArray[np.uint32],
    pose_label: str,
) -> None:
    """Record geometry checks without exposing the ignored source model path."""

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    report = {
        "pose_label": pose_label,
        "reference_only": True,
        "source_model": model_path.name,
        "source_sha256": file_sha256(model_path),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "bounds_min_xyz_m": minimum.tolist(),
        "bounds_max_xyz_m": maximum.tolist(),
        "dimensions_xyz_m": (maximum - minimum).tolist(),
        "center_xyz_m": ((minimum + maximum) / 2).tolist(),
        "coordinate_system": "SMPL native coordinates: X horizontal, Y up, Z depth",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one local reference export."""

    parser = argparse.ArgumentParser(
        description="Export reviewed mmYoga SMPL reference poses as GLB files."
    )
    parser.add_argument(
        "--model",
        choices=("male", "female"),
        default="male",
        help="Select smpl_m.pkl or smpl_f.pkl from OneDrive/SMPL_model.",
    )
    parser.add_argument(
        "--pose",
        choices=POSE_LABELS,
        default="t_pose",
        help="Select one reviewed project pose.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all reviewed reference poses.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override the source pickle path. Use only a trusted SMPL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the generated GLB path when exporting one pose.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the output directory, especially with --all.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Do not render the PNG front/side preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate, validate, preview, and report the requested reference poses."""

    args = parse_args()
    if args.all and args.output is not None:
        raise ValueError("--output cannot be combined with --all; use --output-dir")

    suffix = "m" if args.model == "male" else "f"
    model_path = (
        args.model_path
        if args.model_path is not None
        else DEFAULT_MODEL_DIR / f"smpl_{suffix}.pkl"
    )
    pose_labels = POSE_LABELS if args.all else (args.pose,)
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    model = load_smpl_data(model_path)

    for pose_label in pose_labels:
        output_path = (
            args.output
            if args.output is not None
            else output_dir / f"{pose_label}_{args.model}.glb"
        )
        # Keep review-only artefacts out of frontend/public. Vite only needs
        # the GLB at runtime.
        preview_path = DEFAULT_OUTPUT_DIR / f"{output_path.stem}.png"
        report_path = DEFAULT_OUTPUT_DIR / f"{output_path.stem}.json"

        vertices, faces = generate_pose_mesh(model, pose_label)
        write_glb(output_path, vertices, faces, model_path.name, pose_label)
        validate_glb(output_path)
        if not args.skip_preview:
            render_preview(preview_path, vertices, faces, pose_label)
        write_report(
            report_path,
            model_path,
            output_path,
            vertices,
            faces,
            pose_label,
        )

        print(f"Generated: {output_path}")
        if not args.skip_preview:
            print(f"Preview:   {preview_path}")
        print(f"Report:    {report_path}")
        print(f"Geometry:  {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    main()
