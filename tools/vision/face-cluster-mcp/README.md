# face-cluster-mcp

An MCP-based face analysis service that supports face detection in individual images, face clustering across video frames, and similarity matching between two face images.

## Features

- `face_detect`: Detect all faces in a single image and return bounding boxes, detection scores, gender, and whether an embedding was extracted.
- `face_cluster_video`: Sample video frames at the specified rate, detect faces, and cluster them by embedding to report the number of distinct people and the time intervals in which they appear.
- `face_match`: Compare the primary faces in two images and return a similarity score and a prediction of whether they are the same person.

## Implementation

The service entry point is `server.py`, and the underlying functionality is implemented in `face_utils.py`.

The backend uses a primary method with a fallback:

1. Use `InsightFace` for high-quality face detection and embedding extraction when available.
2. If `InsightFace` is unavailable, automatically fall back to OpenCV Haar Cascade detection and histogram-based feature embeddings.

Clustering uses `AgglomerativeClustering` from `scikit-learn`, and similarity is measured using cosine similarity.

## Dependency Installation

Use Python 3.11.

```bash
bash scripts/install_profiles.sh vision --torch-backend cpu
```

Run this command from the repository root.

Current dependencies include:

- `insightface`
- `onnxruntime`
- `scikit-learn`
- `opencv-python-headless`
- `numpy`

### Model Downloads

On its first run, `FaceAnalysis(name="buffalo_l")` downloads model weights from the [official InsightFace GitHub release](https://github.com/deepinsight/insightface/releases/tag/v0.7). To prepare the cache for offline use in advance, run the following command from the repository root:

```bash
bash scripts/download_models.sh insightface
```

The default installation directory is `~/.insightface/models/buffalo_l/`.

## Starting the Service

Run from the repository root:

```bash
.venvs/vision/bin/python tools/vision/face-cluster-mcp/server.py
```

The service runs as an MCP server and registers the tool definitions in `_tool_defs/face.yaml`.

## Tool Reference

### 1. `face_detect`

Input parameters:

- `image_path`: Absolute path to the image

Returns:

- `total_faces`: Number of detected faces
- `faces`: Structured results for each face, including:
  - `face_id`
  - `bbox`
  - `det_score`
  - `gender`
  - `has_embedding`

Example:

```json
{
  "image_path": "/abs/path/image.jpg"
}
```

### 2. `face_cluster_video`

Input parameters:

- `video_path`: Absolute path to the video
- `start_time`: Start time in seconds; defaults to `0`
- `end_time`: End time in seconds; defaults to the end of the video
- `sample_fps`: Frame sampling rate; defaults to `1.0`
- `distance_threshold`: Clustering distance threshold; defaults to `0.6`

Returns:

- `frames_analyzed`: Number of frames analyzed
- `total_face_detections`: Total number of face detections
- `unique_persons`: Number of distinct people after clustering
- `persons`: Per-person statistics, including first and last appearance timestamps, detection locations, and other details

Example:

```json
{
  "video_path": "/abs/path/video.mp4",
  "start_time": 0,
  "end_time": 30,
  "sample_fps": 1.0,
  "distance_threshold": 0.6
}
```

### 3. `face_match`

Input parameters:

- `image_a`: Absolute path to the first image
- `image_b`: Absolute path to the second image
- `threshold`: Threshold for predicting whether the faces belong to the same person; defaults to `0.4`

Returns:

- `similarity`: Cosine similarity between the primary faces in the two images
- `threshold`: Threshold used
- `is_same_person`: Whether the faces are predicted to belong to the same person

Example:

```json
{
  "image_a": "/abs/path/a.jpg",
  "image_b": "/abs/path/b.jpg",
  "threshold": 0.4
}
```

## Notes

- Input paths must be absolute paths accessible on the local machine.
- By default, `face_match` compares the face with the highest detection score in each image.
- `face_cluster_video` temporarily writes sampled video frames to `/tmp` before running face detection.
- When `InsightFace` is unavailable, the service automatically switches to the OpenCV fallback, with significantly lower accuracy.

## Related Files

- `tools/vision/face-cluster-mcp/server.py`
- `tools/vision/face-cluster-mcp/face_utils.py`
- `_tool_defs/face.yaml`
