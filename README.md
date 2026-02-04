# Face Detection Utility

A Node.js-based face detection and recognition system that identifies unique individuals across multiple images using deep learning and facial recognition algorithms.

## Overview

This utility processes images to:
- **Detect faces** in images using TensorFlow.js and face-api.js
- **Extract facial descriptors** for each detected face
- **Cluster faces** by identity to identify unique individuals
- **Handle duplicates** and overlapping detections intelligently
- **Generate reports** with counts of unique individuals per image

## Features

✨ **Key Capabilities:**
- Multi-face detection in a single image
- Facial feature extraction using 68-point landmarks
- Face recognition using deep learning descriptors
- Intelligent duplicate detection based on spatial overlap
- Single-linkage clustering for face grouping
- Configurable similarity thresholds
- Detailed distance matrices between detected faces
- Batch processing of multiple images

## How It Works

### 1. **Face Detection**
The utility uses the TinyFaceDetector model to locate faces in images with high accuracy even at different scales and angles.

### 2. **Feature Extraction**
For each detected face, the system extracts:
- 68-point facial landmarks
- High-dimensional face descriptors (128-dimensional vectors)

### 3. **Duplicate Filtering**
Overlapping detections in the same spatial region (IoU > 0.3) are filtered as duplicates to avoid counting the same face twice.

### 4. **Face Clustering**
Faces are grouped into clusters using single-linkage clustering based on Euclidean distance between descriptors:
- Faces with distance < 0.55 are considered the same individual
- Distance matrix shows similarity between all detected faces
- Configurable threshold for fine-tuning sensitivity

### 5. **Results Generation**
The system outputs:
- Number of unique individuals per image
- Total count across all processed images
- Debug information (distance matrices, detection counts)

## Installation

### Prerequisites
- Node.js 14+ 
- npm (Node Package Manager)
- macOS, Linux, or Windows with native build tools

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhijit0404/facedetection.git
   cd facedetection
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

   This installs:
   - `@tensorflow/tfjs` - Core TensorFlow.js library
   - `@tensorflow/tfjs-backend-cpu` - CPU backend for inference
   - `@vladmandic/face-api` - Pre-trained face detection and recognition models
   - `canvas` - Image processing capabilities
   - Additional dependencies for model loading and processing

3. **Prepare input images:**
   - Place your images in the `inputimages/` directory
   - Supported formats: `.jpg`, `.jpeg`, `.png`

## Usage

### Basic Usage

Run the face detection on all images in the input directory:

```bash
node faceDetection.js
```

### Example Workflow

1. **Add images:**
   ```bash
   # Copy images to the input directory
   cp ~/Pictures/photo1.jpg inputimages/
   cp ~/Pictures/photo2.jpg inputimages/
   ```

2. **Run detection:**
   ```bash
   node faceDetection.js
   ```

3. **View results:**
   The utility will output:
   ```
   Loading face detection models...
   ✓ Models loaded successfully

   Found 2 image(s)

   Processing: photo1.jpg
     ✓ Image loaded (800x600)
     Running face detection...
     → Found 3 detection(s)
     Filtered to 3 unique detections
     Distance matrix:
       Face 0 <-> Face 1: 0.5234
       Face 0 <-> Face 2: 0.7891
       Face 1 <-> Face 2: 0.6543
     → 2 unique individual(s)

   Processing: photo2.jpg
     ✓ Image loaded (1024x768)
     Running face detection...
     → Found 2 detection(s)
     → 2 unique individual(s)

   ========== RESULTS ==========

   photo1.jpg: 2 individual(s)
   photo2.jpg: 2 individual(s)

   Total: 4 individual(s)
   ```

### Output Explanation

- **Image loaded (WxH)**: Image dimensions in pixels
- **Found N detection(s)**: Total faces detected (including potential duplicates)
- **Filtered to N unique detections**: After removing spatial duplicates
- **Distance matrix**: Shows similarity scores between all faces (lower = more similar)
- **N unique individual(s)**: Count of distinct people identified using clustering

## Configuration

### Adjusting Face Clustering Threshold

Edit the `distanceThreshold` in `faceDetection.js` (line ~148):

```javascript
// Current: 0.55 (stricter - fewer people grouped together)
let distanceThreshold = 0.55;

// More lenient (0.65+): Groups more similar faces together
// More strict (0.45-): Only groups nearly identical faces
```

**Guidelines:**
- `0.45-0.50`: Very strict - only accepts nearly identical faces
- `0.55-0.60`: Balanced - recommended for general use
- `0.65-0.75`: Lenient - may group different people if similar

### Adjusting Face Detection Sensitivity

Edit the `scoreThreshold` in `faceDetection.js` (line ~128):

```javascript
scoreThreshold: 0.05  // Current: Very lenient, detects more faces
```

**Guidelines:**
- `0.05-0.10`: Detect more faces (may include false positives)
- `0.15-0.25`: Balanced detection
- `0.30+`: Conservative (may miss some faces)

## Project Structure

```
FaceDetection/
├── faceDetection.js          # Main detection script
├── package.json              # Dependencies and project metadata
├── README.md                 # This file
├── execution.log             # Sample output from last run
├── inputimages/              # Input directory for images
│   ├── TestCase1.jpeg
│   ├── TestCase2.jpeg
│   ├── TestCase3.jpeg
│   └── TestCase4.jpeg
└── node_modules/             # Installed dependencies
```

## Test Cases

The repository includes 4 test images with varying complexity:

| Test Case | Description | Expected Output |
|-----------|-------------|-----------------|
| TestCase1.jpeg | Single person | 1 unique individual |
| TestCase2.jpeg | Two different people | 2 unique individuals |
| TestCase3.jpeg | Three different people | 3 unique individuals |
| TestCase4.jpeg | Multiple shots of same person | 1 unique individual |

Run the utility to verify it works correctly on these test cases.

## Technical Details

### Models Used

- **TinyFaceDetector**: Fast, lightweight face detection model
- **FaceLandmark68Net**: Detects 68 facial keypoints
- **FaceRecognitionNet**: Generates 128-dimensional face descriptors

### Algorithms

- **Duplicate Detection**: Intersection over Union (IoU) with 30% threshold
- **Clustering**: Union-Find with single-linkage clustering
- **Distance Metric**: Euclidean distance between face descriptors

### Performance

- Model loading: ~30-60 seconds (first run, downloads from CDN)
- Per image processing: ~1-5 seconds depending on face count
- Memory usage: ~500MB-1GB during operation

## Troubleshooting

### Models take too long to load
- First run downloads models from CDN (~100MB)
- Subsequent runs use cached models (faster)
- Ensure stable internet connection

### No faces detected
- Check image quality and lighting
- Ensure faces are clearly visible
- Try increasing `scoreThreshold` for more lenient detection

### False positive detections
- Reduce `scoreThreshold` for stricter detection
- Verify input image quality

### Memory issues
- Process images in smaller batches
- Close other applications
- Increase Node.js heap size: `node --max-old-space-size=4096 faceDetection.js`

## Dependencies

- **@tensorflow/tfjs**: Machine learning framework
- **@vladmandic/face-api**: Face detection and recognition
- **canvas**: Image processing
- **Node.js native modules**: Image handling

## License

ISC

## Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## Contact

For questions or issues, please visit: https://github.com/abhijit0404/facedetection/issues