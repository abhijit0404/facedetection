const tf = require('@tensorflow/tfjs-backend-cpu');
const faceapi = require('@vladmandic/face-api');
const { Canvas, Image, ImageData } = require('canvas');
const fs = require('fs');
const path = require('path');

// Setup canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const INPUT_DIR = '/Users/abhijit/Documents/Abhijit/Coding/petprojects/FaceDetection/inputimages';
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model/';

let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;
  
  console.log('Loading face detection models...');
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    console.log('✓ Models loaded successfully\n');
    modelsLoaded = true;
  } catch (error) {
    console.error('✗ Error loading models:', error.message);
    process.exit(1);
  }
}

function clusterFaces(detections, distanceThreshold = 0.6) {
  if (detections.length === 0) return [];
  if (detections.length === 1) return [[detections[0]]];
  
  // First, filter out duplicate detections in the same spatial region
  const filtered = [];
  const used = new Set();
  
  for (let i = 0; i < detections.length; i++) {
    if (used.has(i)) continue;
    
    filtered.push(detections[i]);
    used.add(i);
    
    // Mark similar detections in the same region as duplicates
    for (let j = i + 1; j < detections.length; j++) {
      if (used.has(j)) continue;
      
      const box1 = detections[i].detection.box;
      const box2 = detections[j].detection.box;
      
      // Calculate overlap (IoU - Intersection over Union)
      const x1 = Math.max(box1.x, box2.x);
      const y1 = Math.max(box1.y, box2.y);
      const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
      const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
      
      if (x2 > x1 && y2 > y1) {
        const intersection = (x2 - x1) * (y2 - y1);
        const area1 = box1.width * box1.height;
        const area2 = box2.width * box2.height;
        const union = area1 + area2 - intersection;
        const iou = intersection / union;
        
        // If overlap is > 30%, it's likely a duplicate
        if (iou > 0.3) {
          used.add(j);
        }
      }
    }
  }
  
  console.log(`  Filtered to ${filtered.length} unique detections`);
  
  // Build distance matrix
  const distanceMatrix = [];
  for (let i = 0; i < filtered.length; i++) {
    distanceMatrix[i] = [];
    for (let j = 0; j < filtered.length; j++) {
      if (i === j) {
        distanceMatrix[i][j] = 0;
      } else {
        const distance = faceapi.euclideanDistance(
          filtered[i].descriptor,
          filtered[j].descriptor
        );
        distanceMatrix[i][j] = distance;
      }
    }
  }
  
  // Debug: show distances between all faces
  console.log('  Distance matrix:');
  for (let i = 0; i < filtered.length; i++) {
    for (let j = i + 1; j < filtered.length; j++) {
      console.log(`    Face ${i} <-> Face ${j}: ${distanceMatrix[i][j].toFixed(4)}`);
    }
  }
  
  // Use Union-Find for clustering
  const parent = Array.from({length: filtered.length}, (_, i) => i);
  
  function find(x) {
    if (parent[x] !== x) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }
  
  function union(x, y) {
    const rootX = find(x);
    const rootY = find(y);
    if (rootX !== rootY) {
      parent[rootY] = rootX;
    }
  }
  
  // Create list of all pairwise distances and sort
  const pairs = [];
  for (let i = 0; i < filtered.length; i++) {
    for (let j = i + 1; j < filtered.length; j++) {
      pairs.push({i, j, distance: distanceMatrix[i][j]});
    }
  }
  pairs.sort((a, b) => a.distance - b.distance);
  
  // Merge faces that are close enough (single-linkage clustering)
  for (const {i, j, distance} of pairs) {
    if (distance < distanceThreshold) {
      union(i, j);
    }
  }
  
  // Group detections by cluster
  const clusters = new Map();
  for (let i = 0; i < filtered.length; i++) {
    const root = find(i);
    if (!clusters.has(root)) {
      clusters.set(root, []);
    }
    clusters.get(root).push(filtered[i]);
  }
  
  return Array.from(clusters.values());
}

async function detectFacesInImage(imagePath) {
  try {
    console.log(`Processing: ${path.basename(imagePath)}`);
    
    // Read image file
    const imageBuffer = fs.readFileSync(imagePath);
    const img = new Image();
    
    // Wait for image to load with timeout
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Image loading timeout'));
      }, 5000);
      
      img.onload = () => {
        clearTimeout(timeout);
        console.log(`  ✓ Image loaded (${img.width}x${img.height})`);
        resolve();
      };
      
      img.onerror = (err) => {
        clearTimeout(timeout);
        reject(new Error('Failed to load image'));
      };
      
      img.src = imageBuffer;
    });
    
    // Detect all faces with landmarks and descriptors
    console.log('  Running face detection...');
    const detections = await faceapi
      .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: 0.05
      }))
      .withFaceLandmarks()
      .withFaceDescriptors();
    
    console.log(`  → Found ${detections.length} detection(s)`);
    
    if (detections.length === 0) {
      return 0;
    }
    
    // Cluster faces by identity - use a more aggressive clustering threshold
    let distanceThreshold = 0.55;
    
    const clusters = clusterFaces(detections, distanceThreshold);
    console.log(`  → ${clusters.length} unique individual(s)`);
    
    return clusters.length;
  } catch (error) {
    console.error(`  ✗ Error: ${error.message}`);
    return 0;
  }
}

async function main() {
  try {
    // Load models
    await loadModels();
    
    // Get image files
    const imageFiles = fs.readdirSync(INPUT_DIR)
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return ['.jpg', '.jpeg', '.png'].includes(ext);
      })
      .sort();
    
    if (imageFiles.length === 0) {
      console.log('No images found in input directory.');
      return;
    }
    
    console.log(`Found ${imageFiles.length} image(s)\n`);
    
    // Process each image
    const results = [];
    for (const imageFile of imageFiles) {
      const imagePath = path.join(INPUT_DIR, imageFile);
      const uniqueCount = await detectFacesInImage(imagePath);
      results.push({ filename: imageFile, count: uniqueCount });
      console.log('');
    }
    
    // Print results
    console.log('========== RESULTS ==========\n');
    results.forEach(r => console.log(`${r.filename}: ${r.count} individual(s)`));
    const total = results.reduce((sum, r) => sum + r.count, 0);
    console.log(`\nTotal: ${total} individual(s)\n`);
    
  } catch (error) {
    console.error('Fatal error:', error.message);
    process.exit(1);
  }
}

main();
