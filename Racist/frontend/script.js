const video = document.getElementById("video");
const canvas = document.getElementById("snapshot");
const statusMessage = document.getElementById("statusMessage");
const knownFacesList = document.getElementById("knownFaces");
const groupSelect = document.getElementById("groupSelect");

const startCameraBtn = document.getElementById("startCameraBtn");
const registerBtn = document.getElementById("registerBtn");
const realTimeBtn = document.getElementById("realTimeBtn");
const batchUploadBtn = document.getElementById("batchUploadBtn");
const batchFileInput = document.getElementById("batchFileInput");
const refreshFacesBtn = document.getElementById("refreshFacesBtn");
const clearFacesBtn = document.getElementById("clearFacesBtn");

let realTimeMode = false;
let realTimeInterval = null;
let cameraActive = false;
let faceCounter = 0;

async function initializeCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera not supported by this browser.", true);
      return;
    }

    // List available devices for debugging
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    console.log('Available cameras:', videoDevices);

    if (videoDevices.length === 0) {
      setStatus("No camera detected on this device.", true);
      return;
    }

    setStatus("Starting camera...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: videoDevices[0].deviceId
      }
    });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      video.play();
      cameraActive = true;
      startCameraBtn.textContent = "Camera Active ✓";
      startCameraBtn.disabled = true;
      startCameraBtn.style.background = "#10b981";
      registerBtn.disabled = false;
      realTimeBtn.disabled = false;
      setStatus("✓ Camera ready");
    };
  } catch (error) {
    console.error("Camera error:", error);
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      setStatus("❌ Camera permission denied. Click the camera icon in the address bar and allow access.", true);
    } else if (error.name === "NotFoundError") {
      setStatus("❌ No camera found. Make sure a camera is connected.", true);
    } else if (error.name === "NotReadableError") {
      setStatus("❌ Camera in use. Close other apps using the camera.", true);
    } else {
      setStatus(`❌ Camera error: ${error.message}`, true);
    }
  }
}

function captureFrame() {
  const context = canvas.getContext("2d");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Unexpected server error.");
  }
  return data;
}

async function registerFace() {
  const group = groupSelect.value;
  try {
    setStatus("Capturing face...");
    const image = captureFrame();
    faceCounter++;
    const name = `Face_${group}_${faceCounter}_${Date.now()}`;
    const result = await postJson("/api/register", { name, image, group });
    setStatus(`✓ Registered to Group ${result.group}`);
    await refreshKnownFaces();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function recognizeFace() {
  try {
    if (!realTimeMode) {
      setStatus("Analyzing face...");
    }
    const image = captureFrame();
    const result = await postJson("/api/recognize", { image });

    if (result.status === "recognized") {
      // Exact match found
      const similarity = (1 - result.distance).toFixed(2);
      setStatus(`✓ GROUP ${result.group} (Match: ${similarity})`);
      if (result.group === "A") {
        statusMessage.style.background = "rgba(59, 130, 246, 0.2)";
        statusMessage.style.border = "2px solid rgba(59, 130, 246, 0.6)";
      } else {
        statusMessage.style.background = "rgba(16, 185, 129, 0.2)";
        statusMessage.style.border = "2px solid rgba(16, 185, 129, 0.6)";
      }
    } else if (result.status === "predicted") {
      // ML prediction for unknown face
      const confidence = (result.confidence * 100).toFixed(0);
      setStatus(`🔮 PREDICTED: GROUP ${result.group} (${confidence}% confidence)`);
      if (result.group === "A") {
        statusMessage.style.background = "rgba(59, 130, 246, 0.15)";
        statusMessage.style.border = "2px dashed rgba(59, 130, 246, 0.6)";
      } else {
        statusMessage.style.background = "rgba(16, 185, 129, 0.15)";
        statusMessage.style.border = "2px dashed rgba(16, 185, 129, 0.6)";
      }
    } else {
      setStatus("❌ Unknown face - not enough data to predict", true);
    }
  } catch (error) {
    if (!realTimeMode) {
      setStatus(error.message, true);
    }
  }
}

async function refreshKnownFaces() {
  try {
    const response = await fetch("/api/faces");
    const data = await response.json();
    knownFacesList.innerHTML = "";
    if (!data.faces || data.faces.length === 0) {
      knownFacesList.innerHTML = "<li>No registered faces yet.</li>";
      return;
    }

    // Count by group
    const groupA = data.faces.filter(f => f.group === "A").length;
    const groupJ = data.faces.filter(f => f.group === "J").length;

    // Display group counts
    const listItemA = document.createElement("li");
    listItemA.textContent = `Group A: ${groupA} members`;
    listItemA.style.color = "#3b82f6";
    listItemA.style.fontWeight = "bold";
    knownFacesList.appendChild(listItemA);

    const listItemJ = document.createElement("li");
    listItemJ.textContent = `Group J: ${groupJ} members`;
    listItemJ.style.color = "#10b981";
    listItemJ.style.fontWeight = "bold";
    knownFacesList.appendChild(listItemJ);

  } catch (error) {
    setStatus(`Could not load known faces: ${error.message}`, true);
  }
}

async function clearKnownFaces() {
  try {
    const response = await fetch("/api/faces", { method: "DELETE" });
    const data = await response.json();
    setStatus("Face database cleared.");
    knownFacesList.innerHTML = "<li>No registered faces yet.</li>";
    return data;
  } catch (error) {
    setStatus(`Failed to clear database: ${error.message}`, true);
  }
}

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.style.background = isError
    ? "rgba(239, 68, 68, 0.15)"
    : "rgba(16, 185, 129, 0.1)";
  statusMessage.style.border = isError
    ? "1px solid rgba(239, 68, 68, 0.4)"
    : "1px solid rgba(16, 185, 129, 0.4)";
}

function toggleRealTimeDetection() {
  realTimeMode = !realTimeMode;
  if (realTimeMode) {
    realTimeBtn.textContent = "Stop Real-Time";
    realTimeBtn.style.background = "#ef4444";
    setStatus("🔴 Real-time detection active...");
    realTimeInterval = setInterval(recognizeFace, 1000); // Check every second
  } else {
    realTimeBtn.textContent = "Real-Time Detection";
    realTimeBtn.style.background = "";
    clearInterval(realTimeInterval);
    setStatus("Real-time detection stopped.");
  }
}

async function handleBatchUpload() {
  const files = batchFileInput.files;
  if (!files || files.length === 0) {
    setStatus("Please select images to upload.", true);
    return;
  }

  const group = groupSelect.value;
  setStatus(`Processing ${files.length} images...`);

  const faces = [];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    try {
      const imageData = await fileToDataURL(file);
      // Extract name from filename (remove extension)
      const name = file.name.replace(/\.[^/.]+$/, "");
      faces.push({ name, image: imageData, group });
    } catch (error) {
      console.error(`Failed to process ${file.name}:`, error);
    }
  }

  try {
    const result = await postJson("/api/register_batch", { faces });
    setStatus(
      `Batch complete: ${result.successful} registered, ${result.failed} failed.`
    );
    await refreshKnownFaces();
  } catch (error) {
    setStatus(`Batch upload failed: ${error.message}`, true);
  }

  batchFileInput.value = ""; // Reset file input
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

startCameraBtn.addEventListener("click", initializeCamera);
registerBtn.addEventListener("click", registerFace);
realTimeBtn.addEventListener("click", toggleRealTimeDetection);
batchUploadBtn.addEventListener("click", () => batchFileInput.click());
batchFileInput.addEventListener("change", handleBatchUpload);
refreshFacesBtn.addEventListener("click", refreshKnownFaces);
clearFacesBtn.addEventListener("click", clearKnownFaces);

refreshKnownFaces();
