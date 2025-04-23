let wasmModule = null;
let video = null;
let canvas = null;
let ctx = null;
let isModuleLoaded = false;
let isModelLoaded = false;
let isStreamActive = false;
let isContinuousMode = false;
let animationFrameId = null;
let pendingPromises = {};
let lastProcessingTime = 0;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const statusElement = document.getElementById('status');
    const threadingInfoElement = document.getElementById('threading-info');
    const startBtn = document.getElementById('startBtn');
    const detectBtn = document.getElementById('detectBtn');
    const continuousBtn = document.getElementById('continuousBtn');
    const stopBtn = document.getElementById('stopBtn');
    const asyncSwitch = document.getElementById('asyncSwitch');
    
    video = document.getElementById('video');
    canvas = document.getElementById('overlay');
    ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = 640;
    canvas.height = 480;
    
    // Define the callback for async landmark detection
    window._landmarkDetectionComplete = function(promiseId, landmarksPtr, numPoints) {
        const promise = pendingPromises[promiseId];
        if (promise) {
            const endTime = performance.now();
            lastProcessingTime = endTime - promise.startTime;
            
            if (landmarksPtr && numPoints > 0) {
                // Extract landmarks
                const landmarks = [];
                for (let i = 0; i < numPoints; i++) {
                    landmarks.push(wasmModule.getValue(landmarksPtr + (i * 4), 'float'));
                }
                
                // Free the WASM memory
                wasmModule.ccall('freeMemory', null, ['number'], [landmarksPtr]);
                
                // Resolve the promise
                promise.resolve(landmarks);
            } else {
                promise.resolve([]);
            }
            
            // Remove from pending promises
            delete pendingPromises[promiseId];
        }
    };
    
    // Initialize WebAssembly module
    FaceLandmarkModule().then(module => {
        wasmModule = module;
        isModuleLoaded = true;
        statusElement.textContent = 'Status: WASM module loaded. Initialize detector...';
        
        // Check if threading is supported
        const threadingSupported = wasmModule.ccall('isThreadingSupported', 'number', [], []) === 1;
        const numThreads = wasmModule.ccall('getNumHardwareThreads', 'number', [], []);
        
        if (threadingSupported) {
            threadingInfoElement.textContent = `Threading is supported. Available threads: ${numThreads}`;
        } else {
            threadingInfoElement.textContent = 'Threading is not supported in this browser. Using single thread mode.';
            asyncSwitch.checked = false;
            asyncSwitch.disabled = true;
        }
        
        // Initialize the detector
        if (wasmModule.ccall('initialize', 'number', [], [])) {
            statusElement.textContent = 'Status: Detector initialized. Ready to use.';
            // In a real app, you would load the model here
            // For demo purposes, we'll simulate model loading
            isModelLoaded = true;
            startBtn.disabled = false;
        } else {
            statusElement.textContent = 'Status: Failed to initialize detector.';
        }
    }).catch(err => {
        console.error('Failed to load WASM module:', err);
        statusElement.textContent = 'Status: Failed to load WASM module.';
    });
    
    // Button event listeners
    startBtn.addEventListener('click', startCamera);
    detectBtn.addEventListener('click', () => detectLandmarks(false));
    continuousBtn.addEventListener('click', toggleContinuousMode);
    stopBtn.addEventListener('click', stopCamera);
    
    // Switch event listener
    asyncSwitch.addEventListener('change', () => {
        const statusText = asyncSwitch.checked ? 
            'Using asynchronous (threaded) processing.' :
            'Using synchronous (blocking) processing.';
        threadingInfoElement.textContent += ' ' + statusText;
    });
});

// Start the camera
function startCamera() {
    if (!isModuleLoaded || !isModelLoaded) {
        alert('Module or model not loaded yet.');
        return;
    }
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            isStreamActive = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('detectBtn').disabled = false;
            document.getElementById('continuousBtn').disabled = false;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('status').textContent = 'Status: Camera active.';
        })
        .catch(err => {
            console.error('Error accessing camera:', err);
            document.getElementById('status').textContent = 'Status: Failed to access camera.';
        });
}

// Toggle continuous detection mode
function toggleContinuousMode() {
    isContinuousMode = !isContinuousMode;
    const continuousBtn = document.getElementById('continuousBtn');
    
    if (isContinuousMode) {
        continuousBtn.textContent = 'Stop Continuous';
        document.getElementById('detectBtn').disabled = true;
        processContinuously();
    } else {
        continuousBtn.textContent = 'Continuous Detection';
        document.getElementById('detectBtn').disabled = false;
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }
}

// Process frames continuously
function processContinuously() {
    if (!isContinuousMode) return;
    
    detectLandmarks(true);
    animationFrameId = requestAnimationFrame(processContinuously);
}

// Detect landmarks in the current video frame
function detectLandmarks(isContinuous = false) {
    if (!isStreamActive || !isModuleLoaded || !isModelLoaded) {
        return;
    }
    
    const useAsync = document.getElementById('asyncSwitch').checked;
    const statusElement = document.getElementById('status');
    const startTime = performance.now();
    
    // Create a temporary canvas to capture the video frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Get image data
    const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Allocate memory for image data in WASM
    const imageDataPtr = wasmModule._malloc(imageData.data.length);
    wasmModule.HEAPU8.set(imageData.data, imageDataPtr);
    
    if (useAsync) {
        // Call the async detection method
        const promiseId = wasmModule.ccall(
            'detectLandmarksAsync',
            'number',
            ['number', 'number', 'number'],
            [imageDataPtr, tempCanvas.width, tempCanvas.height]
        );
        
        // Create a promise that will be resolved when the detection completes
        const promise = new Promise((resolve) => {
            pendingPromises[promiseId] = {
                resolve,
                startTime: startTime
            };
        });
        
        // Process the results when available
        promise.then(landmarks => {
            // Free the image data memory
            wasmModule._free(imageDataPtr);
            
            // Draw the landmarks
            drawLandmarks(landmarks, tempCanvas.width, tempCanvas.height);
            
            // Update status
            const fps = lastProcessingTime > 0 ? Math.round(1000 / lastProcessingTime) : 0;
            statusElement.textContent = `Status: Detected ${landmarks.length/2} landmarks. Processing time: ${Math.round(lastProcessingTime)}ms (${fps} FPS)`;
        });
    } else {
        // Synchronous detection
        // Allocate memory for numPoints output parameter
        const numPointsPtr = wasmModule._malloc(4);
        
        // Call the detection function
        const landmarksPtr = wasmModule.ccall(
            'detectLandmarks', 
            'number',
            ['number', 'number', 'number', 'number'],
            [imageDataPtr, tempCanvas.width, tempCanvas.height, numPointsPtr]
        );
        
        // Get number of points
        const numPoints = wasmModule.getValue(numPointsPtr, 'i32');
        
        // Clear overlay canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Extract landmarks if any were detected
        let landmarks = [];
        if (landmarksPtr && numPoints > 0) {
            for (let i = 0; i < numPoints; i++) {
                landmarks.push(wasmModule.getValue(landmarksPtr + (i * 4), 'float'));
            }
            
            // Free the landmarks array
            wasmModule._free(landmarksPtr);
        }
        
        // Free allocated memory
        wasmModule._free(imageDataPtr);
        wasmModule._free(numPointsPtr);
        
        // Draw the landmarks
        drawLandmarks(landmarks, tempCanvas.width, tempCanvas.height);
        
        // Calculate and show processing time
        const endTime = performance.now();
        lastProcessingTime = endTime - startTime;
        const fps = Math.round(1000 / lastProcessingTime);
        statusElement.textContent = `Status: Detected ${landmarks.length/2} landmarks. Processing time: ${Math.round(lastProcessingTime)}ms (${fps} FPS)`;
    }
}

// Draw landmarks on the canvas
function drawLandmarks(landmarks, sourceWidth, sourceHeight) {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (landmarks.length === 0) return;
    
    const scaleX = canvas.width / sourceWidth;
    const scaleY = canvas.height / sourceHeight;
    
    ctx.fillStyle = 'red';
    
    for (let i = 0; i < landmarks.length; i += 2) {
        const x = landmarks[i] * scaleX;
        const y = landmarks[i + 1] * scaleY;
        
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Stop the camera
function stopCamera() {
    if (isContinuousMode) {
        toggleContinuousMode();
    }
    
    if (video && video.srcObject) {
        const stream = video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        isStreamActive = false;
        
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('detectBtn').disabled = true;
        document.getElementById('continuousBtn').disabled = true;
        document.getElementById('stopBtn').disabled = true;
        document.getElementById('status').textContent = 'Status: Camera stopped.';
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (isStreamActive) {
        stopCamera();
    }
    
    if (isModuleLoaded) {
        wasmModule.ccall('cleanup', null, [], []);
    }
});
