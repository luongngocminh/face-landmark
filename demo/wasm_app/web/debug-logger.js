/**
 * Debug Logger for Face Landmark Detection
 * Provides utilities for debugging face detection issues
 */

class FaceDetectionDebugger {
    constructor() {
        this.logs = [];
        this.maxLogs = 200;
        this.enabled = false;
        this.faceAngleData = [];
        this.faceDetectionStats = {
            totalAttempts: 0,
            successfulDetections: 0,
            failedDetections: 0,
            lastNResults: []
        };
        this.maxStatsEntries = 20;
    }
    
    enable() {
        this.enabled = true;
        console.log("Face detection debugger enabled");
        return this;
    }
    
    disable() {
        this.enabled = false;
        return this;
    }
    
    log(message, data = null) {
        if (!this.enabled) return;
        
        const entry = {
            timestamp: new Date().toISOString(),
            message,
            data
        };
        
        this.logs.unshift(entry);
        if (this.logs.length > this.maxLogs) {
            this.logs.pop();
        }
        
        console.log(`[FaceDectDebug] ${message}`, data || '');
        return this;
    }
    
    trackDetectionAttempt(success, landmarks, processingTime) {
        if (!this.enabled) return;
        
        this.faceDetectionStats.totalAttempts++;
        if (success) {
            this.faceDetectionStats.successfulDetections++;
        } else {
            this.faceDetectionStats.failedDetections++;
        }
        
        // Store the result
        this.faceDetectionStats.lastNResults.unshift({
            timestamp: new Date().toISOString(),
            success,
            landmarksCount: landmarks ? landmarks.length/2 : 0,
            processingTime
        });
        
        // Trim to max size
        if (this.faceDetectionStats.lastNResults.length > this.maxStatsEntries) {
            this.faceDetectionStats.lastNResults.pop();
        }
        
        // Log the result
        this.log(`Detection attempt: ${success ? "SUCCESS" : "FAILURE"}`, {
            landmarksCount: landmarks ? landmarks.length/2 : 0,
            processingTime
        });
        
        // If we have landmarks, analyze head pose
        if (success && landmarks && landmarks.length >= 212) {
            this.analyzeHeadPose(landmarks);
        }
        
        return this;
    }
    
    analyzeHeadPose(landmarks) {
        if (!this.enabled) return;
        
        // Skip ROI values (first 4)
        const actualLandmarks = landmarks.slice(4);
        
        try {
            // Use key facial landmarks to estimate head pose
            const leftEyeX = actualLandmarks[0], leftEyeY = actualLandmarks[1];
            const rightEyeX = actualLandmarks[72], rightEyeY = actualLandmarks[73];
            const noseX = actualLandmarks[54], noseY = actualLandmarks[55];
            
            // Calculate basic metrics
            const eyeDistance = Math.sqrt(Math.pow(rightEyeX - leftEyeX, 2) + Math.pow(rightEyeY - leftEyeY, 2));
            const midEyeX = (leftEyeX + rightEyeX) / 2;
            const midEyeY = (leftEyeY + rightEyeY) / 2;
            const eyeToNoseDistance = Math.sqrt(Math.pow(noseX - midEyeX, 2) + Math.pow(noseY - midEyeY, 2));
            const eyeSlope = (rightEyeY - leftEyeY) / (rightEyeX - leftEyeX);
            
            // Store this data
            this.faceAngleData.unshift({
                timestamp: new Date().toISOString(),
                eyeDistance,
                eyeToNoseDistance,
                eyeSlope,
                estimatedRotation: Math.atan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX) * (180/Math.PI)
            });
            
            // Trim to max size
            if (this.faceAngleData.length > this.maxStatsEntries) {
                this.faceAngleData.pop();
            }
            
            // Log the analysis
            this.log("Head pose analysis", {
                eyeDistance,
                eyeToNoseDistance,
                eyeSlope,
                estimatedRotation: Math.atan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX) * (180/Math.PI)
            });
        } catch (e) {
            this.log("Error analyzing head pose", e.message);
        }
        
        return this;
    }
    
    getSuccessRate() {
        if (this.faceDetectionStats.totalAttempts === 0) return 0;
        return this.faceDetectionStats.successfulDetections / this.faceDetectionStats.totalAttempts;
    }
    
    getDebugReport() {
        return {
            enabled: this.enabled,
            statistics: {
                totalAttempts: this.faceDetectionStats.totalAttempts,
                successfulDetections: this.faceDetectionStats.successfulDetections,
                failedDetections: this.faceDetectionStats.failedDetections,
                successRate: this.getSuccessRate()
            },
            recentDetections: this.faceDetectionStats.lastNResults,
            headPoseData: this.faceAngleData,
            logs: this.logs
        };
    }
    
    downloadDebugReport() {
        const report = this.getDebugReport();
        const json = JSON.stringify(report, null, 2);
        const blob = new Blob([json], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `face-detection-debug-report-${new Date().toISOString().substr(0, 19).replace(/:/g, '-')}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    displayStatsToElement(elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        const stats = this.getDebugReport().statistics;
        const successRate = (stats.successRate * 100).toFixed(1);
        
        let html = `
            <div class="debug-stats">
                <h4>Detection Statistics</h4>
                <p>Total attempts: ${stats.totalAttempts}</p>
                <p>Successful: ${stats.successfulDetections}</p>
                <p>Failed: ${stats.failedDetections}</p>
                <p>Success rate: ${successRate}%</p>
                <div class="progress-bar">
                    <div class="progress" style="width: ${successRate}%"></div>
                </div>
            </div>
        `;
        
        element.innerHTML = html;
    }
}

// Create a global instance
window.faceDebugger = new FaceDetectionDebugger();
