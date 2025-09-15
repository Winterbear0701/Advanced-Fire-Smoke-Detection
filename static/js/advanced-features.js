// Advanced Features JavaScript Module

class FireDetectionApp {
    constructor() {
        this.webcamStream = null;
        this.detectionInterval = null;
        this.isRealTimeDetection = false;
        this.detectionHistory = [];
        this.currentModel = 'yolov8n';
        this.confidence = 0.5;
        
        this.initializeApp();
    }

    async initializeApp() {
        this.setupEventListeners();
        await this.loadAvailableModels();
        await this.loadDetectionHistory();
        this.setupWebSocket();
    }

    setupEventListeners() {
        // File upload with drag and drop
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = uploadArea?.querySelector('input[type="file"]');

        if (uploadArea && fileInput) {
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Model selection
        document.querySelectorAll('input[name="model"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                this.updateModelInfo(e.target.value);
            });
        });

        // Confidence slider
        const confidenceSlider = document.getElementById('confidenceSlider');
        if (confidenceSlider) {
            confidenceSlider.addEventListener('input', (e) => {
                this.confidence = parseFloat(e.target.value);
                document.getElementById('confidenceValue').textContent = this.confidence;
            });
        }

        // Webcam controls
        const webcamBtn = document.getElementById('webcamBtn');
        const stopWebcamBtn = document.getElementById('stopWebcamBtn');
        const captureBtn = document.getElementById('captureBtn');

        if (webcamBtn) webcamBtn.addEventListener('click', this.startWebcam.bind(this));
        if (stopWebcamBtn) stopWebcamBtn.addEventListener('click', this.stopWebcam.bind(this));
        if (captureBtn) captureBtn.addEventListener('click', this.captureFrame.bind(this));

        // Real-time detection toggle
        const realtimeToggle = document.getElementById('realtimeDetection');
        if (realtimeToggle) {
            realtimeToggle.addEventListener('change', this.toggleRealTimeDetection.bind(this));
        }

        // Batch processing
        const batchBtn = document.getElementById('batchProcessBtn');
        if (batchBtn) batchBtn.addEventListener('click', this.startBatchProcessing.bind(this));

        // History controls
        const historyBtn = document.getElementById('viewHistoryBtn');
        if (historyBtn) historyBtn.addEventListener('click', this.showDetectionHistory.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFiles(files);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFiles(files);
        }
    }

    async processFiles(files) {
        if (files.length === 1) {
            await this.processSingleFile(files[0]);
        } else {
            await this.processBatchFiles(files);
        }
    }

    async processSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_type', this.currentModel);
        formData.append('confidence', this.confidence);
        formData.append('save_results', document.getElementById('saveResults')?.checked || false);

        this.showProgress();
        this.updateUploadArea(file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result);
                this.addToHistory(result);
                this.showAlert('success', 'Detection completed successfully!');
            } else {
                this.showAlert('error', result.message || 'Detection failed');
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.showAlert('error', 'An error occurred during detection');
        } finally {
            this.hideProgress();
        }
    }

    async processBatchFiles(files) {
        this.showBatchProgress(files.length);
        const results = [];

        for (let i = 0; i < files.length; i++) {
            try {
                this.updateBatchProgress(i, files.length, files[i].name);
                const result = await this.processSingleFileQuiet(files[i]);
                results.push(result);
            } catch (error) {
                console.error(`Error processing ${files[i].name}:`, error);
                results.push({ error: error.message, filename: files[i].name });
            }
        }

        this.hideBatchProgress();
        this.displayBatchResults(results);
    }

    async processSingleFileQuiet(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_type', this.currentModel);
        formData.append('confidence', this.confidence);

        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    updateUploadArea(file) {
        const uploadArea = document.getElementById('uploadArea');
        const icon = uploadArea?.querySelector('.upload-icon');
        const text = uploadArea?.querySelector('.upload-text');
        
        if (icon && text) {
            icon.className = file.type.startsWith('image') ? 
                'fas fa-image upload-icon' : 'fas fa-video upload-icon';
            text.textContent = `Processing: ${file.name}`;
        }
    }

    showProgress() {
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'block';
            this.animateProgress();
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('progressContainer');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
        this.resetUploadArea();
    }

    animateProgress() {
        let progress = 0;
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            if (progressFill) progressFill.style.width = progress + '%';
            
            if (progressText) {
                if (progress < 30) progressText.textContent = 'Initializing AI model...';
                else if (progress < 60) progressText.textContent = 'Processing media...';
                else if (progress < 90) progressText.textContent = 'Analyzing detections...';
                else progressText.textContent = 'Finalizing results...';
            }
        }, 300);
        
        this.progressInterval = interval;
    }

    resetUploadArea() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        const uploadArea = document.getElementById('uploadArea');
        const icon = uploadArea?.querySelector('.upload-icon');
        const text = uploadArea?.querySelector('.upload-text');
        
        if (icon && text) {
            icon.className = 'fas fa-cloud-upload-alt upload-icon';
            text.textContent = 'Drop files here or click to upload';
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        if (!resultsSection) return;

        resultsSection.style.display = 'block';
        
        // Update media displays
        this.updateResultMedia(result);
        
        // Update statistics
        this.updateStatistics(result);
        
        // Show alerts
        this.showResultAlert(result);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Update charts if available
        this.updateCharts(result);
    }

    updateResultMedia(result) {
        // Update original and processed media
        const originalMedia = document.querySelector('.result-card:first-child .result-media');
        const processedMedia = document.querySelector('.result-card:last-child .result-media');
        
        if (originalMedia && result.uploaded_file) {
            if (result.file_type === 'image') {
                originalMedia.src = result.uploaded_file;
                originalMedia.tagName = 'IMG';
            } else {
                originalMedia.src = result.uploaded_file;
                originalMedia.tagName = 'VIDEO';
            }
        }
        
        if (processedMedia && result.processed_file) {
            if (result.file_type === 'image') {
                processedMedia.src = result.processed_file;
                processedMedia.tagName = 'IMG';
            } else {
                processedMedia.src = result.processed_file;
                processedMedia.tagName = 'VIDEO';
            }
        }
    }

    updateStatistics(result) {
        const updates = {
            'detectionCount': result.detection_count || 0,
            'maxConfidence': result.max_confidence || 'N/A',
            'processingTime': result.processing_time || 'N/A',
            'riskLevel': result.risk_level || 'Low'
        };
        
        Object.entries(updates).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                element.classList.add('stat-updated');
                setTimeout(() => element.classList.remove('stat-updated'), 500);
            }
        });
    }

    showResultAlert(result) {
        let alertType = 'success';
        let message = '✅ No fire or smoke detected.';
        
        if (result.detection_count > 0) {
            if (result.fire_count > 0) {
                alertType = 'error';
                message = `🔥 FIRE DETECTED! ${result.fire_count} fire detection(s) found!`;
            } else if (result.smoke_count > 0) {
                alertType = 'warning';
                message = `💨 SMOKE DETECTED! ${result.smoke_count} smoke detection(s) found!`;
            } else {
                alertType = 'warning';
                message = `⚠️ ${result.detection_count} detection(s) found!`;
            }
        }
        
        this.showAlert(alertType, message);
    }

    showAlert(type, message, duration = 5000) {
        const alertHtml = `
            <div class="alert alert-${type}" style="animation: slideInDown 0.5s ease-out;">
                <i class="fas fa-${this.getAlertIcon(type)}"></i>
                <span>${message}</span>
                <button class="alert-close" onclick="this.parentElement.remove()">×</button>
            </div>
        `;
        
        // Remove existing alerts
        document.querySelectorAll('.alert').forEach(alert => alert.remove());
        
        // Add new alert
        const resultsSection = document.getElementById('resultsSection') || document.body;
        resultsSection.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-remove after duration
        setTimeout(() => {
            const alert = document.querySelector('.alert');
            if (alert) {
                alert.style.animation = 'slideOutUp 0.5s ease-in';
                setTimeout(() => alert.remove(), 500);
            }
        }, duration);
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'warning': 'exclamation-triangle',
            'error': 'times-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    async startWebcam() {
        try {
            this.webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('webcamVideo');
            if (video) {
                video.srcObject = this.webcamStream;
                document.getElementById('webcamSection').style.display = 'block';
                
                const webcamBtn = document.getElementById('webcamBtn');
                if (webcamBtn) {
                    webcamBtn.innerHTML = '<i class="fas fa-video"></i> Webcam Active';
                    webcamBtn.disabled = true;
                }
                
                this.showAlert('success', 'Webcam started successfully!');
            }
        } catch (error) {
            console.error('Webcam error:', error);
            this.showAlert('error', 'Failed to access webcam: ' + error.message);
        }
    }

    stopWebcam() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        
        const webcamSection = document.getElementById('webcamSection');
        if (webcamSection) webcamSection.style.display = 'none';
        
        const webcamBtn = document.getElementById('webcamBtn');
        if (webcamBtn) {
            webcamBtn.innerHTML = '<i class="fas fa-camera"></i> Start Webcam';
            webcamBtn.disabled = false;
        }
        
        this.stopRealTimeDetection();
        this.showAlert('info', 'Webcam stopped.');
    }

    captureFrame() {
        const video = document.getElementById('webcamVideo');
        if (!video || !this.webcamStream) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            const file = new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' });
            await this.processSingleFile(file);
        });
    }

    toggleRealTimeDetection(e) {
        this.isRealTimeDetection = e.target.checked;
        
        if (this.isRealTimeDetection && this.webcamStream) {
            this.startRealTimeDetection();
        } else {
            this.stopRealTimeDetection();
        }
    }

    startRealTimeDetection() {
        if (this.detectionInterval) return;
        
        this.detectionInterval = setInterval(() => {
            this.processWebcamFrame();
        }, 2000); // Process every 2 seconds
        
        this.showAlert('info', 'Real-time detection started');
    }

    stopRealTimeDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }

    async processWebcamFrame() {
        const video = document.getElementById('webcamVideo');
        if (!video || !this.webcamStream) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append('file', blob, 'realtime_frame.jpg');
                formData.append('model_type', this.currentModel);
                formData.append('confidence', this.confidence);
                
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success && result.detection_count > 0) {
                    this.showWebcamAlert(result);
                }
            } catch (error) {
                console.error('Real-time detection error:', error);
            }
        });
    }

    showWebcamAlert(result) {
        const overlay = document.getElementById('webcamOverlay');
        if (!overlay) return;
        
        overlay.style.border = result.fire_count > 0 ? '5px solid red' : '3px solid orange';
        overlay.style.animation = 'pulse 1s ease-in-out';
        
        setTimeout(() => {
            overlay.style.border = 'none';
            overlay.style.animation = 'none';
        }, 2000);
        
        // Play alert sound (if enabled)
        this.playAlertSound();
    }

    playAlertSound() {
        try {
            const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+Dpt2AcBz2bz++EWw8HYr7n5KxaFgdqt+zhpVIeBT6ax/DXfykJJHjJ8tyOQAoTXbPm7axSFQxIot+utFAbAjeLys+MTgwHabLj5q1KFQM+k8fs2IAx/1CJwtl/OgwFajHp8qhJGgJHjsnHYDUOC2m50OJDhjcVwwH');
            audio.volume = 0.3;
            audio.play().catch(e => console.log('Audio play failed:', e));
        } catch (error) {
            console.log('Audio not supported');
        }
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            this.updateModelSelector(data.models);
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    updateModelSelector(models) {
        const modelSelector = document.querySelector('.model-selector');
        if (!modelSelector) return;
        
        modelSelector.innerHTML = '';
        
        models.forEach(model => {
            const optionHtml = `
                <label class="model-option">
                    <input type="radio" name="model" value="${model.name}" ${model.name === this.currentModel ? 'checked' : ''}>
                    <span>${model.description}</span>
                </label>
            `;
            modelSelector.insertAdjacentHTML('beforeend', optionHtml);
        });
        
        // Re-attach event listeners
        document.querySelectorAll('input[name="model"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                this.updateModelInfo(e.target.value);
            });
        });
    }

    updateModelInfo(modelName) {
        // Update model info display if exists
        const modelInfo = document.getElementById('modelInfo');
        if (modelInfo) {
            const descriptions = {
                'yolov8n': 'Fast detection with good accuracy. Best for real-time applications.',
                'yolov8s': 'Balanced speed and accuracy. Good for general use.',
                'best': 'Highest accuracy for fire/smoke detection. Slower but most precise.'
            };
            modelInfo.textContent = descriptions[modelName] || 'Model information not available';
        }
    }

    async loadDetectionHistory() {
        try {
            const response = await fetch('/api/history?limit=10');
            const data = await response.json();
            this.detectionHistory = data.history || [];
            this.updateHistoryDisplay();
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }

    addToHistory(result) {
        this.detectionHistory.unshift({
            timestamp: new Date().toISOString(),
            detection_count: result.detection_count,
            risk_level: result.risk_level,
            processing_time: result.processing_time
        });
        
        // Keep only last 50 records in memory
        this.detectionHistory = this.detectionHistory.slice(0, 50);
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const historyContainer = document.getElementById('historyContainer');
        if (!historyContainer || this.detectionHistory.length === 0) return;
        
        const historyHtml = this.detectionHistory.slice(0, 5).map(record => `
            <div class="history-item">
                <div class="history-time">${new Date(record.timestamp).toLocaleString()}</div>
                <div class="history-details">
                    ${record.detection_count} detections | Risk: ${record.risk_level} | ${record.processing_time}s
                </div>
            </div>
        `).join('');
        
        historyContainer.innerHTML = historyHtml;
    }

    setupWebSocket() {
        // Setup WebSocket for real-time updates if server supports it
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onerror = () => {
                console.log('WebSocket not available, using polling instead');
            };
        } catch (error) {
            console.log('WebSocket not supported');
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'detection_update') {
            this.showAlert('info', `New detection: ${data.message}`);
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fireDetectionApp = new FireDetectionApp();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInDown {
        from { transform: translateY(-100%); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes slideOutUp {
        from { transform: translateY(0); opacity: 1; }
        to { transform: translateY(-100%); opacity: 0; }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stat-updated {
        animation: pulse 0.5s ease-in-out;
        color: var(--accent-color) !important;
    }
    
    .alert {
        position: relative;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .alert-close {
        margin-left: auto;
        background: none;
        border: none;
        color: inherit;
        font-size: 1.2rem;
        cursor: pointer;
        padding: 0;
        width: 24px;
        height: 24px;
    }
    
    .history-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        font-size: 0.9rem;
    }
    
    .history-time {
        color: var(--text-gray);
        font-size: 0.8rem;
    }
`;
document.head.appendChild(style);
