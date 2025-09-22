// VERA - Voice-Enabled Recovery Assistant
// Frontend JavaScript Application

class VERAApp {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.workletNode = null;
        
        this.isRecording = false;
        this.currentStatus = 'idle';
        this.transcriptVisible = false;
        this.currentTranscript = '';
        
        // DOM elements
        this.elements = {};
        this.initializeElements();
        
        // Event listeners
        this.initializeEventListeners();
        
        // Check system health on load
        this.checkSystemHealth();
    }
    
    initializeElements() {
        const elementIds = [
            'setupSection', 'callSection', 'completeSection',
            'honorific', 'patientName', 'voiceSelect', 'voiceRate', 'scenarioSelect',
            'startBtn', 'endCallBtn', 'downloadBtn', 'newCallBtn',
            'statusIndicator', 'statusText', 'progressFill', 'progressText',
            'ttsPlayer', 'healthStatus', 'healthDot', 'healthText',
            'errorModal', 'modalClose', 'modalOk', 'errorMessage',
            'transcriptContainer', 'toggleTranscript', 'transcriptMessages', 'currentTranscriptText'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
        // Mic meter inner elements
        this.elements.micMeter = document.getElementById('micMeter');
        this.elements.micLevel = document.getElementById('micLevel');
    }
    
    initializeEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.startCall());
        this.elements.endCallBtn.addEventListener('click', () => this.endCall());
        this.elements.downloadBtn.addEventListener('click', () => this.downloadTranscript());
        this.elements.newCallBtn.addEventListener('click', () => this.startNewCall());
        
        // Modal controls
        this.elements.modalClose.addEventListener('click', () => this.hideModal());
        this.elements.modalOk.addEventListener('click', () => this.hideModal());
        
        // Transcript toggle
        this.elements.toggleTranscript.addEventListener('change', () => this.toggleTranscript());
        
        // Form validation
        this.elements.patientName.addEventListener('input', () => this.validateForm());
        
        // Initial form validation
        this.validateForm();
        
        // Initialize transcript visibility based on checkbox
        this.transcriptVisible = this.elements.toggleTranscript.checked;
        if (this.elements.transcriptContainer) {
            this.elements.transcriptContainer.style.display = this.transcriptVisible ? 'block' : 'none';
            console.log('Transcript visibility initialized:', this.transcriptVisible);
            
            // Add initial message to transcript
            if (this.transcriptVisible && this.elements.transcriptMessages) {
                this.elements.transcriptMessages.innerHTML = '<div class="transcript-message ai-question"><div class="transcript-message-header">AI: Ready</div><div class="transcript-message-text">Waiting for conversation to start...</div></div>';
            }
        } else {
            console.error('Transcript container not found!');
        }
    }
    
    validateForm() {
        const isValid = this.elements.patientName.value.trim().length > 0;
        this.elements.startBtn.disabled = !isValid;
    }
    
    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            
            this.updateHealthStatus(health);
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateHealthStatus({ 
                status: 'error', 
                message: 'System unavailable' 
            });
        }
    }
    
    updateHealthStatus(health) {
        const { healthDot, healthText } = this.elements;
        
        healthDot.className = 'status-dot';
        
        if (health.status === 'healthy') {
            healthDot.classList.add('healthy');
            healthText.textContent = 'System ready';
        } else if (health.status === 'degraded') {
            healthDot.classList.add('warning');
            healthText.textContent = health.message || 'System degraded';
        } else {
            healthDot.classList.add('error');
            healthText.textContent = health.message || 'System error';
        }
    }
    
    async startCall() {
        try {
            this.setStatus('Preparing call...');
            this.showSection('call');
            
            // Gather form data
            const callData = {
                honorific: this.elements.honorific.value,
                patient_name: this.elements.patientName.value.trim(),
                voice: this.elements.voiceSelect.value,
                rate: parseFloat(this.elements.voiceRate.value),
                scenario: this.elements.scenarioSelect.value
            };
            
            // Start session
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(callData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            
            // Initialize audio
            await this.initializeAudio();
            
            // Connect WebSocket
            await this.connectWebSocket();
            
        } catch (error) {
            console.error('Failed to start call:', error);
            this.showError('Failed to start call: ' + error.message);
            this.showSection('setup');
        }
    }
    
    async initializeAudio() {
        try {
            this.setStatus('Requesting microphone access...');
            
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context
            this.audioContext = new AudioContext({ sampleRate: 16000 });
            
            // Load and create audio worklet
            await this.audioContext.audioWorklet.addModule('/static/pcm-worklet.js');
            
            // Create worklet node
            this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-processor');
            
            // Connect audio pipeline
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.workletNode);
            
            // Handle audio data from worklet
            this.workletNode.port.onmessage = (event) => {
                const msg = event.data;
                if (msg && msg.type === 'level') {
                    // Update UI mic meter
                    this.updateMicLevel(msg.value);
                    return;
                }
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN && this.isRecording) {
                    this.websocket.send(msg);
                }
            };
            
            this.setStatus('Audio initialized');
            // Show mic meter when recording
            this.elements.micMeter.style.display = 'block';
            
        } catch (error) {
            throw new Error('Microphone access denied or unavailable');
        }
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.setStatus('Connecting to server...');
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/audio/${this.sessionId}`;
            
            this.websocket = new WebSocket(wsUrl);
            this.websocket.binaryType = 'arraybuffer';
            
            this.websocket.onopen = () => {
                this.setStatus('Connected - Starting conversation...');
                this.isRecording = true;
                // Show transcript by default when call starts
                this.transcriptVisible = true;
                this.elements.transcriptContainer.style.display = 'block';
                this.elements.toggleTranscript.textContent = 'Hide';
                this.elements.transcriptMessages.innerHTML = ''; // Clear previous transcript
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    // Received TTS audio
                    this.playTTSAudio(event.data);
                } else {
                    // Received text message
                    try {
                        const message = JSON.parse(event.data);
                        if (message.type === 'text_message') {
                            // Graceful degradation: show text message instead of audio
                            this.showTextMessage(message.message, message.prompt);
                        } else if (message.type === 'progress_update') {
                            // Update progress bar
                            console.log('Received progress update:', message.progress);
                            console.log('Progress data:', JSON.stringify(message.progress, null, 2));
                            this.updateProgressFromMessage(message.progress);
                        } else if (message.type === 'transcript_update') {
                            // Update live transcript
                            this.addTranscriptMessage(message.text, message.confidence, message.is_final);
                        } else if (message.type === 'ai_question') {
                            // Add AI question to transcript
                            this.addAIQuestion(message.text, message.question_key);
                        } else if (message.error) {
                            console.error('Server error:', message.error);
                            this.showError('Server error: ' + message.error);
                        }
                    } catch (e) {
                        console.log('Server message:', event.data);
                    }
                }
            };
            
            this.websocket.onclose = (event) => {
                this.isRecording = false;
                
                if (event.code === 1000) {
                    // Normal closure - call completed
                    this.setStatus('Call completed');
                    this.updateProgress(100);
                    this.showSection('complete');
                } else {
                    // Unexpected closure
                    console.error('WebSocket closed unexpectedly:', event.code, event.reason);
                    if (this.currentStatus !== 'ending') {
                        this.showError('Connection lost: ' + (event.reason || 'Unknown error'));
                        this.showSection('setup');
                    }
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(new Error('Failed to connect to server'));
            };
            
            // Timeout for connection
            setTimeout(() => {
                if (this.websocket.readyState !== WebSocket.OPEN) {
                    reject(new Error('Connection timeout'));
                }
            }, 10000);
        });
    }
    
    playTTSAudio(audioData) {
        try {
            // Validate audio data
            if (!audioData || audioData.byteLength === 0) {
                console.error('Received empty or invalid audio data');
                return;
            }
            
            // Validate WAV header
            const dataView = new DataView(audioData);
            if (dataView.getUint32(0, true) !== 0x46464952 || // "RIFF"
                dataView.getUint32(8, true) !== 0x45564157) {  // "WAVE"
                console.warn('Received audio data may not be valid WAV format');
            }
            
            console.log(`Playing TTS audio: ${audioData.byteLength} bytes`);
            
            const blob = new Blob([audioData], { type: 'audio/wav' });
            const url = URL.createObjectURL(blob);
            
            // Store previous recording state
            const wasRecording = this.isRecording;
            this.isRecording = false;
            
            // Set up audio element
            this.elements.ttsPlayer.src = url;
            this.elements.ttsPlayer.volume = 1.0;
            
            // Handle playback events
            this.elements.ttsPlayer.onloadstart = () => {
                console.log('TTS audio loading started');
                this.setStatus('AI is speaking...');
            };
            
            this.elements.ttsPlayer.oncanplay = () => {
                console.log('TTS audio ready to play');
            };
            
            this.elements.ttsPlayer.onplay = () => {
                console.log('TTS audio playback started');
            };
            
            this.elements.ttsPlayer.onpause = () => {
                console.log('TTS audio playback paused');
            };
            
            this.elements.ttsPlayer.onerror = (error) => {
                console.error('TTS audio playback error:', error);
                this.setStatus('Audio playback error');
                // Clean up and resume recording
                URL.revokeObjectURL(url);
                this.resumeRecording(wasRecording);
            };
            
            this.elements.ttsPlayer.onended = () => {
                console.log('TTS audio playback completed');
                URL.revokeObjectURL(url);
                this.setStatus('Listening...');
                this.resumeRecording(wasRecording);
            };
            
            // Start playback
            const playPromise = this.elements.ttsPlayer.play();
            
            if (playPromise !== undefined) {
                playPromise.catch(error => {
                    console.error('Failed to start TTS playback:', error);
                    this.setStatus('Audio playback failed');
                    URL.revokeObjectURL(url);
                    this.resumeRecording(wasRecording);
                });
            }
            
        } catch (error) {
            console.error('Failed to play TTS audio:', error);
            this.setStatus('Audio playback error');
        }
    }
    
    resumeRecording(wasRecording) {
        // Resume recording after TTS finishes
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.isRecording = wasRecording || true;
            console.log('Resumed audio recording');
        }
    }
    
    showTextMessage(message, prompt) {
        // Show text message as fallback when TTS fails
        console.log('Showing text message:', message);
        if (prompt) {
            console.log('Original prompt:', prompt);
        }
        
        // Update status to show the message
        this.setStatus(message);
        
        // You could also show this in a modal or dedicated text area
        // For now, we'll just log it and update the status
        setTimeout(() => {
            this.setStatus('Listening...');
        }, 3000); // Show message for 3 seconds then return to listening
    }
    
    toggleTranscript() {
        this.transcriptVisible = this.elements.toggleTranscript.checked;
        this.elements.transcriptContainer.style.display = this.transcriptVisible ? 'block' : 'none';
    }
    
    addAIQuestion(text, questionKey) {
        if (!this.transcriptVisible) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'transcript-message ai-question';
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'transcript-message-header';
        headerDiv.textContent = `AI: ${questionKey === 'greeting' ? 'Greeting' : 'Question'}`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'transcript-message-text';
        textDiv.textContent = text;
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(textDiv);
        
        this.elements.transcriptMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.elements.transcriptMessages.scrollTop = this.elements.transcriptMessages.scrollHeight;
    }
    
    addTranscriptMessage(text, confidence, isFinal = true) {
        if (!this.transcriptVisible) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `transcript-message ${isFinal ? 'final' : 'partial'}`;
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'transcript-message-header';
        headerDiv.textContent = `You: ${isFinal ? 'Final' : 'Partial'} (${Math.round(confidence * 100)}% confidence)`;
        
        const textDiv = document.createElement('div');
        textDiv.className = 'transcript-message-text';
        textDiv.textContent = text;
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(textDiv);
        
        this.elements.transcriptMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.elements.transcriptMessages.scrollTop = this.elements.transcriptMessages.scrollHeight;
    }
    
    updateCurrentTranscript(text, isProcessing = false) {
        if (!this.transcriptVisible) return;
        
        this.currentTranscript = text;
        this.elements.currentTranscriptText.textContent = text || 'Listening...';
        this.elements.currentTranscriptText.className = `transcript-text ${text ? (isProcessing ? 'processing' : '') : 'listening'}`;
    }
    
    clearCurrentTranscript() {
        this.currentTranscript = '';
        this.updateCurrentTranscript('');
    }
    
    endCall() {
        this.currentStatus = 'ending';
        this.setStatus('Ending call...');
        
        if (this.websocket) {
            this.websocket.close(1000, 'User ended call');
        }
        
        this.cleanup();
        this.showSection('setup');
    }
    
    cleanup() {
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        // Clear worklet
        this.workletNode = null;
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.isRecording = false;
        this.currentStatus = 'idle';
        this.updateProgress(0);
        if (this.elements.micMeter) {
            this.elements.micMeter.style.display = 'none';
        }
        
        // Clear transcript
        this.elements.transcriptMessages.innerHTML = '';
        this.clearCurrentTranscript();
    }

    updateMicLevel(level) {
        if (!this.elements.micLevel) return;
        const pct = Math.max(0, Math.min(100, Math.round(level * 100)));
        this.elements.micLevel.style.width = pct + '%';
    }
    
    async downloadTranscript() {
        if (!this.sessionId) {
            this.showError('No session available for download');
            return;
        }
        
        try {
            this.setStatus('Preparing download...');
            
            const response = await fetch(`/api/download/${this.sessionId}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            // Create download link
            const a = document.createElement('a');
            a.href = url;
            a.download = `vera_session_${this.sessionId}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error('Download failed:', error);
            this.showError('Failed to download transcript: ' + error.message);
        }
    }
    
    startNewCall() {
        this.cleanup();
        this.sessionId = null;
        this.showSection('setup');
        this.elements.patientName.focus();
    }
    
    setStatus(status) {
        this.elements.statusText.textContent = status;
        this.currentStatus = status.toLowerCase();
        
        // Update indicator color based on status
        const indicator = this.elements.statusIndicator;
        indicator.className = 'status-indicator';
        
        if (status.includes('error') || status.includes('failed')) {
            indicator.classList.add('error');
        } else if (status.includes('listening')) {
            indicator.classList.add('listening');
        } else if (status.includes('speaking')) {
            indicator.classList.add('speaking');
        }
    }
    
    updateProgress(percent) {
        console.log(`Updating progress bar to: ${percent}%`);
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressText.textContent = `${Math.round(percent)}% Complete`;
    }
    
    updateProgressFromMessage(progress) {
        // Update progress bar with detailed information from backend
        const percent = Math.round(progress.progress_percent || 0);
        const currentQuestion = progress.answered_questions + 1;
        const totalQuestions = progress.total_questions;
        
        console.log('Updating progress bar elements:', {
            progressFill: this.elements.progressFill,
            progressText: this.elements.progressText,
            percent: percent,
            currentQuestion: currentQuestion,
            totalQuestions: totalQuestions
        });
        
        // Update progress bar
        if (this.elements.progressFill) {
            this.elements.progressFill.style.width = `${percent}%`;
            console.log(`Set progress bar width to: ${percent}%`);
        } else {
            console.error('Progress fill element not found!');
        }
        
        // Update progress text with detailed format
        if (this.elements.progressText) {
            this.elements.progressText.textContent = `Question ${currentQuestion}/${totalQuestions}, ${percent}% Complete`;
            console.log(`Set progress text to: Question ${currentQuestion}/${totalQuestions}, ${percent}% Complete`);
        } else {
            console.error('Progress text element not found!');
        }
        
        // Log progress details for debugging
        console.log(`Progress update: Question ${currentQuestion}/${totalQuestions}, ${percent}% (${progress.answered_questions} answered)`);
        
        // Update status with more detailed information
        if (progress.finished) {
            this.setStatus('Conversation completed');
        } else if (progress.answered_questions > 0) {
            this.setStatus(`Question ${currentQuestion} of ${totalQuestions}`);
        }
    }
    
    showSection(section) {
        // Hide all sections
        this.elements.setupSection.style.display = 'none';
        this.elements.callSection.style.display = 'none';
        this.elements.completeSection.style.display = 'none';
        
        // Show requested section
        const sectionElement = this.elements[section + 'Section'];
        if (sectionElement) {
            sectionElement.style.display = 'block';
        }
    }
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.style.display = 'flex';
    }
    
    hideModal() {
        this.elements.errorModal.style.display = 'none';
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.veraApp = new VERAApp();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.veraApp) {
        window.veraApp.cleanup();
    }
});

