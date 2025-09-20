// PCM Audio Worklet for VERA
// Converts Float32 audio to Int16 PCM and sends 20-30ms frames

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // Configuration
        this.sampleRate = 16000;  // Target sample rate
        this.frameSize = Math.floor(this.sampleRate * 0.025); // 25ms frames
        
        // Buffers
        this.inputBuffer = new Float32Array(this.frameSize);
        this.bufferIndex = 0;
        
        // Processing state
        this.isActive = true;
        
        console.log(`PCM Worklet initialized: ${this.frameSize} samples per frame at ${this.sampleRate}Hz`);
    }
    
    process(inputs, outputs, parameters) {
        if (!this.isActive) {
            return false;
        }
        
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }
        
        // Get the first channel (mono)
        const inputChannel = input[0];
        if (!inputChannel) {
            return true;
        }
        
        // Compute simple level meter (RMS) for UI feedback
        let sumSquares = 0;
        for (let i = 0; i < inputChannel.length; i++) {
            const s = inputChannel[i];
            sumSquares += s * s;
        }
        const rms = Math.sqrt(sumSquares / inputChannel.length);
        const level = Math.min(1, rms * 3); // scale for visibility
        this.port.postMessage({ type: 'level', value: level });

        // Process each sample in the input
        for (let i = 0; i < inputChannel.length; i++) {
            // Add sample to buffer
            this.inputBuffer[this.bufferIndex] = inputChannel[i];
            this.bufferIndex++;
            
            // When buffer is full, send frame
            if (this.bufferIndex >= this.frameSize) {
                this.sendFrame();
                this.bufferIndex = 0;
            }
        }
        
        return true;
    }
    
    sendFrame() {
        // Convert Float32 to Int16 PCM
        const pcmFrame = new Int16Array(this.frameSize);
        
        for (let i = 0; i < this.frameSize; i++) {
            // Clamp to [-1, 1] range
            let sample = Math.max(-1, Math.min(1, this.inputBuffer[i]));
            
            // Convert to 16-bit signed integer
            pcmFrame[i] = Math.round(sample * 32767);
        }
        
        // Send as ArrayBuffer to main thread
        this.port.postMessage(pcmFrame.buffer, [pcmFrame.buffer]);
    }
}

// Register the processor
registerProcessor('pcm-processor', PCMProcessor);

