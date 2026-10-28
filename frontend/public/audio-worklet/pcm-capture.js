/**
 * AudioWorklet processor that captures mic audio, downsamples/decimates to 16 kHz mono if needed,
 * converts samples to 16-bit signed PCM (s16le), and buffers 100 ms frames (1600 samples)
 * to send back to the main thread for WebSocket transmission.
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Int16Array(1600); // 100 ms at 16,000 Hz
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0]; // Mono channel (Float32Array)

        // If sampleRate is 16000, 1:1 copy. Otherwise simple linear decimation.
        const step = sampleRate / 16000;
        let inputIdx = 0;

        while (inputIdx < channelData.length) {
            const sample = Math.max(-1, Math.min(1, channelData[Math.floor(inputIdx)]));
            // Convert float (-1.0 .. 1.0) to 16-bit signed int (-32768 .. 32767)
            this.buffer[this.bufferIndex++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;

            if (this.bufferIndex >= this.buffer.length) {
                // Post a copy of the filled 100 ms frame
                const frame = this.buffer.slice(0);
                this.port.postMessage(frame.buffer, [frame.buffer]);
                this.bufferIndex = 0;
            }

            inputIdx += step;
        }

        return true;
    }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor);

