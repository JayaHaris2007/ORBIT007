import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE_URL } from '../services/api';
import { AudioQueue, stopVoiceTurn } from '../services/voiceService';

export function useVoiceSession({ onTranscript = null, onReply = null, audioQueue = null }) {
    const [state, setState] = useState('idle'); // idle | listening | transcribing | thinking | speaking
    const [transcript, setTranscript] = useState('');
    const [reply, setReply] = useState('');
    const [error, setError] = useState(null);
    const [mode, setMode] = useState('listen'); // listen | live
    const [timing, setTiming] = useState({ speechEnd: 0, transcript: 0, firstSentence: 0, firstAudio: 0, total: 0 });

    const wsRef = useRef(null);
    const micStreamRef = useRef(null);
    const audioContextRef = useRef(null);
    const workletNodeRef = useRef(null);
    const queueRef = useRef(audioQueue || new AudioQueue());
    const timingRef = useRef({ speechEnd: 0, transcript: 0, firstSentence: 0, firstAudio: 0 });
    const nextExpectedAudioSeqRef = useRef(null);

    const getWsUrl = () => {
        const base = API_BASE_URL.replace(/^http/, 'ws');
        return `${base}/ws/voice`;
    };

    const cleanupAudioCapture = useCallback(() => {
        if (workletNodeRef.current) {
            try {
                workletNodeRef.current.disconnect();
                workletNodeRef.current.port.close();
            } catch {}
            workletNodeRef.current = null;
        }
        if (micStreamRef.current) {
            micStreamRef.current.getTracks().forEach((track) => track.stop());
            micStreamRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            try {
                audioContextRef.current.close();
            } catch {}
            audioContextRef.current = null;
        }
    }, []);

    const stopSession = useCallback(() => {
        cleanupAudioCapture();
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            try {
                wsRef.current.send(JSON.stringify({ type: 'stop' }));
                wsRef.current.close();
            } catch {}
        }
        wsRef.current = null;
        queueRef.current.stop();
        stopVoiceTurn().catch(() => {});
        setState('idle');
        setTranscript('');
    }, [cleanupAudioCapture]);

    const interrupt = useCallback(() => {
        queueRef.current.stop();
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
        }
    }, []);

    const startSession = useCallback(
        async (sessionMode = 'listen', chatId = null) => {
            stopSession();
            setMode(sessionMode);
            setError(null);
            setTranscript('');
            setReply('');
            timingRef.current = { speechEnd: 0, transcript: 0, firstSentence: 0, firstAudio: 0 };

            try {
                // 1. Get mic permission with echo cancellation
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        channelCount: 1
                    }
                });
                micStreamRef.current = stream;

                // 2. Setup AudioContext (prefer 16 kHz if supported)
                let audioCtx;
                try {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                } catch {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                }
                audioContextRef.current = audioCtx;

                // 3. Load AudioWorklet
                await audioCtx.audioWorklet.addModule('/audio-worklet/pcm-capture.js');
                const source = audioCtx.createMediaStreamSource(stream);
                const workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture-processor');
                workletNodeRef.current = workletNode;
                source.connect(workletNode);

                // 4. Connect WebSocket
                const ws = new WebSocket(getWsUrl());
                ws.binaryType = 'arraybuffer';
                wsRef.current = ws;

                ws.onopen = () => {
                    ws.send(JSON.stringify({ type: 'start', mode: sessionMode, chat_id: chatId }));
                    setState('listening');
                };

                workletNode.port.onmessage = (event) => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(event.data); // Binary PCM frame (100 ms)
                    }
                };

                ws.onmessage = async (event) => {
                    if (typeof event.data === 'string') {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.type === 'state') {
                                setState(data.value);
                                if (data.value === 'transcribing') {
                                    timingRef.current.speechEnd = performance.now();
                                }
                            } else if (data.type === 'transcript') {
                                setTranscript(data.text);
                                const now = performance.now();
                                timingRef.current.transcript = now;
                                onTranscript?.(data.text);
                            } else if (data.type === 'reply') {
                                setReply(data.text);
                                timingRef.current.firstSentence = performance.now();
                                onReply?.(data.text);
                            } else if (data.type === 'audio') {
                                nextExpectedAudioSeqRef.current = data;
                            } else if (data.type === 'error') {
                                setError(data.message);
                            }
                        } catch (err) {
                            console.error('WS JSON parse error:', err);
                        }
                    } else if (event.data instanceof ArrayBuffer) {
                        // Binary WAV frame received
                        const audioMeta = nextExpectedAudioSeqRef.current;
                        nextExpectedAudioSeqRef.current = null;
                        if (!timingRef.current.firstAudio && timingRef.current.speechEnd) {
                            const now = performance.now();
                            timingRef.current.firstAudio = now;
                            const t = timingRef.current;
                            setTiming({
                                speechEnd: t.speechEnd,
                                transcript: Math.round(t.transcript - t.speechEnd),
                                firstSentence: Math.round(t.firstSentence - t.transcript),
                                firstAudio: Math.round(t.firstAudio - t.firstSentence),
                                total: Math.round(t.firstAudio - t.speechEnd)
                            });
                        }
                        queueRef.current.push(event.data, queueRef.current.epoch);
                    }
                };

                ws.onerror = (err) => {
                    console.error('Voice WebSocket error:', err);
                    setError('Voice WebSocket connection error.');
                };

                ws.onclose = () => {
                    if (sessionMode !== 'live') {
                        cleanupAudioCapture();
                        setState('idle');
                    }
                };
            } catch (err) {
                console.error('Failed to initialize mic/voice session:', err);
                setError(err.message || 'Microphone access denied or unavailable.');
                cleanupAudioCapture();
                setState('idle');
            }
        },
        [cleanupAudioCapture, stopSession, onTranscript, onReply]
    );

    useEffect(() => {
        return () => {
            stopSession();
        };
    }, [stopSession]);

    return {
        state,
        transcript,
        reply,
        error,
        mode,
        timing,
        startSession,
        stopSession,
        interrupt,
        isIdle: state === 'idle'
    };
}
