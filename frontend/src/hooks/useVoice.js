import { useCallback, useEffect, useRef, useState } from 'react';
import {
    AudioQueue,
    fetchTTSAudio,
    fetchVoiceSettings,
    splitSentencesClient,
    stopVoiceTurn,
    updateVoiceSettings
} from '../services/voiceService';

const SPEAK_LOCAL_KEY = 'spidey_speak_enabled';

export function useVoice() {
    const [speakEnabled, setSpeakEnabled] = useState(() => {
        return localStorage.getItem(SPEAK_LOCAL_KEY) === '1';
    });
    const [engineReady, setEngineReady] = useState(false);
    const [device, setDevice] = useState('');
    const [error, setError] = useState(null);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [cloudStt, setCloudStt] = useState(false);

    const audioQueueRef = useRef(null);
    const abortControllerRef = useRef(null);

    if (!audioQueueRef.current) {
        audioQueueRef.current = new AudioQueue();
    }

    // Connect queue callbacks to speaking state
    useEffect(() => {
        const queue = audioQueueRef.current;
        queue.onPlay = () => setIsSpeaking(true);
        queue.onIdle = () => setIsSpeaking(false);
    }, []);

    // Load initial settings from server
    useEffect(() => {
        let mounted = true;
        const loadSettings = async () => {
            try {
                const data = await fetchVoiceSettings();
                if (!mounted) return;
                setSpeakEnabled(Boolean(data.speak_enabled));
                setEngineReady(Boolean(data.engine_ready));
                setDevice(data.device || '');
                setError(data.error || null);
                setCloudStt(Boolean(data.allow_cloud_stt));
                localStorage.setItem(SPEAK_LOCAL_KEY, data.speak_enabled ? '1' : '0');
            } catch (err) {
                if (!mounted) return;
                console.warn('Voice settings not reachable:', err);
                setEngineReady(false);
            }
        };
        loadSettings();
        return () => {
            mounted = false;
        };
    }, []);

    const toggleSpeak = useCallback(async () => {
        const next = !speakEnabled;
        setSpeakEnabled(next);
        localStorage.setItem(SPEAK_LOCAL_KEY, next ? '1' : '0');
        try {
            const data = await updateVoiceSettings({ speak_enabled: next });
            setSpeakEnabled(Boolean(data.speak_enabled));
            setEngineReady(Boolean(data.engine_ready));
        } catch (err) {
            console.error('Failed to update speak setting on backend:', err);
        }
    }, [speakEnabled]);

    const stop = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        if (audioQueueRef.current) {
            audioQueueRef.current.stop();
        }
        setIsSpeaking(false);
        stopVoiceTurn().catch(() => {});
    }, []);

    const speak = useCallback(
        async (text) => {
            if (!speakEnabled || !text || !text.trim()) return;

            // Stop any existing playback first
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            abortControllerRef.current = new AbortController();
            const signal = abortControllerRef.current.signal;

            const queue = audioQueueRef.current;
            queue.stop();
            const currentEpoch = queue.epoch;

            const sentences = splitSentencesClient(text);
            if (!sentences.length) return;

            setIsSpeaking(true);

            try {
                for (const sentence of sentences) {
                    if (signal.aborted || currentEpoch !== queue.epoch) break;
                    const wavBuffer = await fetchTTSAudio(sentence, signal);
                    if (signal.aborted || currentEpoch !== queue.epoch) break;
                    await queue.push(wavBuffer, currentEpoch);
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.error('TTS playback stream error:', err);
                }
            }
        },
        [speakEnabled]
    );

    return {
        speakEnabled,
        engineReady,
        device,
        error,
        isSpeaking,
        cloudStt,
        toggleSpeak,
        speak,
        stop,
        audioQueue: audioQueueRef.current
    };
}
