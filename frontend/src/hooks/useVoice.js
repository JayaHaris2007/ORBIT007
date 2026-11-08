import { useCallback, useEffect, useRef, useState } from 'react';
import {
    fetchTTSAudio,
    fetchVoiceSettings,
    getSharedAudioQueue,
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

    const abortControllerRef = useRef(null);
    const audioQueue = getSharedAudioQueue();

    // Connect queue callbacks to speaking state and listen for global audio stop
    useEffect(() => {
        if (!audioQueue) return;
        audioQueue.setCallbacks(
            () => setIsSpeaking(true),
            () => setIsSpeaking(false)
        );

        const onExternalStop = () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            audioQueue.stop();
            setIsSpeaking(false);
        };
        window.addEventListener('spidey:audio_stop', onExternalStop);
        return () => window.removeEventListener('spidey:audio_stop', onExternalStop);
    }, [audioQueue]);

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
        if (audioQueue) {
            audioQueue.stop();
        }
        setIsSpeaking(false);
        stopVoiceTurn().catch(() => {});
    }, [audioQueue]);

    const speak = useCallback(
        async (text) => {
            if (!speakEnabled || !text || !text.trim() || !audioQueue) return;

            window.dispatchEvent(new CustomEvent('spidey:audio_stop'));

            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            abortControllerRef.current = new AbortController();
            const signal = abortControllerRef.current.signal;

            audioQueue.stop();
            const currentEpoch = audioQueue.epoch;

            const sentences = splitSentencesClient(text);
            if (!sentences.length) return;

            setIsSpeaking(true);

            try {
                for (const sentence of sentences) {
                    if (signal.aborted || currentEpoch !== audioQueue.epoch) break;
                    const wavBuffer = await fetchTTSAudio(sentence, signal);
                    if (signal.aborted || currentEpoch !== audioQueue.epoch) break;
                    await audioQueue.push(wavBuffer, currentEpoch);
                }
            } catch (err) {
                if (err.name !== 'AbortError') {
                    console.error('TTS playback stream error:', err);
                }
            }
        },
        [speakEnabled, audioQueue]
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
        audioQueue
    };
}
