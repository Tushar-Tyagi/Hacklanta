"""
Audio Match Agent: Local BPM detection and energy analysis with Librosa,
confidence scoring, and OpenRouter API fallback for music genre and mood matching.
"""

import time
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Try importing librosa for audio analysis
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from agents.base_agent import BaseAgent, ProcessingMode, AgentResponse
from openrouter_client import OpenRouterClient
from openrouter_client.exceptions import OpenRouterError, APIError


class BPMRange(Enum):
    """BPM range categories."""
    VERY_SLOW = "very_slow"      # < 70 BPM
    SLOW = "slow"              # 70-90 BPM
    MODERATE = "moderate"      # 90-120 BPM
    FAST = "fast"              # 120-150 BPM
    VERY_FAST = "very_fast"    # > 150 BPM


class EnergyLevel(Enum):
    """Energy level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class TimbreType(Enum):
    """Timbre characteristics."""
    DARK = "dark"
    BRIGHT = "bright"


@dataclass
class AudioFeatures:
    """Extracted audio features from local analysis."""
    bpm: float
    bpm_confidence: float
    energy: float
    energy_confidence: float
    rms_amplitude: float
    spectral_centroid: float
    zero_crossing_rate: float
    spectral_rolloff: float
    tempo_category: Optional[BPMRange] = None
    energy_level: Optional[EnergyLevel] = None
    timbre: Optional[TimbreType] = None
    duration_seconds: float = 0.0
    local_analysis_complete: bool = False
    analysis_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "energy": self.energy,
            "energy_confidence": self.energy_confidence,
            "rms_amplitude": self.rms_amplitude,
            "spectral_centroid": self.spectral_centroid,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_rolloff": self.spectral_rolloff,
            "tempo_category": self.tempo_category.value if self.tempo_category else None,
            "energy_level": self.energy_level.value if self.energy_level else None,
            "timbre": self.timbre.value if self.timbre else None,
            "duration_seconds": self.duration_seconds,
            "local_analysis_complete": self.local_analysis_complete,
            "analysis_error": self.analysis_error,
        }


@dataclass
class GenreMoodResult:
    """Genre and mood classification result."""
    genres: List[str]
    moods: List[str]
    confidence: float
    source: str  # "local" or "api"
    model_used: Optional[str] = None
    processing_time_ms: int = 0
    fallback_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genres": self.genres,
            "moods": self.moods,
            "confidence": self.confidence,
            "source": self.source,
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
            "fallback_triggered": self.fallback_triggered,
        }


class AudioMatchAgent(BaseAgent):
    """
    Audio Match Agent for music analysis with local BPM detection and API fallback.

    Features:
    - Local BPM detection using Librosa
    - Energy analysis (RMS amplitude, spectral features)
    - Confidence scoring for local analysis
    - OpenRouter API fallback for genre/mood classification
    - Hybrid processing mode (local + API)

    Usage:
        agent = AudioMatchAgent(
            api_key="your-api-key",
            mode=ProcessingMode.HYBRID,
        )
        
        # Analyze audio file
        audio_features = agent.analyze_audio_file("path/to/audio.mp3")
        
        # Get genre/mood classification
        result = agent.classify_genre_mood("path/to/audio.mp3")
        print(result.genres, result.moods, result.confidence)
    """

    DEFAULT_API_MODEL = "openai/gpt-4o-mini"
    MIN_BPM = 40.0
    MAX_BPM = 220.0
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.5

    # Genre mapping based on BPM and energy
    BPM_GENRE_MAP = {
        BPMRange.VERY_SLOW: ["ambient", "downtempo", "classical", "drone"],
        BPMRange.SLOW: ["soul", "r&b", "jazz", "folk", "acoustic"],
        BPMRange.MODERATE: ["pop", "rock", "dance", "house", "indie"],
        BPMRange.FAST: ["techno", "trance", "drum & bass", "edm"],
        BPMRange.VERY_FAST: ["hardcore", "hardstyle", "gabber", "speedcore"],
    }

    ENERGY_MOOD_MAP = {
        EnergyLevel.HIGH: ["energetic", "intense", "aggressive"],
        EnergyLevel.MODERATE: ["happy", "uplifting", "neutral"],
        EnergyLevel.LOW: ["calm", "peaceful", "melancholic"],
    }

    def __init__(
        self,
        api_key: str,
        mode: ProcessingMode = ProcessingMode.HYBRID,
        confidence_threshold: float = 0.7,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        max_retries: int = 3,
        enable_genre_fallback: bool = True,
        local_analyze_duration: float = 30.0,
        api_model: str = DEFAULT_API_MODEL,
    ):
        """
        Initialize the Audio Match Agent.

        Args:
            api_key: OpenRouter API key
            mode: Processing mode (LOCAL_ONLY, API_ONLY, HYBRID, API_FALLBACK)
            confidence_threshold: Minimum confidence for local result acceptance
            enable_cache: Enable response caching
            cache_ttl_seconds: Cache time-to-live in seconds
            max_retries: Maximum API retry attempts
            enable_genre_fallback: Enable API fallback for genre/mood classification
            local_analyze_duration: Max audio duration to analyze locally (seconds)
            api_model: OpenRouter model identifier
        """
        self.api_key = api_key
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.enable_genre_fallback = enable_genre_fallback
        self.local_analyze_duration = local_analyze_duration

        # Track librosa availability
        self._librosa_available = LIBROSA_AVAILABLE

        # Initialize OpenRouter client
        self._client = OpenRouterClient(
            api_key=api_key,
            enable_caching=enable_cache,
            cache_ttl_seconds=cache_ttl_seconds,
            max_retries=max_retries,
        )

        # Statistics
        self._stats = {
            "audio_files_analyzed": 0,
            "local_bpm_detections": 0,
            "local_energy_analyses": 0,
            "api_genre_requests": 0,
            "cache_hits": 0,
            "fallback_count": 0,
        }

    def _analyze_audio_librosa(self, audio_path: str) -> AudioFeatures:
        """
        Analyze audio file using Librosa for BPM and energy features.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioFeatures with extracted features
        """
        try:
            # Load audio file with limited duration for efficiency
            duration = min(
                librosa.get_duration(filename=audio_path),
                self.local_analyze_duration
            )

            y, sr = librosa.load(audio_path, duration=duration, sr=22050)

            # Get BPM using tempo estimation
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo)

            # Adjust BPM if needed (librosa sometimes returns 2x or 0.5x actual)
            if bpm < self.MIN_BPM:
                bpm = bpm * 2  # Double if too slow (common issue)
            if bpm > self.MAX_BPM:
                bpm = bpm / 2  # Halve if too fast

            # Calculate BPM confidence based on beat detection strength
            beat_strength = np.mean(librosa.beat.beat_track(y=y, sr=sr, units="strength"))
            bpm_confidence = min(1.0, beat_strength / 100.0) if beat_strength else 0.5

            # Energy analysis - RMS amplitude
            rms = librosa.feature.rms(y=y)
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))

            # Calculate energy confidence based on RMS consistency
            rms_cv = rms_std / (rms_mean + 0.001)  # Coefficient of variation
            energy_confidence = max(0.0, 1.0 - (rms_cv / 2))  # Lower CV = higher confidence

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            sc_mean = float(np.mean(spectral_centroid))

            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = float(np.mean(zcr))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            sr_mean = float(np.mean(spectral_rolloff))

            # Categorize tempo
            if bpm < 70:
                tempo_category = BPMRange.VERY_SLOW
            elif bpm < 90:
                tempo_category = BPMRange.SLOW
            elif bpm < 120:
                tempo_category = BPMRange.MODERATE
            elif bpm < 150:
                tempo_category = BPMRange.FAST
            else:
                tempo_category = BPMRange.VERY_FAST

            # Energy level
            if rms_mean < 0.1:
                energy_level = EnergyLevel.LOW
            elif rms_mean < 0.3:
                energy_level = EnergyLevel.MODERATE
            else:
                energy_level = EnergyLevel.HIGH

            # Timbre based on spectral centroid
            sc_normalized = sc_mean / (sr / 2)  # Normalize by Nyquist frequency
            if sc_normalized > 0.5:
                timbre = TimbreType.BRIGHT
            else:
                timbre = TimbreType.DARK

            return AudioFeatures(
                bpm=bpm,
                bpm_confidence=bpm_confidence,
                energy=rms_mean,
                energy_confidence=energy_confidence,
                rms_amplitude=rms_mean,
                spectral_centroid=sc_mean,
                zero_crossing_rate=zcr_mean,
                spectral_rolloff=sr_mean,
                tempo_category=tempo_category,
                energy_level=energy_level,
                timbre=timbre,
                duration_seconds=duration,
                local_analysis_complete=True,
            )

        except Exception as e:
            return AudioFeatures(
                bpm=0.0,
                bpm_confidence=0.0,
                energy=0.0,
                energy_confidence=0.0,
                rms_amplitude=0.0,
                spectral_centroid=0.0,
                zero_crossing_rate=0.0,
                spectral_rolloff=0.0,
                local_analysis_complete=False,
                analysis_error=str(e),
            )

    def _local_genre_classification(
        self,
        features: AudioFeatures,
    ) -> GenreMoodResult:
        """
        Perform local genre/mood classification based on extracted features.

        Args:
            features: AudioFeatures from local analysis

        Returns:
            GenreMoodResult with classification
        """
        genres = []
        moods = []

        # Genre based on BPM category
        tempo_genres = self.BPM_GENRE_MAP.get(features.tempo_category, [])
        genres.extend(tempo_genres[:2])

        # Mood based on energy level
        energy_moods = self.ENERGY_MOOD_MAP.get(features.energy_level, [])
        moods.extend(energy_moods[:2])

        # Add timbre-influenced moods
        if features.timbre == TimbreType.BRIGHT:
            moods.append("bright")
        elif features.timbre == TimbreType.DARK:
            moods.append("dark")

        # Calculate confidence based on feature quality
        confidence = (
            features.bpm_confidence * 0.4 +
            features.energy_confidence * 0.3 +
            (1.0 if features.local_analysis_complete else 0.0) * 0.3
        )

        return GenreMoodResult(
            genres=genres[:4],
            moods=moods[:4],
            confidence=confidence,
            source="local",
        )

    def _api_genre_classification(
        self,
        features: AudioFeatures,
        audio_path: str,
    ) -> GenreMoodResult:
        """
        Use OpenRouter API for genre/mood classification.

        Args:
            features: AudioFeatures from local analysis
            audio_path: Path to audio file (for context)

        Returns:
            GenreMoodResult with API classification
        """
        start_time = time.time()

        # Build prompt with audio features
        prompt = f"""Based on the following audio features, classify the music into genres and moods:

Audio Features:
- BPM: {features.bpm:.1f}
- Tempo Category: {features.tempo_category.value if features.tempo_category else "unknown"}
- Energy Level: {features.energy_level.value if features.energy_level else "unknown"}
- Timbre: {features.timbre.value if features.timbre else "unknown"}
- Duration: {features.duration_seconds:.1f} seconds
- RMS Amplitude: {features.rms_amplitude:.4f}
- Spectral Centroid: {features.spectral_centroid:.1f} Hz
- Zero Crossing Rate: {features.zero_crossing_rate:.4f}

Respond in JSON format with exactly these fields:
{{"genres": ["genre1", "genre2"], "moods": ["mood1", "mood2"]}}

Provide 2-4 genres and 2-4 moods that best describe this audio."""

        messages = [
            {
                "role": "system",
                "content": "You are a music expert. Classify music into genres and moods based on audio features. Respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                temperature=0.3,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            content = response.get_content()
            processing_time = int((time.time() - start_time) * 1000)

            # Parse JSON response
            result_data = json.loads(content)

            genres = result_data.get("genres", [])
            moods = result_data.get("moods", [])

            # Validate and sanitize
            if not isinstance(genres, list):
                genres = [genres]
            if not isinstance(moods, list):
                moods = [moods]

            return GenreMoodResult(
                genres=genres[:4],
                moods=moods[:4],
                confidence=0.85,  # Higher confidence for API
                source="api",
                model_used=self.api_model,
                processing_time_ms=processing_time,
            )

        except (OpenRouterError, json.JSONDecodeError) as e:
            # Fall back to local result
            return self._local_genre_classification(features)

    def analyze_audio_file(self, audio_path: str) -> AudioFeatures:
        """
        Analyze an audio file to extract BPM and energy features.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioFeatures with extracted features
        """
        if not self._librosa_available:
            return AudioFeatures(
                bpm=0.0,
                bpm_confidence=0.0,
                energy=0.0,
                energy_confidence=0.0,
                rms_amplitude=0.0,
                spectral_centroid=0.0,
                zero_crossing_rate=0.0,
                spectral_rolloff=0.0,
                local_analysis_complete=False,
                analysis_error="Librosa not available",
            )

        try:
            features = self._analyze_audio_librosa(audio_path)
            return features

        except Exception as e:
            return AudioFeatures(
                bpm=0.0,
                bpm_confidence=0.0,
                energy=0.0,
                energy_confidence=0.0,
                rms_amplitude=0.0,
                spectral_centroid=0.0,
                zero_crossing_rate=0.0,
                spectral_rolloff=0.0,
                local_analysis_complete=False,
                analysis_error=str(e),
            )

    def classify_genre_mood(
        self,
        audio_path: str,
        force_mode: Optional[ProcessingMode] = None,
    ) -> GenreMoodResult:
        """
        Classify music into genres and moods.

        Args:
            audio_path: Path to the audio file
            force_mode: Force a specific processing mode

        Returns:
            GenreMoodResult with genre and mood classification
        """
        mode = force_mode or self.mode

        # First, get local audio features
        features = self.analyze_audio_file(audio_path)

        if not features.local_analysis_complete:
            error_msg = features.analysis_error or "Local analysis failed"
            # If local analysis failed and API fallback enabled
            if mode == ProcessingMode.API_ONLY or (
                self.enable_genre_fallback and mode == ProcessingMode.API_FALLBACK
            ):
                if self.enable_genre_fallback:
                    try:
                        return self._api_genre_classification(features, audio_path)
                    except Exception:
                        pass  # Fall back to local result

            # Return local result (may have low confidence)
            return GenreMoodResult(
                genres=[],
                moods=[],
                confidence=0.0,
                source="local",
            )

        # Get local classification first
        local_result = self._local_genre_classification(features)

        # Handle different modes
        if mode == ProcessingMode.LOCAL_ONLY:
            return local_result

        if mode == ProcessingMode.API_ONLY:
            if self.enable_genre_fallback:
                return self._api_genre_classification(features, audio_path)
            return local_result

        # For HYBRID or API_FALLBACK modes
        if mode in (ProcessingMode.HYBRID, ProcessingMode.API_FALLBACK):
            # Use local if confidence is high enough
            if local_result.confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                return local_result
            
            # Low confidence - try API fallback
            if self.enable_genre_fallback:
                try:
                    api_result = self._api_genre_classification(features, audio_path)
                    api_result.fallback_triggered = True
                    return api_result
                except Exception:
                    pass  # Keep using local result

        return local_result

    def process(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        force_mode: Optional[ProcessingMode] = None,
    ) -> AgentResponse:
        """
        Not implemented for AudioMatchAgent.
        Use classify_genre_mood() instead.
        """
        raise NotImplementedError("AudioMatchAgent uses classify_genre_mood() instead of process()")

    def get_audio_stats(self) -> Dict[str, Any]:
        """Get audio analysis statistics."""
        stats = dict(self._stats)
        stats["librosa_available"] = self._librosa_available
        stats["enable_genre_fallback"] = self.enable_genre_fallback
        return stats

    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"]

    def is_supported_format(self, file_path: str) -> bool:
        """Check if audio format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_formats()

    def reset_stats(self) -> None:
        """Reset audio analysis statistics."""
        self._stats = {
            "audio_files_analyzed": 0,
            "local_bpm_detections": 0,
            "local_energy_analyses": 0,
            "api_genre_requests": 0,
            "cache_hits": 0,
            "fallback_count": 0,
        }

    def _local_processing_impl(
        self,
        prompt: str,
        system_prompt: str = None,
    ):
        """
        Not implemented for AudioMatchAgent.
        Use classify_genre_mood() instead.
        """
        raise NotImplementedError("AudioMatchAgent uses classify_genre_mood() instead of process()")


# Convenience function for quick initialization
def create_audio_agent(
    api_key: str,
    **kwargs,
) -> AudioMatchAgent:
    """Create and return an AudioMatchAgent instance."""
    return AudioMatchAgent(api_key=api_key, **kwargs)
