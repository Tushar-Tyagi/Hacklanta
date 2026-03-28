"""Style Director Agent - Image aesthetic analysis with histogram matching."""

import time
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .base_agent import BaseAgent, AgentResponse, ProcessingMode


class ColorSpace(Enum):
    """Color space options for histogram computation."""
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    YCrCb = "ycrcb"


@dataclass
class LUTRecommendation:
    """Cinematic LUT recommendation."""
    name: str
    description: str
    category: str  # cinematic, vintage, moody, modern
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    color_grading_notes: str
    compatible_styles: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "color_grading_notes": self.color_grading_notes,
            "compatible_styles": self.compatible_styles,
        }


@dataclass
class StyleResult:
    """Result of style analysis."""
    primary_colors: List[Tuple[int, int, int]]  # RGB tuples
    color_distribution: Dict[str, float]  # color name -> percentage
    dominant_tones: List[str]
    mood: str
    contrast_level: str  # "low", "medium", "high"
    saturation_level: str  # "muted", "moderate", "vibrant"
    temperature: str  # "cool", "neutral", "warm"
    confidence: float
    lut_recommendations: List[LUTRecommendation] = field(default_factory=list)
    target_image_histogram: Optional[np.ndarray] = None
    source_image_histogram: Optional[np.ndarray] = None
    histogram_similarity: float = 0.0
    processing_time_ms: int = 0
    source: str = "local"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_colors": [list(c) for c in self.primary_colors],
            "color_distribution": self.color_distribution,
            "dominant_tones": self.dominant_tones,
            "mood": self.mood,
            "contrast_level": self.contrast_level,
            "saturation_level": self.saturation_level,
            "temperature": self.temperature,
            "confidence": self.confidence,
            "lut_recommendations": [lut.to_dict() for lut in self.lut_recommendations],
            "histogram_similarity": self.histogram_similarity,
            "processing_time_ms": self.processing_time_ms,
            "source": self.source,
            "error": self.error,
            "metadata": self.metadata,
        }


class StyleDirectorAgent(BaseAgent):
    """
    Style Director Agent for image aesthetic analysis.
    
    Features:
    - Local color histogram matching with OpenCV
    - Basic style transfer using histogram matching
    - Confidence scoring based on analysis quality
    - OpenRouter API fallback for advanced aesthetic analysis
    - Cinematic LUT recommendations
    """
    
    DEFAULT_API_MODEL = "openai/gpt-4o-mini"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    
    # Predefined cinematic LUTs
    CINEMATIC_LUTS = [
        LUTRecommendation(
            name="Cinematic Teal & Orange",
            description="Classic blockbuster look with teal shadows and orange highlights",
            category="cinematic",
            intensity=0.8,
            confidence=0.9,
            color_grading_notes="Boost cyan/blue in shadows, push orange/red in highlights",
            compatible_styles=["action", "dramatic", "modern"],
        ),
        LUTRecommendation(
            name="Vintage Film 35mm",
            description="Kodak 35mm film emulation with subtle grain",
            category="vintage",
            intensity=0.7,
            confidence=0.85,
            color_grading_notes="Slight desaturation, warm highlights, lifted blacks",
            compatible_styles=["nostalgic", "documentary", "romantic"],
        ),
        LUTRecommendation(
            name="Moody Desaturated",
            description="Desaturated, moody look with crushed blacks",
            category="moody",
            intensity=0.6,
            confidence=0.8,
            color_grading_notes="Reduce saturation 30%, add contrast curve, deep shadows",
            compatible_styles=["thriller", "drama", "artistic"],
        ),
        LUTRecommendation(
            name="Travel Bright",
            description="High key, bright and airy travel photography style",
            category="modern",
            intensity=0.75,
            confidence=0.85,
            color_grading_notes="Lift shadows, boost whites, slight warmth in highlights",
            compatible_styles=["travel", "lifestyle", "bright"],
        ),
        LUTRecommendation(
            name="Golden Hour Portrait",
            description="Warm, flattering portrait with golden hour lighting feel",
            category="portrait",
            intensity=0.7,
            confidence=0.9,
            color_grading_notes="Warm skin tones, soft highlights, reduced contrast",
            compatible_styles=["portrait", "landscape", "sunset"],
        ),
        LUTRecommendation(
            name="Noir Mystery",
            description="High contrast black and white with moody atmosphere",
            category="moody",
            intensity=0.85,
            confidence=0.8,
            color_grading_notes="Desaturate, increase contrast, add vignette",
            compatible_styles=["mystery", "dramatic", "classic"],
        ),
        LUTRecommendation(
            name="Cyberpunk Neon",
            description="Futuristic neon-lit night scene aesthetic",
            category="cinematic",
            intensity=0.9,
            confidence=0.75,
            color_grading_notes="Boost cyans, magentas, increase saturation in neons",
            compatible_styles=["sci-fi", "night", "urban"],
        ),
        LUTRecommendation(
            name="Soft Beauty",
            description="Soft, flattering beauty and fashion editorial look",
            category="portrait",
            intensity=0.5,
            confidence=0.85,
            color_grading_notes="Soft skin smoothing, lifted shadows, neutral tones",
            compatible_styles=["portrait", "beauty", "lifestyle"],
        ),
    ]
    
    def __init__(
        self,
        api_key: str,
        mode: ProcessingMode = ProcessingMode.HYBRID,
        histogram_bins: int = 256,
        color_space: ColorSpace = ColorSpace.HSV,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        **kwargs,
    ):
        """Initialize Style Director Agent."""
        self.histogram_bins = histogram_bins
        self.color_space = color_space
        self._cv2_available = CV2_AVAILABLE
        
        super().__init__(
            api_key=api_key,
            mode=mode,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
    
    def _local_processing_impl(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """Not used - image processing handled separately."""
        raise NotImplementedError("Use analyze_style() or apply_style_transfer() instead")
    
    def _compute_histogram(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Compute color histogram for an image."""
        if self.color_space == ColorSpace.HSV:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == ColorSpace.LAB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == ColorSpace.YCrCb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        channels = cv2.split(image)
        histograms = []
        
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [self.histogram_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        return tuple(histograms)
    
    def _calculate_histogram_similarity(
        self,
        hist1: Tuple[np.ndarray, ...],
        hist2: Tuple[np.ndarray, ...],
    ) -> float:
        """Calculate similarity between two histograms."""
        if len(hist1) != len(hist2):
            return 0.0
        
        similarities = []
        for h1, h2 in zip(hist1, hist2):
            correlation = cv2.compareHist(
                h1.astype(np.float32),
                h2.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarity = (correlation + 1) / 2
            similarities.append(similarity)
        
        chi_similarities = []
        for h1, h2 in zip(hist1, hist2):
            chi_sq = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CHISQR)
            chi_similarities.append(max(0, 1 - chi_sq / 100.0))
        
        intersections = []
        for h1, h2 in zip(hist1, hist2):
            intersection = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_INTERSECT)
            intersections.append(min(1.0, intersection))
        
        avg_correlation = np.mean(similarities)
        avg_chi = np.mean(chi_similarities)
        avg_intersection = np.mean(intersections)
        
        combined_similarity = 0.4 * avg_correlation + 0.3 * avg_chi + 0.3 * avg_intersection
        return min(1.0, max(0.0, combined_similarity))
    
    def _extract_primary_colors(self, image: np.ndarray, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract primary colors using K-means clustering."""
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1]
        
        primary_colors = []
        for idx in sorted_indices[:num_colors]:
            center = centers[idx].astype(np.int32)
            rgb = (center[2], center[1], center[0])
            primary_colors.append(rgb)
        
        return primary_colors
    
    def _analyze_contrast(self, image: np.ndarray) -> str:
        """Analyze image contrast level."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = gray.std()
        if std_dev < 40:
            return "low"
        elif std_dev < 80:
            return "medium"
        else:
            return "high"
    
    def _analyze_saturation(self, image: np.ndarray) -> str:
        """Analyze image saturation level."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = saturation.mean()
        if avg_saturation < 60:
            return "muted"
        elif avg_saturation < 120:
            return "moderate"
        else:
            return "vibrant"
    
    def _analyze_temperature(self, image: np.ndarray) -> str:
        """Analyze image color temperature."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        avg_b = b_channel.mean() - 128
        
        if avg_b < -10:
            return "cool"
        elif avg_b > 10:
            return "warm"
        else:
            return "neutral"
    
    def _determine_mood(self, contrast: str, saturation: str, temperature: str) -> str:
        """Determine image mood."""
        if contrast == "high" and saturation in ["muted", "moderate"]:
            return "moody"
        elif contrast == "high" and saturation == "vibrant":
            return "dramatic"
        elif contrast == "low" and saturation == "vibrant":
            return "bright"
        elif temperature == "warm":
            return "warm"
        elif temperature == "cool":
            return "cool"
        else:
            return "neutral"
    
    def _calculate_confidence_local(self, image: np.ndarray, histogram_similarity: float) -> float:
        """Calculate confidence for local analysis."""
        confidence = 0.5
        
        # Check image quality (not too dark or bright)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        if 40 < mean_brightness < 220:
            confidence += 0.15
        
        # Good histogram similarity boosts confidence
        if histogram_similarity > 0.5:
            confidence += histogram_similarity * 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_color_distribution(self, primary_colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """Calculate approximate color distribution."""
        distribution = {}
        
        for rgb in primary_colors:
            r, g, b = rgb
            if r > g and r > b:
                if r > 200 and g < 100 and b < 100:
                    color = "red"
                elif r > 150 and g > 100 and b < 100:
                    color = "orange"
                else:
                    color = "warm"
            elif b > g and b > r:
                if b > 200 and g < 100 and r < 100:
                    color = "blue"
                elif b > 150 and g > 100:
                    color = "cyan"
                else:
                    color = "cool"
            elif g > r and g > b:
                if g > 150:
                    color = "green"
                else:
                    color = "nature"
            elif r > 200 and g > 200 and b < 100:
                color = "yellow"
            elif r < 50 and g < 50 and b < 50:
                color = "black"
            elif r > 200 and g > 200 and b > 200:
                color = "white"
            else:
                color = "neutral"
            distribution[color] = distribution.get(color, 0.0) + 20.0
        
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total * 100 for k, v in distribution.items()}
        
        return distribution
    
    def _get_dominant_tones(self, primary_colors: List[Tuple[int, int, int]]) -> List[str]:
        """Get dominant tone descriptions."""
        tones = []
        
        for rgb in primary_colors:
            r, g, b = rgb
            luminance = (r + g + b) / 3
            
            if luminance < 50:
                tones.append("deep black")
            elif luminance < 100:
                tones.append("shadow")
            elif luminance < 160:
                tones.append("midtone")
            elif luminance < 220:
                tones.append("highlight")
            else:
                tones.append("bright white")
        
        for rgb in primary_colors:
            r, g, b = rgb
            min_c = min(r, g, b)
            max_c = max(r, g, b)
            saturation = (max_c - min_c) / max_c if max_c > 0 else 0
            luminance = (r + g + b) / 3
            
            if saturation < 0.2 and 50 < luminance < 200:
                tones.append("desaturated")
            elif saturation > 0.7 and luminance > 80:
                tones.append("vibrant")
        
        seen = set()
        unique_tones = []
        for tone in tones:
            if tone not in seen:
                seen.add(tone)
                unique_tones.append(tone)
        
        return unique_tones[:5]
    
    def _get_matching_luts(self, mood: str, dominant_tones: List[str]) -> List[LUTRecommendation]:
        """Get LUT recommendations matching image mood."""
        matching_luts = []
        
        for lut in self.CINEMATIC_LUTS:
            score = 0.0
            if mood == "moody" and lut.category in ["moody", "vintage"]:
                score += 0.8
            elif mood == "dramatic" and lut.category in ["cinematic", "modern"]:
                score += 0.8
            elif mood == "bright" and lut.category in ["modern", "portrait"]:
                score += 0.7
            
            for tone in dominant_tones:
                if "vibrant" in tone and "cinematic" in lut.category:
                    score += 0.3
                elif "desaturated" in tone and lut.category == "moody":
                    score += 0.3
            
            if score > 0:
                lut.confidence = min(1.0, score)
                matching_luts.append(lut)
        
        if not matching_luts:
            matching_luts = self.CINEMATIC_LUTS[:3]
        
        return sorted(matching_luts, key=lambda x: x.confidence, reverse=True)[:4]

    def analyze_style(self, image_path: str, reference_path: Optional[str] = None) -> StyleResult:
        """Analyze the style of an image."""
        start_time = time.time()
        
        if not self._cv2_available:
            return StyleResult(
                primary_colors=[],
                color_distribution={},
                dominant_tones=[],
                mood="unknown",
                contrast_level="unknown",
                saturation_level="unknown",
                temperature="unknown",
                confidence=0.0,
                error="OpenCV not available",
                source="local",
            )
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return StyleResult(
                    primary_colors=[],
                    color_distribution={},
                    dominant_tones=[],
                    mood="unknown",
                    contrast_level="unknown",
                    saturation_level="unknown",
                    temperature="unknown",
                    confidence=0.0,
                    error=f"Could not load image: {image_path}",
                    source="local",
                )
            
            hist = self._compute_histogram(image)
            
            histogram_similarity = 0.0
            reference_hist = None
            if reference_path:
                ref_image = cv2.imread(str(reference_path))
                if ref_image is not None:
                    reference_hist = self._compute_histogram(ref_image)
                    histogram_similarity = self._calculate_histogram_similarity(hist, reference_hist)
            
            primary_colors = self._extract_primary_colors(image)
            contrast_level = self._analyze_contrast(image)
            saturation_level = self._analyze_saturation(image)
            temperature = self._analyze_temperature(image)
            mood = self._determine_mood(contrast_level, saturation_level, temperature)
            color_distribution = self._calculate_color_distribution(primary_colors)
            dominant_tones = self._get_dominant_tones(primary_colors)
            confidence = self._calculate_confidence_local(image, histogram_similarity)
            lut_recommendations = self._get_matching_luts(mood, dominant_tones)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return StyleResult(
                primary_colors=primary_colors,
                color_distribution=color_distribution,
                dominant_tones=dominant_tones,
                mood=mood,
                contrast_level=contrast_level,
                saturation_level=saturation_level,
                temperature=temperature,
                confidence=confidence,
                lut_recommendations=lut_recommendations,
                target_image_histogram=hist[0],
                source_image_histogram=reference_hist[0] if reference_hist else None,
                histogram_similarity=histogram_similarity,
                processing_time_ms=processing_time,
                source="local",
            )
            
        except Exception as e:
            return StyleResult(
                primary_colors=[],
                color_distribution={},
                dominant_tones=[],
                mood="unknown",
                contrast_level="unknown",
                saturation_level="unknown",
                temperature="unknown",
                confidence=0.0,
                error=f"Analysis failed: {str(e)}",
                source="local",
            )
    
    def get_lut_recommendations(self, style_result: StyleResult) -> List[LUTRecommendation]:
        """Get LUT recommendations based on style analysis."""
        return self._get_matching_luts(style_result.mood, style_result.dominant_tones)
    
    def apply_style_transfer(self, source: str, reference: str) -> Optional[str]:
        """Apply style transfer using histogram matching."""
        if not self._cv2_available:
            return None
        
        try:
            src_image = cv2.imread(str(source))
            ref_image = cv2.imread(str(reference))
            
            if src_image is None or ref_image is None:
                return None
            
            src_lab = cv2.cvtColor(src_image, cv2.COLOR_BGR2LAB)
            ref_lab = cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB)
            
            src_channels = cv2.split(src_lab)
            ref_channels = cv2.split(ref_lab)
            
            matched_channels = []
            for src_ch, ref_ch in zip(src_channels, ref_channels):
                ref_hist = cv2.calcHist([ref_ch], [0], None, [256], [0, 256])
                ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
                
                lookup = np.zeros(256, dtype=np.uint8)
                src_hist = cv2.calcHist([src_ch], [0], None, [256], [0, 256])
                src_hist = cv2.normalize(src_hist, src_hist).flatten()
                
                src_cdf = np.cumsum(src_hist)
                ref_cdf = np.cumsum(ref_hist)
                
                src_cdf = src_cdf / src_cdf[-1]
                ref_cdf = ref_cdf / ref_cdf[-1]
                
                for i in range(256):
                    diff = np.abs(src_cdf[i] - ref_cdf)
                    j = np.where(diff == np.min(diff))[0][0]
                    lookup[i] = j
                
                matched = cv2.LUT(src_ch, lookup)
                matched_channels.append(matched)
            
            matched_lab = cv2.merge(matched_channels)
            result = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
            
            output_path = source.replace(".jpg", "_styled.jpg").replace(".png", "_styled.png")
            cv2.imwrite(output_path, result)
            
            return output_path
            
        except Exception as e:
            print(f"Style transfer failed: {e}")
            return None

    def analyze_style_with_api(
        self,
        image_path: str,
        style_result: StyleResult,
    ) -> StyleResult:
        """Enhance style analysis with OpenRouter API."""
        if self.mode == ProcessingMode.LOCAL_ONLY:
            return style_result
        
        if style_result.confidence >= self.confidence_threshold:
            return style_result
        
        try:
            prompt = f"""Analyze this image's aesthetic style and provide recommendations.

Current Local Analysis:
- Primary Colors: {style_result.primary_colors[:3]}
- Dominant Tones: {style_result.dominant_tones}
- Mood: {style_result.mood}
- Contrast: {style_result.contrast_level}
- Saturation: {style_result.saturation_level}
- Temperature: {style_result.temperature}
- Confidence: {style_result.confidence:.2f}

Provide a JSON response with:
1. refined_mood: (string) Refined mood description
2. enhanced_notes: (string) Additional aesthetic insights
3. lut_selections: (array) Names of recommended LUTs from: {', '.join([lut.name for lut in self.CINEMATIC_LUTS])}
4. additional_recommendations: (array) Any other style advice"""

            response = self.process(prompt)
            
            if response.content:
                style_result.metadata["api_analysis"] = response.content
                style_result.metadata["api_confidence"] = response.confidence
                style_result.source = "api"
            
        except Exception as e:
            style_result.metadata["api_error"] = str(e)
        
        return style_result
    
    def full_analysis(
        self,
        image_path: str,
        reference_path: Optional[str] = None,
        use_api_fallback: bool = True,
    ) -> StyleResult:
        """Full image style analysis with optional API enhancement."""
        result = self.analyze_style(image_path, reference_path)
        
        if use_api_fallback and result.confidence < self.confidence_threshold:
            result = self.analyze_style_with_api(image_path, result)
        
        return result


def create_style_director(
    api_key: str,
    mode: ProcessingMode = ProcessingMode.HYBRID,
    **kwargs,
) -> StyleDirectorAgent:
    """Create and return a StyleDirectorAgent instance."""
    return StyleDirectorAgent(api_key=api_key, mode=mode, **kwargs)
