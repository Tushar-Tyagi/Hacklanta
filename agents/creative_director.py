"""
Creative Director Agent - Generates video edit compositions and decisions
based on audio analysis, visual style, and target duration.
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResponse, ProcessingMode

@dataclass
class EditDecision:
    """A single edit decision in the composition."""
    cut_type: str
    time: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cut_type": self.cut_type,
            "time": self.time,
            "reason": self.reason
        }

class CreativeDirectorAgent(BaseAgent):
    """
    Creative Director Agent for final video composition.
    
    This agent takes analyzed data from other agents and synthesizes 
    it into a cohesive edit plan (composition) that fits a specific duration.
    """
    
    DEFAULT_API_MODEL = "openai/gpt-4o-mini"
    
    def __init__(self, api_key: str, mode: ProcessingMode = ProcessingMode.HYBRID, **kwargs):
        super().__init__(api_key=api_key, mode=mode, **kwargs)
        self.system_prompt = (
            "You are a world-class AI Creative Director and Film Editor. "
            "Your goal is to take a set of audio and visual analysis data and "
            "create a precise, cinematic edit composition. "
            "You favor pacing, rhythm, and emotional resonance. "
            "You use professional terminology like Jump Cut, L-Cut, Match Cut, "
            "Cross-fade, and Straight Cut."
        )

    def _local_processing_impl(self, prompt: str, system_prompt: Optional[str] = None) -> AgentResponse:
        """Local processing logic for creative director is simplified to basic rules."""
        # Simple logical fallback if no API is available
        return AgentResponse(
            content=json.dumps({
                "edit_decisions": [
                    {"cut_type": "Straight Cut", "time": 0.0, "reason": "Opening shot"},
                    {"cut_type": "Straight Cut", "time": 5.0, "reason": "Basic interval"}
                ]
            }),
            confidence=0.4,
            mode_used="local",
            source="local"
        )

    def generate_composition(
        self, 
        audio_features: Dict[str, Any], 
        style_result: Dict[str, Any], 
        target_duration: str,
        video_count: int = 1,
        source_videos: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a full video composition plan in format compatible with VideoEditor.
        
        Args:
            audio_features: Results from AudioMatchAgent
            style_result: Results from StyleDirectorAgent
            target_duration: Target duration in seconds (string)
            video_count: Number of source video clips available
            source_videos: List of source video file paths
        """
        source_context = f"You have {video_count} separate video clips to work with." if video_count > 1 else "You have 1 video clip to work with."
        
        # Get style info for the prompt
        style_name = style_result.get('name', 'N/A')
        style_category = style_result.get('category', 'cinematic')
        style_notes = style_result.get('color_grading_notes', '')
        
        prompt = f"""Generate a cinematic video edit composition plan in JSON format compatible with FFmpeg video editing.

TARGET DURATION: {target_duration} seconds
{source_context}

INPUT DATA:
- AUDIO: BPM={audio_features.get('bpm', 'N/A')}, Mood={audio_features.get('mood', 'N/A')}, Energy={audio_features.get('energy', 'N/A')}
- STYLE: Name={style_name}, Mood={style_result.get('mood', 'N/A')}, Category={style_category}

REQUIREMENTS:
1. Honor the exact target duration of {target_duration} seconds.
2. Generate video_parts: list of video segments to include. Format: [{{"filename": "start-end"}}, ...]
   - Use filenames from: {source_videos or ['video1.mp4']}
   - Time format: HH:MM:SS-HH:MM:SS or seconds (e.g., "00:00:34-00:01:02" or "34-62")
3. Generate cuts: list of cut types matching each segment transition (e.g., "straight_cut", "j_cut", "l_cut", "match_cut")
4. Generate transitions: list of transitions between segments (e.g., "none", "crossfade", "dissolve", "fade")
5. Include style object with lut_name, intensity, category, and color_grading_notes
6. Distribute cuts based on BPM if music is present - faster BPM = more cuts
7. Return ONLY valid JSON (no markdown).

JSON FORMAT:
{{
  "video_parts": [
    {{"video1.mp4": "00:00:00-00:00:10"}},
    {{"video2.mp4": "00:00:05-00:00:20"}}
  ],
  "cuts": ["straight_cut", "crossfade"],
  "transitions": ["none", "crossfade"],
  "total_duration": "{target_duration}",
  "style": {{
    "lut_name": "{style_name}",
    "intensity": 0.8,
    "category": "{style_category}",
    "color_grading_notes": "{style_notes}"
  }},
  "audio_sync": {{
    "bpm": {audio_features.get('bpm', 0)},
    "beat_markers": [0.0, 2.4, 4.8, 7.2]
  }}
}}
"""
        response = self.process(prompt)
        
        try:
            # Clean up response if it has markdown blocks
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            
            # Ensure required fields exist
            if "video_parts" not in data:
                data["video_parts"] = [{"video1.mp4": f"00:00:00-{target_duration}"}]
            if "cuts" not in data:
                data["cuts"] = ["straight_cut"]
            if "transitions" not in data:
                data["transitions"] = ["none"]
            if "style" not in data:
                data["style"] = {
                    "lut_name": style_name,
                    "intensity": 0.8,
                    "category": style_category,
                    "color_grading_notes": style_notes
                }
            
            data["total_duration"] = target_duration
            return data
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # Safe default fallback in new format
            return {
                "video_parts": [
                    {source_videos[0] if source_videos else "video1.mp4": f"00:00:00-{target_duration}"}
                ],
                "cuts": ["straight_cut"],
                "transitions": ["none"],
                "total_duration": target_duration,
                "style": {
                    "lut_name": style_name,
                    "intensity": 0.8,
                    "category": style_category,
                    "color_grading_notes": style_notes
                },
                "fallback": True,
                "error": str(e)
            }

    def synthesize_scenes(
        self, 
        chunk_summaries: List[Dict[str, Any]], 
        detected_scenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synthesize 10s chunk summaries into logical narrative scenes.
        
        Args:
            chunk_summaries: List of 10s chunk analysis results
            detected_scenes: List of histogram-based scene cuts from VideoProcessor
            
        Returns:
            List of synthesized scenes with semantic descriptions
        """
        if not chunk_summaries:
            return detected_scenes

        # Prepare context for the LLM
        chunks_info = []
        for i, chunk in enumerate(chunk_summaries):
            start = i * 10
            end = (i+1) * 10
            summary = chunk.get('summary', 'No summary')
            actions = ", ".join(chunk.get('actions', []))
            camera = chunk.get('camera_movement', 'Unknown')
            chunks_info.append(f"[{start}s-{end}s]: {summary} (Actions: {actions}, Camera: {camera})")
        
        detected_info = [f"Shot {s['id']}: {s['start']:.1f}s-{s['end']:.1f}s" for s in detected_scenes]
        
        prompt = f"""As a Creative Director, merge these 10-second video chunk summaries into a sequence of cohesive narrative scenes.
        
10-SECOND CHUNK SUMMARIES:
{chr(10).join(chunks_info)}

DETECTED TECHNICAL CUTS (Histogram-based):
{chr(10).join(detected_info)}

INSTRUCTIONS:
1. Group contiguous 10s chunks that represent the same narrative beat or visual sequence.
2. Use the "Technical Cuts" as a guide for where transitions might happen, but prioritize narrative continuity.
3. For each synthesized scene, provide:
   - start/end timestamps
   - a descriptive title
   - a detailed action summary
   - predominant camera movement
   - energy pacing (Low/Medium/High)
4. Return ONLY valid JSON.

JSON FORMAT:
{{
  "scenes": [
    {{
      "start": 0.0,
      "end": 15.0,
      "title": "Scene Title",
      "description": "Detailed description of what is happening.",
      "camera_movement": "Description of movement",
      "pacing": "Low/Medium/High"
    }}
  ]
}}
"""
        response = self.process(prompt)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            # Ensure it's a list even if empty
            return data.get("scenes", detected_scenes)
        except Exception as e:
            print(f"Scene synthesis failed: {e}")
            return detected_scenes
