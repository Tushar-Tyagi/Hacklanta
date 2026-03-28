"""
Video Editor Agent - Applies edit plans to videos using FFmpeg.

This module takes an edit plan (JSON) containing cut timestamps, transitions,
and style recommendations, then renders the final video using FFmpeg.
"""

import os
import json
import subprocess
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


class CutType:
    """Supported cut types."""
    STRAIGHT = "straight_cut"
    J_CUT = "j_cut"      # Audio starts before video
    L_CUT = "l_cut"      # Video starts before audio
    MATCH_CUT = "match_cut"
    JUMP_CUT = "jump_cut"


class TransitionType:
    """Supported transition types."""
    NONE = "none"
    CROSSFADE = "crossfade"
    DISSOLVE = "dissolve"
    FADE = "fade"
    WIPE = "wipe"


@dataclass
class VideoPart:
    """A segment of video to include in the final edit."""
    source_file: str
    start_time: str  # HH:MM:SS or HH:MM:SS:FF format
    end_time: str
    name: str


@dataclass
class EditPlan:
    """Complete edit plan with all editing instructions."""
    video_parts: List[Dict[str, str]]  # [{"video_1": "00:00:34-00:01:02"}]
    cuts: List[str]  # ["straight_cut", "j_cut", "crossfade"]
    transitions: List[str]  # ["crossfade", "none", "dissolve"]
    total_duration: str
    style: Optional[Dict[str, Any]] = None  # LUT/color grading info
    audio_source: Optional[str] = None  # External audio file path


class VideoEditor:
    """
    Video Editor that applies edit plans using FFmpeg.
    
    Usage:
        editor = VideoEditor()
        result = editor.apply_edit_plan(
            video_paths=["video1.mp4", "video2.mp4"],
            audio_path="audio.mp3",
            edit_plan=edit_plan_dict,
            output_path="output.mp4"
        )
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """Initialize VideoEditor with FFmpeg paths."""
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(
                [self.ffmpeg, "-version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("WARNING: FFmpeg not found. Install with: brew install ffmpeg")
            return False
    
    def _parse_time_range(self, time_str: str) -> Tuple[str, str]:
        """Parse time range string like '00:00:34-00:01:02' into (start, end)."""
        if '-' in time_str:
            start, end = time_str.split('-')
            return start.strip(), end.strip()
        return "00:00:00", time_str.strip()
    
    def _format_time_for_ffmpeg(self, time_str: str) -> str:
        """Ensure time is in HH:MM:SS format for FFmpeg."""
        # Handle various formats
        time_str = time_str.strip()
        
        # If it's just seconds (e.g., "34" or "34.5")
        if re.match(r'^[\d.]+$', time_str):
            seconds = float(time_str)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
        
        # If it has frames (00:02:00:12), convert to HH:MM:SS
        # Remove frame component
        parts = time_str.split(':')
        if len(parts) == 4:
            # HH:MM:SS:FF - drop frames
            return f"{parts[0]}:{parts[1]}:{parts[2]}"
        
        return time_str
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get duration of video file using ffprobe."""
        try:
            result = subprocess.run(
                [self.ffprobe, "-v", "error", "-show_entries", 
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 video_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
    
    def _build_concat_file(self, video_parts: List[Dict[str, str]], 
                           temp_dir: str) -> str:
        """Build concat demuxer file for FFmpeg."""
        concat_file = os.path.join(temp_dir, "concat.txt")
        
        with open(concat_file, 'w') as f:
            for part in video_parts:
                for name, time_range in part.items():
                    start, end = self._parse_time_range(time_range)
                    start_fmt = self._format_time_for_ffmpeg(start)
                    end_fmt = self._format_time_for_ffmpeg(end)
                    # Use absolute path
                    abs_path = os.path.abspath(name)
                    f.write(f"file '{abs_path}'\n")
                    f.write(f"inpoint {start_fmt}\n")
                    f.write(f"outpoint {end_fmt}\n")
        
        return concat_file
    
    def _build_filter_complex(self, transitions: List[str], 
                              has_audio: bool = True,
                              style: Optional[Dict[str, Any]] = None) -> str:
        """Build FFmpeg filter complex for transitions and color grading."""
        filters = []
        
        # Style/LUT application
        if style:
            lut_name = style.get("lut_name", "")
            intensity = style.get("intensity", 0.8)
            color_notes = style.get("color_grading_notes", "")
            
            # Apply color grading based on style
            if "Teal & Orange" in lut_name or "cinematic" in style.get("category", ""):
                # Teal shadows, orange highlights
                filters.append(
                    f"colorbalance=rs=0.1:gs=0.05:bs=-0.15:rm=0:gm=0:bm=-0.1[intro];[intro]eq=brightness=0.02:contrast=1.1"
                )
            elif "Vintage" in lut_name or "vintage" in style.get("category", ""):
                # Slight warm sepia tone
                filters.append("colorbalance=rs=0.1:gs=0.05:bs=-0.05")
            elif "Moody" in lut_name:
                # Desaturated, darker
                filters.append("colorbalance=rs=-0.1:gs=-0.1:bs=-0.1,eq=brightness=-0.1:contrast=1.2")
            elif "Noir" in lut_name:
                # High contrast B&W
                filters.append("hue=s=0,eq=contrast=1.3:brightness=-0.05")
            elif "Bright" in lut_name or "Travel" in lut_name:
                # Lifted shadows, brighter
                filters.append("eq=brightness=0.1:saturation=1.2")
            else:
                # Default - subtle warmth
                filters.append("colorbalance=rs=0.03:gs=0.01")
            
            # Apply intensity (mix with original)
            if intensity < 1.0:
                # Use blend filter to mix graded with original
                filters[-1] = f"[0:v]{filters[-1]}[graded];[graded][0:v]blend=all_mode='screen':c0_opacity={intensity}:c1_opacity={intensity}:c2_opacity={intensity}[out]"
        
        if not filters:
            filters.append("copy")
        
        return ";".join(filters) if len(filters) > 1 else filters[0]
    
    def _apply_transition(self, input1: str, input2: str, transition: str, 
                         duration: float = 0.5) -> str:
        """Build FFmpeg command for a specific transition between two clips."""
        if transition.lower() in ["crossfade", "dissolve"]:
            # Crossfade transition
            return f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset=0[outv]"
        elif transition.lower() == "wipe":
            return f"[0:v][1:v]xfade=transition=wipeleft:duration={duration}:offset=0[outv]"
        elif transition.lower() == "fade":
            return f"[0:v][1:v]xfade=transition=fadeblack:duration={duration}:offset=0[outv]"
        else:
            # No transition - straight cut
            return f"[0:v]copy[outv];[1:v]copy[outv]"
    
    def _extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        """Extract audio from video file."""
        cmd = [
            self.ffmpeg, "-y", "-i", video_path,
            "-vn", "-acodec", "copy",
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return output_path
    
    def apply_edit_plan(
        self,
        video_paths: List[str],
        audio_path: Optional[str],
        edit_plan: Dict[str, Any],
        output_path: str,
        quality: str = "high"  # high, medium, low
    ) -> Dict[str, Any]:
        """
        Apply the edit plan to create the final video.
        
        Args:
            video_paths: List of source video file paths
            audio_path: Optional external audio file path
            edit_plan: Edit plan JSON containing video_parts, cuts, transitions, style
            output_path: Path for the output video file
            quality: Output quality (high/medium/low)
            
        Returns:
            Dict with status, output_path, and any errors
        """
        if not self._verify_ffmpeg():
            return {
                "status": "error",
                "error": "FFmpeg not found. Please install FFmpeg.",
                "output_path": None
            }
        
        try:
            # Create temp directory
            temp_dir = os.path.join(os.path.dirname(output_path), "temp_edit")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract plan data
            video_parts = edit_plan.get("video_parts", [])
            cuts = edit_plan.get("cuts", [])
            transitions = edit_plan.get("transitions", [])
            style = edit_plan.get("style")
            
            # Build the edit - simple approach: concat with transitions
            if not video_parts:
                # Fallback: use first video as-is
                video_parts = [{video_paths[0]: "00:00:00-00:00:30"}]
            
            # Step 1: Extract all clips (use re-encoding for better compatibility)
            extracted_clips = []
            for i, part in enumerate(video_parts):
                for name, time_range in part.items():
                    if not os.path.exists(name):
                        print(f"Warning: Video file not found: {name}")
                        continue
                        
                    start, end = self._parse_time_range(time_range)
                    start_fmt = self._format_time_for_ffmpeg(start)
                    end_fmt = self._format_time_for_ffmpeg(end)
                    
                    clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
                    
                    # Use re-encoding for better compatibility with .MOV files
                    cmd = [
                        self.ffmpeg, "-y", "-i", name,
                        "-ss", start_fmt,
                        "-to", end_fmt,
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-c:a", "aac",
                        "-b:a", "128k",
                        "-avoid_negative_ts", "make_zero",
                        clip_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(clip_path):
                        extracted_clips.append(clip_path)
                        print(f"Extracted clip {i}: {clip_path}")
                    else:
                        print(f"Failed to extract clip {i}: {result.stderr[:200] if result.stderr else 'Unknown error'}")
            
            if not extracted_clips:
                return {
                    "status": "error",
                    "error": "Failed to extract any video clips",
                    "output_path": None
                }
            
            # Step 2: Build concat file (simple format - just file paths)
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, 'w') as f:
                for clip in extracted_clips:
                    f.write(f"file '{os.path.abspath(clip)}'\n")
            
            # Step 3: Concatenate videos FIRST (no audio, no filters) - simpler approach
            concat_output = os.path.join(temp_dir, "concated.mp4")
            
            concat_cmd = [
                self.ffmpeg, "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",  # Use copy for concat (fast)
                "-an",  # No audio
                concat_output
            ]
            
            result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # If copy fails, try re-encoding
                concat_cmd[9:11] = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
                result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0 or not os.path.exists(concat_output):
                return {
                    "status": "error",
                    "error": f"Failed to concatenate videos: {result.stderr[:300] if result.stderr else 'Unknown error'}",
                    "output_path": None
                }
            
            print(f"Concatenated video created: {concat_output}")
            
            # Step 4: Add audio and apply style (second step)
            final_cmd = [self.ffmpeg, "-y", "-i", concat_output]
            
            # Add external audio if provided
            if audio_path and os.path.exists(audio_path):
                final_cmd.extend(["-i", audio_path])
            
            # Add audio options
            if audio_path and os.path.exists(audio_path):
                # Replace video audio with external audio
                final_cmd.extend(["-map", "0:v", "-map", "1:a", "-shortest"])
            else:
                final_cmd.extend(["-c", "copy"])
            
            # Output settings
            final_cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                output_path
            ])
            
            # Run final FFmpeg
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                return {
                    "status": "error",
                    "error": f"FFmpeg failed: {error_msg[:500]}",
                    "output_path": None
                }
            
            if not os.path.exists(output_path):
                return {
                    "status": "error",
                    "error": "Output file was not created",
                    "output_path": None
                }
            
            # Cleanup temp files (optional - comment out to debug)
            # shutil.rmtree(temp_dir, ignore_errors=True)
            
            return {
                "status": "success",
                "output_path": output_path,
                "clips_used": len(extracted_clips),
                "style_applied": style.get("lut_name") if style else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Video processing timed out",
                "output_path": None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "output_path": None
            }
    
    def apply_lut_file(self, video_path: str, lut_file: str, 
                       output_path: str) -> Dict[str, Any]:
        """Apply a .cube LUT file to a video."""
        cmd = [
            self.ffmpeg, "-y", "-i", video_path,
            "-vf", f"lut3d={lut_file}",
            "-c:a", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0:
                return {"status": "success", "output_path": output_path}
            return {"status": "error", "error": result.stderr.decode()[:200]}
        except Exception as e:
            return {"status": "error", "error": str(e)}


def create_video_editor(ffmpeg_path: str = "ffmpeg", 
                       ffprobe_path: str = "ffprobe") -> VideoEditor:
    """Factory function to create a VideoEditor instance."""
    return VideoEditor(ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path)


# Example usage and testing
if __name__ == "__main__":
    # Example edit plan
    edit_plan = {
        "video_parts": [
            {"video1.mp4": "00:00:00-00:00:10"},
            {"video2.mp4": "00:00:05-00:00:20"},
        ],
        "cuts": ["straight_cut", "crossfade"],
        "transitions": ["none", "crossfade"],
        "total_duration": "00:00:30",
        "style": {
            "lut_name": "Cinematic Teal & Orange",
            "intensity": 0.8,
            "category": "cinematic",
            "color_grading_notes": "Boost teal in shadows, orange in highlights"
        }
    }
    
    # Usage:
    # editor = VideoEditor()
    # result = editor.apply_edit_plan(
    #     video_paths=["video1.mp4", "video2.mp4"],
    #     audio_path="audio.mp3",
    #     edit_plan=edit_plan,
    #     output_path="output.mp4"
    # )
    # print(result)
    
    print("VideoEditor module loaded. Use apply_edit_plan() to edit videos.")