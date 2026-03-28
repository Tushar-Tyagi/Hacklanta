"""
Video Processor Agent - Real scene detection and keyframe extraction.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class VideoProcessor:
    """
    Utility class for processing raw video files:
    - Histogram-based scene detection.
    - Representative keyframe extraction.
    """
    
    def __init__(self, scene_threshold: float = 0.35):
        """Initialize the video processor with a scene detection sensitivity."""
        self.scene_threshold = scene_threshold
        
    def detect_scenes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Detect scenes in a video using color histogram differencing.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of detected scenes with start, end, and duration
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        last_hist = None
        last_scene_frame = 0
        
        # Skip frames for efficiency (e.g., check every 5th frame)
        frame_skip = 5
        
        for frame_idx in range(0, total_frames, frame_skip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to HSV for robust histogram comparison
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [12, 15], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if last_hist is not None:
                # Compare similarity (1.0 = identical, 0.0 = completely different)
                similarity = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                
                # If similarity drops below threshold, a cut is likely
                if similarity < self.scene_threshold:
                    scene_start = float(last_scene_frame / fps)
                    scene_end = float(frame_idx / fps)
                    
                    scenes.append({
                        "id": len(scenes) + 1,
                        "start": scene_start,
                        "end": scene_end,
                        "duration": scene_end - scene_start,
                        "type": "Detected Shot",
                        "confidence": float(1.0 - similarity),
                        "source": os.path.basename(video_path)
                    })
                    last_scene_frame = frame_idx
                    
            last_hist = hist
            
        # Add final scene
        final_end = float(total_frames / fps)
        final_start = float(last_scene_frame / fps)
        scenes.append({
            "id": len(scenes) + 1,
            "start": final_start,
            "end": final_end,
            "duration": final_end - final_start,
            "type": "Final Shot",
            "confidence": 1.0,
            "source": os.path.basename(video_path)
        })
        
        cap.release()
        return scenes
        
    def extract_keyframe(self, video_path: str, timestamp: float = 2.0) -> Optional[str]:
        """
        Extract a single keyframe from a video at a specific time.
        
        Args:
            video_path: Path to video
            timestamp: Time in seconds to grab the frame from
            
        Returns:
            Path to the extracted .jpg file
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Save frame to same directory as video
            output_path = f"{os.path.splitext(video_path)[0]}_frame.jpg"
            cv2.imwrite(output_path, frame)
            return output_path
            
        return None

    def extract_frames_at_fps(self, video_path: str, target_fps: float = 1.0) -> List[str]:
        """
        Extract frames from a video at a specific FPS.
        
        Args:
            video_path: Path to video
            target_fps: Number of frames per second to extract
            
        Returns:
            List of paths to extracted .jpg files
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        # Calculate frame indices to extract
        frame_indices = []
        for t in np.arange(0, duration, 1.0 / target_fps):
            idx = int(t * video_fps)
            if idx < total_frames:
                frame_indices.append(idx)
        
        extracted_paths = []
        base_name = os.path.splitext(video_path)[0]
        # Use a temporary directory for frames
        output_dir = f"{base_name}_temp_frames"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for efficiency if needed, but keeping original for now
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_paths.append(frame_path)
        
        cap.release()
        return extracted_paths

    def get_video_chunks(self, frame_paths: List[str], chunk_size: int = 10) -> List[List[str]]:
        """Group frame paths into segments (chunks)."""
        return [frame_paths[i:i + chunk_size] for i in range(0, len(frame_paths), chunk_size)]
