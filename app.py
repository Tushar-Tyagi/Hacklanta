import os
import time
from dotenv import load_dotenv

# Load environment variables from secrets.env
load_dotenv('secrets.env')

import re
import secrets
import subprocess
import json as json_module
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# Agent imports
from agents import (
    AudioMatchAgent, 
    StyleDirectorAgent, 
    CreativeDirectorAgent, 
    VideoEditor,
    VideoProcessor,
    ProcessingMode
)

# Utils
from utils.cache_manager import default_cache

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov', 'm4v'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'ogg', 'flac'}

# Supported URL patterns for audio extraction
SUPPORTED_URL_PATTERNS = [
    r'(https?://)?(www\.)?youtube\.com/watch',
    r'(https?://)?(www\.)?youtu\.be/',
    r'(https?://)?(www\.)?youtube\.com/shorts/',
    r'(https?://)?(www\.)?instagram\.com/(reel|p|stories)/',
    r'(https?://)?(www\.)?tiktok\.com/',
    r'(https?://)?vm\.tiktok\.com/',
]

def is_supported_url(url):
    """Check if a URL matches supported platforms."""
    return any(re.match(pattern, url) for pattern in SUPPORTED_URL_PATTERNS)

def download_audio_from_url(url, task_id):
    """
    Download audio from a YouTube/Instagram/TikTok URL using yt-dlp.
    Returns (audio_path, metadata_dict) or raises an exception.
    """
    output_template = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_%(title)s.%(ext)s")
    
    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'mp3',
        '--audio-quality', '192K',
        '--no-playlist',
        '--no-warnings',
        '--max-filesize', '50m',
        '--output', output_template,
        '--print-json',
        '--no-simulate',
        '--cookies', os.path.join(os.path.dirname(__file__), 'youtube_cookies.txt'),
        '--js-runtimes', 'node',
        '--remote-components', 'ejs:github',
        '--no-check-certificates',
        url
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120  # 2 minute timeout
    )
    
    if result.returncode != 0:
        # Filter out WARNING lines — only keep actual ERROR lines
        error_lines = [l for l in result.stderr.strip().splitlines()
                       if not l.startswith('WARNING:')]
        error_msg = '\n'.join(error_lines).strip() or result.stderr.strip()
        # Clean up common yt-dlp error messages
        if 'is not a valid URL' in error_msg or 'Unsupported URL' in error_msg:
            raise ValueError(f"Unsupported or invalid URL: {url}")
        if 'Private video' in error_msg:
            raise ValueError("This video is private and cannot be accessed.")
        if 'Video unavailable' in error_msg:
            raise ValueError("This video is unavailable.")
        raise RuntimeError(f"yt-dlp error: {error_msg[:200]}")
    
    # Parse the JSON output to get the actual filename
    try:
        info = json_module.loads(result.stdout.strip().split('\n')[-1])
    except (json_module.JSONDecodeError, IndexError):
        # Fallback: find the most recent mp3 in uploads
        info = {}
    
    # Find the downloaded file
    downloaded_path = None
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.startswith(task_id) and f.endswith('.mp3'):
            downloaded_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            break
    
    if not downloaded_path or not os.path.exists(downloaded_path):
        raise RuntimeError("Audio download completed but file not found.")
    
    metadata = {
        'title': info.get('title', 'Unknown'),
        'duration': info.get('duration', 0),
        'uploader': info.get('uploader', 'Unknown'),
        'platform': info.get('extractor', 'Unknown'),
        'filename': os.path.basename(downloaded_path)
    }
    
    return downloaded_path, metadata

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/download-audio', methods=['POST'])
def download_audio_endpoint():
    """Download audio from a social media URL and return metadata."""
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url'].strip()
    
    if not url:
        return jsonify({'error': 'URL cannot be empty'}), 400
    
    if not is_supported_url(url):
        return jsonify({'error': 'Unsupported URL. Supported platforms: YouTube, Instagram, TikTok'}), 400
    
    task_id = secrets.token_hex(8)
    
    try:
        audio_path, metadata = download_audio_from_url(url, task_id)
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'audio_path': audio_path,
            'metadata': metadata
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Download timed out. The URL may be too long or the server is slow.'}), 408
    except Exception as e:
        app.logger.error(f"Audio download error: {str(e)}")
        return jsonify({'error': f'Failed to download audio: {str(e)}'}), 500


@app.route('/api/process', methods=['POST'])
def process_media():
    if 'video' not in request.files:
        return jsonify({'error': 'No video files provided'}), 400
        
    video_files = request.files.getlist('video')
    audio_file = request.files.get('audio')
    audio_url = request.form.get('audio_url', '').strip()
    target_duration = request.form.get('target_duration', '30')
    
    if not video_files or video_files[0].filename == '':
        return jsonify({'error': 'No selected video files'}), 400
        
    # Generate unique IDs for the task
    task_id = secrets.token_hex(8)
    
    video_paths = []
    for i, video_file in enumerate(video_files):
        if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({'error': f'Invalid format for file {video_file.filename}.'}), 400
            
        video_filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{i}_{video_filename}")
        video_file.save(video_path)
        video_paths.append(video_path)
    
    audio_path = None
    audio_metadata = None
    
    # Priority: uploaded file > URL
    if audio_file and audio_file.filename != '':
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': 'Invalid audio format. Allowed formats: ' + ', '.join(ALLOWED_AUDIO_EXTENSIONS)}), 400
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{audio_filename}")
        audio_file.save(audio_path)
    elif audio_url:
        try:
            audio_path, audio_metadata = download_audio_from_url(audio_url, task_id)
        except Exception as e:
            return jsonify({'error': f'Failed to download audio from URL: {str(e)}'}), 400

    # In a real app, this would start a background task (e.g. Celery).
    # Since we need synchronous response or a simple polling interface, 
    # we'll run mock processing here that simulates calling the agents.
    
    try:
        # Placeholder for agent calls
        results = run_agents(video_paths, audio_path, task_id, target_duration)
        if audio_metadata:
            results['audio_source'] = audio_metadata
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': f"Internal server error calculating results: {str(e)}"}), 500


def run_agents(video_paths, audio_path, task_id, target_duration='30', apply_edits=True):
    """
    Real agent integration that calls specialists for analysis and composition.
    """
    overall_start = time.time()
    timing_log = {}
    
    API_KEY = os.environ.get('OPENROUTER_API_KEY')
    if not API_KEY:
        return {"status": "error", "error": "OPENROUTER_API_KEY not found in environment."}

    # Initialize Agents
    video_proc = VideoProcessor()
    audio_agent = AudioMatchAgent(api_key=API_KEY, mode=ProcessingMode.HYBRID)
    style_agent = StyleDirectorAgent(api_key=API_KEY, mode=ProcessingMode.API_ONLY)
    director = CreativeDirectorAgent(api_key=API_KEY, mode=ProcessingMode.API_ONLY)
    
    all_scenes = []
    video_styles = []
    
    # 1. Process Videos (with caching)
    video_start = time.time()
    for v_path in video_paths:
        v_hash = default_cache.hash_file(v_path)
        # Add a suffix to hash for fine-grained analysis to avoid cache collision with old data
        v_hash_fg = f"{v_hash}_finegrained_v1"
        cached_v = default_cache.get_cached_result(v_hash_fg)
        
        if cached_v:
            cache_time = time.time() - video_start
            timing_log[f"video_{os.path.basename(v_path)}_cache_hit"] = f"{cache_time:.2f}s"
            all_scenes.extend(cached_v.get("scenes", []))
            video_styles.append(cached_v.get("style"))
        else:
            # Step A: Technical Scene Detection (Histogram-based)
            scene_start = time.time()
            detected_scenes = video_proc.detect_scenes(v_path)
            timing_log[f"scene_detect_{os.path.basename(v_path)}"] = f"{time.time() - scene_start:.2f}s"
            
            # Step B: Temporal Frame Extraction (1 FPS)
            frame_start = time.time()
            all_frames = video_proc.extract_frames_at_fps(v_path, target_fps=1.0)
            timing_log[f"frame_extract_{os.path.basename(v_path)}"] = f"{time.time() - frame_start:.2f}s"
            
            # Step C: Chunking (10s segments)
            chunk_start = time.time()
            chunks = video_proc.get_video_chunks(all_frames, chunk_size=10)
            timing_log[f"chunk_{os.path.basename(v_path)}"] = f"{time.time() - chunk_start:.2f}s"
            
            # Step D: Parallel Chunk Analysis (Semantic)
            analysis_start = time.time()
            print(f"Starting parallel analysis of {len(chunks)} chunks for {os.path.basename(v_path)}...")
            chunk_summaries = style_agent.analyze_chunks_parallel(chunks, max_workers=5)
            timing_log[f"chunk_analysis_{os.path.basename(v_path)}"] = f"{time.time() - analysis_start:.2f}s"
            
            # Step E: Narrative Scene Synthesis
            synthesis_start = time.time()
            semantic_scenes = director.synthesize_scenes(chunk_summaries, detected_scenes)
            timing_log[f"scene_synthesis_{os.path.basename(v_path)}"] = f"{time.time() - synthesis_start:.2f}s"
            
            # Step F: Global Style Analysis (using a representative frame, e.g., the 3rd one)
            style_start = time.time()
            keyframe_path = all_frames[min(2, len(all_frames)-1)] if all_frames else None
            style_result = style_agent.full_analysis(keyframe_path) if keyframe_path else None
            style_data = style_result.to_dict() if style_result else None
            timing_log[f"style_analysis_{os.path.basename(v_path)}"] = f"{time.time() - style_start:.2f}s"
            
            # Cache the new fine-grained results
            video_data = {"scenes": semantic_scenes, "style": style_data}
            default_cache.save_result(v_hash_fg, video_data)
            timing_log[f"video_{os.path.basename(v_path)}_cached"] = "saved"
            
            all_scenes.extend(semantic_scenes)
            video_styles.append(style_data)

            # Cleanup temporary frames
            if all_frames:
                try:
                    import shutil
                    temp_dir = os.path.dirname(all_frames[0])
                    if "_temp_frames" in temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Cleanup failed: {e}")
    
    timing_log["total_video_processing"] = f"{time.time() - video_start:.2f}s"

    # 2. Process Audio (with caching)
    audio_start = time.time()
    audio_results = None
    if audio_path:
        a_hash = default_cache.hash_file(audio_path)
        cached_a = default_cache.get_cached_result(a_hash)
        
        if cached_a:
            timing_log["audio_cache_hit"] = f"{time.time() - audio_start:.2f}s"
            audio_results = cached_a
        else:
            # Real Audio Analysis
            audio_features_start = time.time()
            audio_features = audio_agent.analyze_audio_file(audio_path)
            timing_log["audio_feature_extract"] = f"{time.time() - audio_features_start:.2f}s"
            
            genre_mood_start = time.time()
            genre_mood = audio_agent.classify_genre_mood(audio_path)
            timing_log["audio_genre_mood"] = f"{time.time() - genre_mood_start:.2f}s"
            
            audio_results = {
                "analyzed": True,
                "bpm": audio_features.bpm,
                "energy": "High" if audio_features.energy > 0.2 else "Moderate",
                "mood": ", ".join(genre_mood.moods) if genre_mood.moods else "Unknown",
                "genres": genre_mood.genres
            }
            default_cache.save_result(a_hash, audio_results)
            timing_log["audio_cached"] = "saved"
    else:
        audio_results = {"analyzed": False, "bpm": 0, "energy": "N/A", "mood": "N/A"}
    
    timing_log["total_audio_processing"] = f"{time.time() - audio_start:.2f}s"

    # 3. Consolidate Style (Use the first video or average them)
    primary_style = video_styles[0] if video_styles else {
        "name": "Default", "mood": "Neutral", "category": "modern", "color_grading_notes": ""
    }

    # 4. Generate Creative Composition
    composition_start = time.time()
    composition = director.generate_composition(
        audio_results, 
        primary_style, 
        target_duration, 
        video_count=len(video_paths),
        source_videos=video_paths
    )
    timing_log["composition_generation"] = f"{time.time() - composition_start:.2f}s"
    
    # Build final response
    results = {
        "task_id": task_id,
        "status": "success",
        "tasks": {
            "upload": True,
            "scene_detection": True,
            "audio_analysis": audio_path is not None,
            "style_recommendation": True,
            "composition": True,
            "video_render": apply_edits
        },
        "results": {
            "scenes": all_scenes,
            "audio": audio_results,
            "style": primary_style,
            "composition": composition
        }
    }
    
    # 5. Apply Edits (Render)
    render_start = time.time()
    if apply_edits and video_paths:
        try:
            editor = VideoEditor()
            output_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                f"{task_id}_edited.mp4"
            )
            
            edit_result = editor.apply_edit_plan(
                video_paths=video_paths,
                audio_path=audio_path,
                edit_plan=composition,
                output_path=output_path,
                quality="high"
            )
            
            results["results"]["rendered_video"] = {
                "status": edit_result.get("status"),
                "output_path": edit_result.get("output_path"),
                "clips_used": edit_result.get("clips_used"),
                "style_applied": edit_result.get("style_applied"),
                "error": edit_result.get("error")
            }
            
        except Exception as e:
            results["results"]["rendered_video"] = {"status": "error", "error": str(e)}
    
    timing_log["video_render"] = f"{time.time() - render_start:.2f}s"
    
    # Add total timing
    overall_time = time.time() - overall_start
    timing_log["total_processing_time"] = f"{overall_time:.2f}s"
    results["timing"] = timing_log
    
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
