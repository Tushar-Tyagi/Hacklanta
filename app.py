import os
import secrets
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mkv', 'avi', 'mov', 'm4v'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'ogg', 'flac'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_media():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video_file = request.files['video']
    audio_file = request.files.get('audio')
    
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400
        
    if not allowed_file(video_file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid video format. Allowed formats: ' + ', '.join(ALLOWED_VIDEO_EXTENSIONS)}), 400
        
    # Generate unique IDs for the files to prevent overriding and simplify tracking
    task_id = secrets.token_hex(8)
    
    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{video_filename}")
    video_file.save(video_path)
    
    audio_path = None
    if audio_file and audio_file.filename != '':
        if not allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({'error': 'Invalid audio format. Allowed formats: ' + ', '.join(ALLOWED_AUDIO_EXTENSIONS)}), 400
        audio_filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{audio_filename}")
        audio_file.save(audio_path)

    # In a real app, this would start a background task (e.g. Celery).
    # Since we need synchronous response or a simple polling interface, 
    # we'll run mock processing here that simulates calling the agents.
    
    try:
        # Placeholder for agent calls
        results = run_agents(video_path, audio_path, task_id)
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({'error': f"Internal server error calculating results: {str(e)}"}), 500


def run_agents(video_path, audio_path, task_id):
    """
    Mock integration of the AI agents.
    In real implementation, imports and initializes:
    from agents.style_director import StyleDirectorAgent
    from agents.audio_match_agent import AudioMatchAgent
    """
    # This simulates what the agents would output
    
    return {
        "task_id": task_id,
        "status": "success",
        "tasks": {
            "upload": True,
            "scene_detection": True,
            "audio_analysis": audio_path is not None,
            "style_recommendation": True,
            "composition": True
        },
        "results": {
            "scenes": [
                {"id": 1, "start": 0.0, "end": 4.5, "type": "Establishing Shot", "confidence": 0.88},
                {"id": 2, "start": 4.5, "end": 12.0, "type": "Close Up", "confidence": 0.92},
                {"id": 3, "start": 12.0, "end": 18.2, "type": "Action Sequence", "confidence": 0.85}
            ],
            "audio": {
                "analyzed": audio_path is not None,
                "bpm": 120.5 if audio_path else 0,
                "energy": "High" if audio_path else "N/A",
                "mood": "Energetic/Upbeat" if audio_path else "N/A"
            },
            "style": {
                "name": "Cinematic Teal & Orange",
                "description": "High contrast cinematic look emphasizing skin tones against cool shadows.",
                "category": "cinematic",
                "intensity": 0.8,
                "confidence": 0.95,
                "palette": ["#113a5d", "#1c5d99", "#f4a261", "#e76f51"],
                "color_grading_notes": "Boost saturation in midtones. Cool down the shadows specifically targeting the blue/cyan channels."
            },
            "composition": {
                "edit_decisions": [
                    {"cut_type": "Straight Cut", "time": 4.5, "reason": "Beat drop alignment"},
                    {"cut_type": "J-Cut", "time": 12.0, "reason": "Smooth transition to action"}
                ]
            }
        }
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
