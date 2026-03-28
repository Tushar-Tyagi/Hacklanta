import os
import io
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = 'test_uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.test_client() as client:
        yield client
    # Cleanup after tests
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    os.rmdir(app.config['UPLOAD_FOLDER'])


def test_index_page(client):
    """Test that the index page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'AI Video Director' in response.data


def test_process_no_video(client):
    """Test API rejects request without video file."""
    response = client.post('/api/process', data={})
    assert response.status_code == 400
    assert b'No video file provided' in response.data


def test_process_invalid_video_format(client):
    """Test API rejects invalid video extensions."""
    data = {
        'video': (io.BytesIO(b"dummy image data"), 'test.jpg')
    }
    response = client.post('/api/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert b'Invalid video format' in response.data


def test_process_valid_video(client):
    """Test processing a valid video file mock."""
    data = {
        'video': (io.BytesIO(b"dummy video data"), 'test_movie.mp4')
    }
    response = client.post('/api/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    json_data = response.get_json()
    assert 'task_id' in json_data
    assert json_data['status'] == 'success'
    assert json_data['tasks']['upload'] is True
    assert json_data['tasks']['audio_analysis'] is False


def test_process_video_and_audio(client):
    """Test processing both video and audio files."""
    data = {
        'video': (io.BytesIO(b"dummy video data"), 'test_movie.mp4'),
        'audio': (io.BytesIO(b"dummy audio data"), 'soundtrack.mp3')
    }
    response = client.post('/api/process', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    json_data = response.get_json()
    assert json_data['status'] == 'success'
    assert json_data['tasks']['audio_analysis'] is True
    assert 'results' in json_data
    assert json_data['results']['audio']['analyzed'] is True
    assert json_data['results']['audio']['bpm'] > 0

