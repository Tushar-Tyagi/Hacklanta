document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('videoFile');
    const audioInput = document.getElementById('audioFile');
    const audioUrlInput = document.getElementById('audioUrl');
    const processBtn = document.getElementById('processBtn');
    const previewUrlBtn = document.getElementById('previewUrlBtn');

    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultsSection = document.getElementById('results-section');

    // Status elements
    const videoStatus = document.getElementById('video-status');
    const audioStatus = document.getElementById('audio-status');
    const urlStatus = document.getElementById('url-status');

    // Toggle buttons
    const toggleUpload = document.getElementById('toggleUpload');
    const toggleUrl = document.getElementById('toggleUrl');
    const audioUploadPanel = document.getElementById('audio-upload-panel');
    const audioUrlPanel = document.getElementById('audio-url-panel');

    // Track which audio source mode is active
    let audioSourceMode = 'upload'; // 'upload' or 'url'

    // --- Audio Source Toggle ---
    function switchAudioMode(mode) {
        audioSourceMode = mode;

        if (mode === 'upload') {
            toggleUpload.classList.add('active');
            toggleUrl.classList.remove('active');
            audioUploadPanel.classList.remove('hidden');
            audioUrlPanel.classList.add('hidden');
            // Clear URL input when switching to upload
            audioUrlInput.value = '';
            urlStatus.textContent = '';
            urlStatus.className = 'file-status';
        } else {
            toggleUrl.classList.add('active');
            toggleUpload.classList.remove('active');
            audioUrlPanel.classList.remove('hidden');
            audioUploadPanel.classList.add('hidden');
            // Clear file input when switching to URL
            audioInput.value = '';
            audioStatus.textContent = 'No file selected';
            audioStatus.style.color = 'var(--text-light)';
        }
    }

    toggleUpload.addEventListener('click', () => switchAudioMode('upload'));
    toggleUrl.addEventListener('click', () => switchAudioMode('url'));

    // --- File Selection Handlers ---
    videoInput.addEventListener('change', function () {
        if (this.files && this.files.length > 0) {
            const count = this.files.length;
            videoStatus.textContent = `${count} video${count > 1 ? 's' : ''} selected`;
            videoStatus.style.color = 'var(--success-color)';

            // Render file list
            const videoList = document.getElementById('selected-video-list');
            videoList.innerHTML = '';
            videoList.classList.remove('hidden');

            Array.from(this.files).forEach(file => {
                const size = (file.size / (1024 * 1024)).toFixed(1);
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <i class="fas fa-file-video"></i>
                    <span class="file-name">${file.name}</span>
                    <span class="file-size">${size} MB</span>
                `;
                videoList.appendChild(fileItem);
            });

            processBtn.disabled = false;
        } else {
            videoStatus.textContent = 'No files selected';
            videoStatus.style.color = 'var(--text-light)';
            const videoList = document.getElementById('selected-video-list');
            if (videoList) videoList.classList.add('hidden');
            processBtn.disabled = true;
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            audioStatus.textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / (1024 * 1024)).toFixed(2)} MB)`;
            audioStatus.style.color = 'var(--success-color)';
        } else {
            audioStatus.textContent = 'No file selected (Optional)';
            audioStatus.style.color = 'var(--text-light)';
        }
    });

    // --- URL Verification ---
    previewUrlBtn.addEventListener('click', async () => {
        const url = audioUrlInput.value.trim();
        if (!url) {
            urlStatus.textContent = 'Please enter a URL first';
            urlStatus.className = 'file-status error';
            return;
        }

        previewUrlBtn.disabled = true;
        urlStatus.textContent = '⏳ Verifying and downloading audio...';
        urlStatus.className = 'file-status loading';

        try {
            const response = await fetch('/api/download-audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });

            const data = await response.json();

            if (!response.ok) {
                urlStatus.textContent = `✗ ${data.error}`;
                urlStatus.className = 'file-status error';
                return;
            }

            const meta = data.metadata;
            urlStatus.textContent = `✓ "${meta.title}" by ${meta.uploader} (${meta.platform}, ${formatDuration(meta.duration)})`;
            urlStatus.className = 'file-status success';

        } catch (error) {
            urlStatus.textContent = `✗ Connection error: ${error.message}`;
            urlStatus.className = 'file-status error';
        } finally {
            previewUrlBtn.disabled = false;
        }
    });

    // --- Form Submission ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!videoInput.files.length) {
            alert('Please select a video file.');
            return;
        }

        // Setup UI for processing
        processBtn.disabled = true;
        progressContainer.classList.remove('hidden');
        resultsSection.innerHTML = '';
        resultsSection.classList.add('hidden');

        // Form Data
        const formData = new FormData();

        // Append all video files
        if (videoInput.files.length > 0) {
            Array.from(videoInput.files).forEach(file => {
                formData.append('video', file);
            });
        }

        const targetDuration = document.getElementById('targetDuration').value;
        formData.append('target_duration', targetDuration);

        // Add audio based on current mode
        if (audioSourceMode === 'upload' && audioInput.files.length > 0) {
            formData.append('audio', audioInput.files[0]);
        } else if (audioSourceMode === 'url' && audioUrlInput.value.trim()) {
            formData.append('audio_url', audioUrlInput.value.trim());
        }

        try {
            updateProgress(10, 'Uploading files...', 'step-upload');

            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Processing failed');
            }

            // Simulate progress steps
            updateProgress(30, 'Detecting scenes...', 'step-scene');
            await new Promise(r => setTimeout(r, 1000));

            updateProgress(50, 'Analyzing audio...', 'step-audio');
            await new Promise(r => setTimeout(r, 1000));

            updateProgress(70, 'Recommending styles...', 'step-style');
            await new Promise(r => setTimeout(r, 1000));

            updateProgress(90, 'Composing final edit...', 'step-compose');

            const result = await response.json();

            updateProgress(100, 'Processing complete!', null);
            displayResults(result);

        } catch (error) {
            console.error('Error:', error);
            progressText.textContent = `Error: ${error.message}`;
            progressText.style.color = 'var(--error-color)';
            progressBar.style.backgroundColor = 'var(--error-color)';
        } finally {
            processBtn.disabled = false;
        }
    });

    // --- Helpers ---
    function formatDuration(seconds) {
        if (!seconds) return 'Unknown duration';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function updateProgress(percent, text, stepId) {
        progressBar.style.width = `${percent}%`;
        progressText.textContent = text;

        document.querySelectorAll('.step-list li').forEach(li => {
            if (li.classList.contains('active')) {
                li.classList.replace('active', 'completed');
            }
        });

        if (stepId) {
            const step = document.getElementById(stepId);
            if (step) {
                step.classList.remove('pending', 'completed');
                step.classList.add('active');
            }
        }
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        const header = document.createElement('h2');
        header.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success-color)"></i> Processing Complete';
        resultsSection.appendChild(header);

        // --- Task Pipeline Summary ---
        if (data.tasks) {
            const tasksDiv = document.createElement('div');
            tasksDiv.className = 'card result-card';
            tasksDiv.innerHTML = `
                <h3><i class="fas fa-tasks"></i> Pipeline Summary</h3>
                <div class="pipeline-steps">
                    ${Object.entries(data.tasks).map(([key, val]) => `
                        <div class="pipeline-step ${val ? 'success' : 'failed'}">
                            <i class="fas ${val ? 'fa-check-circle' : 'fa-times-circle'}"></i>
                            <span>${formatLabel(key)}</span>
                        </div>
                    `).join('')}
                </div>`;
            resultsSection.appendChild(tasksDiv);
        }

        // --- Audio Source (if from URL) ---
        if (data.audio_source) {
            const audioDiv = document.createElement('div');
            audioDiv.className = 'card result-card';
            audioDiv.innerHTML = `
                <h3><i class="fas fa-music"></i> Audio Source</h3>
                <div class="result-grid">
                    <div class="result-stat">
                        <span class="stat-label">Title</span>
                        <span class="stat-value">${data.audio_source.title}</span>
                    </div>
                    <div class="result-stat">
                        <span class="stat-label">Artist</span>
                        <span class="stat-value">${data.audio_source.uploader}</span>
                    </div>
                    <div class="result-stat">
                        <span class="stat-label">Platform</span>
                        <span class="stat-value">${data.audio_source.platform}</span>
                    </div>
                    <div class="result-stat">
                        <span class="stat-label">Duration</span>
                        <span class="stat-value">${formatDuration(data.audio_source.duration)}</span>
                    </div>
                </div>`;
            resultsSection.appendChild(audioDiv);
        }

        const r = data.results;
        if (!r) return;

        // --- Scenes ---
        if (r.scenes && r.scenes.length > 0) {
            const scenesDiv = document.createElement('div');
            scenesDiv.className = 'card result-card';
            scenesDiv.innerHTML = `
                <h3><i class="fas fa-film"></i> Scene Detection</h3>
                <p class="result-subtitle">${r.scenes.length} scene${r.scenes.length !== 1 ? 's' : ''} detected</p>
                <div class="scenes-timeline">
                    ${r.scenes.map(s => `
                        <div class="scene-block">
                            <div class="scene-header">
                                <span class="scene-type">${s.type || 'Scene'}</span>
                                <span class="scene-source"><i class="fas fa-paperclip"></i> ${s.source || 'Unknown'}</span>
                            </div>
                            <div class="scene-time">${(s.start || 0).toFixed(1)}s — ${(s.end || 0).toFixed(1)}s</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${((s.confidence || 0) * 100)}%"></div>
                                <span class="confidence-text">${((s.confidence || 0) * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    `).join('')}
                </div>`;
            resultsSection.appendChild(scenesDiv);
        }

        // --- Audio Analysis ---
        if (r.audio) {
            const audioDiv = document.createElement('div');
            audioDiv.className = 'card result-card';
            const a = r.audio;
            audioDiv.innerHTML = `
                <h3><i class="fas fa-headphones"></i> Audio Analysis</h3>
                ${a.analyzed ? `
                    <div class="result-grid">
                        <div class="result-stat highlight">
                            <span class="stat-label">BPM</span>
                            <span class="stat-value big">${a.bpm}</span>
                        </div>
                        <div class="result-stat highlight">
                            <span class="stat-label">Energy</span>
                            <span class="stat-value big">${a.energy}</span>
                        </div>
                        <div class="result-stat highlight">
                            <span class="stat-label">Mood</span>
                            <span class="stat-value big">${a.mood}</span>
                        </div>
                    </div>
                ` : `<p class="result-empty">No audio file provided — skipped analysis</p>`}`;
            resultsSection.appendChild(audioDiv);
        }

        // --- Style Recommendation ---
        if (r.style) {
            const styleDiv = document.createElement('div');
            styleDiv.className = 'card result-card';
            const s = r.style;

            // Get first LUT recommendation if available
            const lut = s.lut_recommendations && s.lut_recommendations.length > 0 ? s.lut_recommendations[0] : null;

            styleDiv.innerHTML = `
                <h3><i class="fas fa-palette"></i> Style Recommendation</h3>
                <div class="style-header-row">
                    <div>
                        <span class="style-name">${lut ? lut.name : s.mood || 'Custom'}</span>
                        <span class="style-category">${lut ? lut.category : s.temperature || 'modern'}</span>
                    </div>
                    <span class="style-confidence">${((s.confidence || 0) * 100).toFixed(0)}% match</span>
                </div>
                <p class="style-description">${lut ? lut.description : s.color_distribution ? Object.keys(s.color_distribution).join(', ') : 'Based on local color analysis'}</p>
                
                <div class="style-details">
                    <div class="style-detail">
                        <span class="stat-label">Contrast</span>
                        <span class="stat-value">${s.contrast_level || 'N/A'}</span>
                    </div>
                    <div class="style-detail">
                        <span class="stat-label">Saturation</span>
                        <span class="stat-value">${s.saturation_level || 'N/A'}</span>
                    </div>
                    <div class="style-detail">
                        <span class="stat-label">Temperature</span>
                        <span class="stat-value">${s.temperature || 'N/A'}</span>
                    </div>
                </div>

                ${s.dominant_tones && s.dominant_tones.length > 0 ? `
                    <div class="style-details">
                        <span class="stat-label">Dominant Tones</span>
                        <div class="palette-swatches">
                            ${s.dominant_tones.map(t => `
                                <div class="swatch" style="background-color: #888">
                                    <span class="swatch-label">${t}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${s.color_grading_notes ? `
                    <div class="grading-notes">
                        <span class="stat-label">Color Grading Notes</span>
                        <p>${s.color_grading_notes}</p>
                    </div>
                ` : ''}
                
                ${lut ? `
                    <div class="grading-notes">
                        <span class="stat-label">LUT Recommendation</span>
                        <p>${lut.color_grading_notes}</p>
                    </div>
                ` : ''}`;
            resultsSection.appendChild(styleDiv);
        }

        // --- Edit Composition ---
        if (r.composition && r.composition.edit_decisions) {
            const compDiv = document.createElement('div');
            compDiv.className = 'card result-card';
            compDiv.innerHTML = `
                <h3><i class="fas fa-cut"></i> Edit Composition</h3>
                <p class="result-subtitle">${r.composition.edit_decisions.length} edit decision${r.composition.edit_decisions.length !== 1 ? 's' : ''}</p>
                <div class="edit-timeline">
                    ${r.composition.edit_decisions.map((d, i) => `
                        <div class="edit-decision">
                            <div class="edit-marker">${i + 1}</div>
                            <div class="edit-info">
                                <div class="edit-type">${d.cut_type}</div>
                                <div class="edit-meta">
                                    <span><i class="fas fa-clock"></i> ${d.time.toFixed(1)}s</span>
                                    <span class="edit-reason">${d.reason}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>`;
            resultsSection.appendChild(compDiv);
        }

        // --- Timing Info ---
        if (data.timing) {
            const timingDiv = document.createElement('div');
            timingDiv.className = 'card result-card';

            // Build timing list
            let timingRows = '';
            for (const [key, value] of Object.entries(data.timing)) {
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                timingRows += `
                    <div class="timing-row">
                        <span class="timing-label">${label}</span>
                        <span class="timing-value">${value}</span>
                    </div>`;
            }

            timingDiv.innerHTML = `
                <h3><i class="fas fa-stopwatch"></i> Processing Time</h3>
                <div class="timing-list">
                    ${timingRows}
                </div>`;
            resultsSection.appendChild(timingDiv);
        }

        // --- Raw JSON (collapsed) ---
        const detailsEl = document.createElement('details');
        detailsEl.className = 'card raw-json-details';
        detailsEl.innerHTML = `
            <summary><i class="fas fa-code"></i> Raw JSON Response</summary>
            <pre class="json-output">${JSON.stringify(data, null, 2)}</pre>`;
        resultsSection.appendChild(detailsEl);
    }

    function formatLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
});
