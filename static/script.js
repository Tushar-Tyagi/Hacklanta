document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const videoInput = document.getElementById('videoFile');
    const audioInput = document.getElementById('audioFile');
    const processBtn = document.getElementById('processBtn');
    
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultsSection = document.getElementById('results-section');
    
    // Status elements
    const videoStatus = document.getElementById('video-status');
    const audioStatus = document.getElementById('audio-status');

    // Update status text when files selected
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            videoStatus.textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / (1024 * 1024)).toFixed(2)} MB)`;
            videoStatus.style.color = 'var(--success-color)';
            processBtn.disabled = false;
        } else {
            videoStatus.textContent = 'No file selected';
            videoStatus.style.color = 'var(--text-light)';
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

    // Form submission processing
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
        formData.append('video', videoInput.files[0]);
        if (audioInput.files.length > 0) {
            formData.append('audio', audioInput.files[0]);
        }

        try {
            updateProgress(10, 'Uploading files...', 'step-upload');
            
            // Note: Polling mechanism should ideally be implemented to update progress
            // based on backend task ID. For now, we simulate progress during fetch.
            
            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Processing failed');
            }

            // Simulate progress steps (In real app, backend would stream progress or we'd poll)
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

    function updateProgress(percent, text, stepId) {
        progressBar.style.width = `${percent}%`;
        progressText.textContent = text;
        
        // Reset all active
        document.querySelectorAll('.step-list li').forEach(li => {
            if(li.classList.contains('active')) {
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
        
        // Build generic JSON view for now, customized UI can be built later
        const header = document.createElement('h2');
        header.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success-color)"></i> Processing Complete';
        resultsSection.appendChild(header);

        if(data.tasks) {
            const tasksDiv = document.createElement('div');
            tasksDiv.className = 'card';
            tasksDiv.innerHTML = `<h3>Task Summary</h3><ul>` +
                Object.keys(data.tasks).map(t => `<li><strong>${t}:</strong> ${data.tasks[t] ? 'Success' : 'Failed'}</li>`).join('') +
                `</ul>`;
            resultsSection.appendChild(tasksDiv);
        }

        const jsonView = document.createElement('div');
        jsonView.className = 'card json-output';
        jsonView.textContent = JSON.stringify(data, null, 2);
        resultsSection.appendChild(jsonView);
    }
});
