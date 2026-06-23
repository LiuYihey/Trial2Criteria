document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded");
    
    const form = document.getElementById('research-form');
    const titleInput = document.getElementById('research-title');
    const resultsContainer = document.getElementById('results-container');
    const loader = document.getElementById('loader');
    const submitButton = document.getElementById('submit-button');
    const downloadButton = document.getElementById('download-button');
    const placeholder = document.querySelector('#results-container .placeholder');
    const analysisStatus = document.getElementById('analysis-status');
    
    // 初始化页面中已有的可折叠元素和高亮功能
    initializeCollapsibleElements();
    initializeCriterionHighlighting();

    // Selected mode state (default to Standard)
    let selectedMode = "Standard";
    
    // Context data storage for the final criteria generation step
    let contextData = {};
    let currentTitle = "";

    const stepNames = [
        "Keyword Extraction", "RAG Expansion", "Disease Information", "Drug Information",
        "Relevant Papers", "Similar Trials", "Final Criteria Generation"
    ];

    // --- SVG Icons for each step ---
    const icons = {
        1: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6" /></svg>',
        2: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" /></svg>',
        3: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" /></svg>',
        4: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25 2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75m-18 0h18M-3 14.25h18" /></svg>',
        5: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>',
        6: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" /></svg>',
        7: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>',
        8: '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" /></svg>',
    };

    // Mode descriptions
    const modeDescriptions = {
        "Standard": "Balance scientific validity, patient safety, and recruitment feasibility.",
        "Broader": "Maximize real-world applicability and diversity while maintaining essential safety.",
        "Stringent": "Reduce confounding and maximize signal detection for confirmatory efficacy."
    };

    // Setup mode options click events
    // Removed modePopup and modeOptions as they are no longer used

    // Pre-select Standard mode
    // Removed modePopup.style.display = 'none';

    // Generate button click event
    // Removed generateButton.addEventListener('click', ...)

    let eventSource;

    downloadButton.addEventListener('click', () => {
        window.location.href = '/download-latest-log';
    });

    form.addEventListener('submit', (event) => {
        event.preventDefault();

        const title = titleInput.value.trim();
        if (!title) {
            alert('Please enter a research title.');
            return;
        }

        currentTitle = title;
        
        // Reset context data
        contextData = {};

        if (eventSource) {
            eventSource.close();
        }

        resultsContainer.innerHTML = '';
        loader.style.display = 'block';
        submitButton.disabled = true;
        submitButton.textContent = 'Analyzing...';
        if(placeholder) placeholder.style.display = 'none';

        // Show and update analysis status bar
        analysisStatus.style.display = 'flex';
        analysisStatus.innerHTML = `<div class="mini-loader"></div> Now processing... Next: Step 1 - ${stepNames[0]}`;

        // Replace EventSource with fetch for POST request capability
        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                title: title,
                // TODO: Add a UI element to select the year dynamically
                until_year: new Date().getFullYear() 
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            // Process the streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            function push() {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        resetUI();
                        return;
                    }
                    
                    // Decode the chunk of data and process it
                    const chunk = decoder.decode(value, { stream: true });
                    const lines = chunk.split('\n\n');

                    lines.forEach(line => {
                        if (line.startsWith('data: ')) {
                            const jsonData = line.substring(6);
                            if (jsonData.trim()) {
                                const result = JSON.parse(jsonData);
                                handleStreamedData(result);
                            }
                        }
                    });
                    
                    push(); // Continue reading the stream
                }).catch(err => {
                    console.error('Stream reading error:', err);
                    displayError('Connection to the server was lost. Please try again.');
                    resetUI();
                });
            }
            push();
        })
        .catch(err => {
            console.error('Fetch error:', err);
            displayError(err.message);
            resetUI();
        });
    });

    // Initialize a flag to track if it's the first data chunk
    let isFirstDataChunk = true;

    // 在handleStreamedData函数中，简化判断条件，使其更容易匹配最后一步
    function handleStreamedData(result) {
        // On the very first data chunk, hide the main loader
        if (isFirstDataChunk) {
            loader.style.display = 'none';
            isFirstDataChunk = false; // Unset the flag
        }

        if (result.error) {
            displayError(result.error);
            // Since we don't have an eventSource to close, we just stop processing
            resetUI();
            return;
        }

        // Special handling for mode selection - create an embedded mode selection card
        if (result.data && result.data.waiting_for_mode === true) {
            
            // Store context data for final generation
            storeContextData(result);
            
            // Create a standard result card
            const card = document.createElement('div');
            card.className = 'result-card';
            
            const header = document.createElement('div');
            header.className = 'result-card-header';
            
            // Use the proper icon
            const iconSvg = icons[result.step] || icons[7];

            header.innerHTML = `
                <div class="icon">${iconSvg}</div>
                <h3>Step ${result.step}: ${result.name}</h3>
            `;

            // Create the content div with mode selection UI
            const contentDiv = document.createElement('div');
            contentDiv.className = 'result-card-content';
            
            // Create embedded mode selection UI (similar to the popup but embedded)
            contentDiv.innerHTML = `
                <h3 style="margin-bottom: 20px; text-align: center;">Please select a criteria generation mode to continue:</h3>
                <div class="mode-options">
                    <div class="mode-option" data-mode="Broader">
                        <div class="mode-icon broader-icon">🌍</div>
                        <h3>Broader</h3>
                        <p>Maximize real-world applicability and diversity while maintaining essential safety.</p>
                    </div>
                    
                    <div class="mode-option" data-mode="Standard">
                        <div class="mode-icon standard-icon">⚖️</div>
                        <h3>Standard</h3>
                        <p>Balance scientific validity, patient safety, and recruitment feasibility.</p>
                    </div>
                    
                    <div class="mode-option" data-mode="Stringent">
                        <div class="mode-icon stringent-icon">🔍</div>
                        <h3>Stringent</h3>
                        <p>Reduce confounding and maximize signal detection for confirmatory efficacy.</p>
                    </div>
                </div>
                <button id="embedded-generate-criteria-button" class="generate-button">Generate Criteria</button>
            `;

            card.appendChild(header);
            card.appendChild(contentDiv);
            resultsContainer.appendChild(card);

            // Add event listeners to the embedded mode options
            const embeddedModeOptions = contentDiv.querySelectorAll('.mode-option');
            embeddedModeOptions.forEach(option => {
                option.addEventListener('click', () => {
                    
                    // Update selected state
                    embeddedModeOptions.forEach(opt => opt.classList.remove('selected'));
                    option.classList.add('selected');
                    
                    // Update selected mode
                    selectedMode = option.getAttribute('data-mode');
                });
            });
            
            // Pre-select Standard mode
            contentDiv.querySelector('.mode-option[data-mode="Standard"]').classList.add('selected');
            
            // Add event listener to the generate button
            const embeddedGenerateButton = contentDiv.querySelector('#embedded-generate-criteria-button');
            embeddedGenerateButton.addEventListener('click', () => {
                
                // Update button state
                embeddedGenerateButton.classList.add('loading');
                embeddedGenerateButton.textContent = 'Generating...';
                embeddedGenerateButton.disabled = true;
                
                // Generate criteria with selected mode
                generateCriteriaWithMode();
            });
            
            // Update status bar
            analysisStatus.innerHTML = `<div class="mini-loader"></div> Select a mode to continue`;
            
            return; // Stop here, we've created our card
        }

        // Update status bar for the next step
        if (result.next_step_name) {
            if (result.next_step_name === "Analysis Complete!") {
                analysisStatus.innerHTML = `
                    <div class="status-content">
                        <div class="icon">${icons[7] || ''}</div>
                        <span>${result.next_step_name}</span>
                    </div>
                `;
            } else {
                analysisStatus.innerHTML = `
                    <div class="status-content">
                        <div class="mini-loader"></div>
                        <span>${result.next_step_name}</span>
                    </div>
                `;
            }
        }
        
        // Store context data for final generation
        storeContextData(result);
        
        // Default card rendering for other steps
        let card = resultsContainer.querySelector(`#step-${result.step}`);
        if (!card) {
            card = document.createElement('div');
            card.id = `step-${result.step}`;
            card.className = 'result-card';

            const header = document.createElement('div');
            header.className = 'result-card-header';
            
            // Use a default icon if step number is out of bounds
            const iconSvg = icons[result.step] || icons[4];

            header.innerHTML = `
                <div class="icon">${iconSvg}</div>
                <h3>Step ${result.step}: ${result.name}</h3>
            `;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'result-card-content';
            
            // 简化条件判断，如果是最后一步或相关步骤，使用renderCriteriaSet
            if (result.next_step_name === "Analysis Complete!" && result.data) {
                contentDiv.innerHTML = renderCriteriaSet(result.data, selectedMode);
            } else {
                contentDiv.innerHTML = formatDataAsHtml(result);
            }

            card.appendChild(header);
            card.appendChild(contentDiv);
        }
        
        resultsContainer.appendChild(card);

        // Scroll to the new card
        card.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }

    // Store relevant context data for the final criteria generation
    function storeContextData(result) {
        // Store disease info (step 3)
        if (result.step === 3 && typeof result.data === 'object') {
            contextData.disease_info = result.data;
        }
        
        // Store drug info (steps 4+)
        if (result.name && result.name.startsWith('Drug Information') && typeof result.data === 'string') {
            if (!contextData.drug_info) contextData.drug_info = [];
            contextData.drug_info.push(result.data);
        }
        
        // Store paper info (step with name "Searching for Relevant Papers")
        if (result.name === 'Searching for Relevant Papers' && Array.isArray(result.data)) {
            contextData.papers_data = result.data;
        }
        
        // Store trial data (step with name "Retrieving Similar Clinical Trials")
        if (result.name === 'Retrieving Similar Clinical Trials' && typeof result.data === 'object') {
            contextData.trials_data = result.data;
        }
        
        // Store description and outcome from step 2
        if (result.step === 2 && typeof result.data === 'object') {
            contextData.description = result.data.description;
            contextData.primary_outcome = result.data.primary_outcome;
        }
    }

    // Initialize variables and listeners
    document.addEventListener('DOMContentLoaded', function() {
        // Add click listener to hide status bar
        analysisStatus.addEventListener('click', function() {
            analysisStatus.classList.add('hidden');
        });
    });

    // Handle completion of criteria generation
    function handleCriteriaGenerated() {
        // Update status to completion
        analysisStatus.innerHTML = `
            <div class="status-content">
                <div class="icon">${icons[8] || ''}</div>
                <span>Analysis Complete! Criteria Generated Successfully.</span>
            </div>
        `;
        
        // Also ensure generate button is reset
        resetGenerateButton();
        
        // Scroll to the last result card (the criteria)
        setTimeout(() => {
            const resultCards = document.querySelectorAll('.result-card');
            if (resultCards.length > 0) {
                const lastCard = resultCards[resultCards.length - 1];
                lastCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            
            // Auto-hide status bar after 5 seconds
            setTimeout(() => {
                analysisStatus.classList.add('hidden');
            }, 5000);
        }, 300);
    }
    
    // Generate criteria with selected mode
    function generateCriteriaWithMode() {
        // 获取当前最高步骤编号
        let maxStepNumber = 0;
        document.querySelectorAll('.result-card').forEach(card => {
            const cardId = card.id;
            if (cardId && cardId.startsWith('step-')) {
                const stepNum = parseInt(cardId.replace('step-', ''));
                if (!isNaN(stepNum) && stepNum > maxStepNumber) {
                    maxStepNumber = stepNum;
                }
            }
        });
        
        // Show loader and update status
        loader.style.display = 'block';
        analysisStatus.innerHTML = `
            <div class="status-content">
                <div class="mini-loader"></div>
                <span>Generating criteria with ${selectedMode} mode...</span>
            </div>
        `;
        
        // Make API call to generate criteria with selected mode
        fetch('/generate_criteria', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: currentTitle,
                mode: selectedMode,
                context: contextData,
                current_step: maxStepNumber // 传递当前最高步骤编号
            }),
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`Failed to generate criteria (${response.status}): ${text}`);
                });
            }
            return response.json();
        })
        .then(result => {
            loader.style.display = 'none';
            handleStreamedData(result);
            handleCriteriaGenerated(); // Mark as completed
            resetUI();
            resetGenerateButton(); // Reset generate criteria button state
        })
        .catch(err => {
            console.error('Generation error:', err);
            displayError('Failed to generate criteria: ' + err.message);
            resetUI();
            resetGenerateButton(); // Reset generate criteria button even on error
        });
    }

    function displayError(message) {
        if(loader) loader.style.display = 'none';
        const errorCard = document.createElement('div');
        errorCard.className = 'result-card';
        errorCard.innerHTML = `
            <div class="result-card-header" style="color: #c0392b;">
                <div class="icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z" /></svg></div>
                <h3>An Error Occurred</h3>
            </div>
            <div class="result-card-content"><p>${message}</p></div>
        `;
        resultsContainer.appendChild(errorCard);
    }

    function formatDataAsHtml(result) {
        const data = result.data;
        const step = result.step;
        const name = result.name;

        // Completely skip rendering for waiting_for_mode objects
        if (data && typeof data === 'object' && data.waiting_for_mode === true) {
            return ''; // Return empty string to prevent rendering
        }

        if (data === null || data === undefined || (typeof data === 'object' && Object.keys(data).length === 0)) {
            return `<p>No data available for this step.</p>`;
        }

        // --- Custom Renderers for specific steps based on name ---
        if (name && name.startsWith('Drug Information')) {
            if (typeof data === 'string') {
                // Restore the card-based view for each drug feature
                let html = '<div class="drug-features-grid">';
                const lines = data.split('\n');
                lines.forEach(line => {
                    if (line.startsWith('* ')) {
                        const content = line.substring(2);
                        const delimiterPos = content.indexOf(':');
                        if (delimiterPos !== -1) {
                            const key = content.substring(0, delimiterPos).trim();
                            const value = content.substring(delimiterPos + 1).trim();
                            if (key && value) {
                                html += `
                                    <div class="feature-card">
                                        <div class="feature-key">${key}</div>
                                        <div class="feature-value">${value}</div>
                                    </div>
                                `;
                            }
                        }
                    }
                });
                html += '</div>';
                return html;
            }
        }
        
        if (name && name === 'Searching for Relevant Papers') {
            if (Array.isArray(data)) {
                let listHtml = '<ul>';
                data.forEach(paper => {
                    listHtml += `<li>
                        <p class="item-title">${paper.title || 'N/A'}</p>
                        <p><strong>PMID:</strong> ${paper.pmid || 'N/A'}</p>
                        <p><strong>Score:</strong> ${paper.score ? paper.score.toFixed(4) : 'N/A'}</p>
                        <p class="item-abstract"><strong>Abstract:</strong> ${paper.abstract || 'N/A'}</p>
                    </li>`;
                });
                listHtml += '</ul>';
                return listHtml;
            } else {
                 return '<p>No relevant papers found.</p>';
            }
        }
        
        // 简化对最终标准生成的处理
        // 删除特定标准生成步骤的复杂条件判断，让所有内容都能显示出来
        if (typeof data === 'object' && (data.InclusionCriteria || data.ExclusionCriteria || data.Reasoning)) {
            return renderCriteriaSet(data, selectedMode);
        }

        // --- Generic Fallback Renderer ---
        if (typeof data === 'string') {
            return `<p>${data.replace(/\n/g, '<br>')}</p>`;
        }
        if (typeof data !== 'object') {
            return `<p>${data}</p>`;
        }

        const toTitleCase = (str) => str.replace(/_/g, ' ').replace(/\w\S*/g, (txt) => txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase());

        let html = '<dl>';
        for (const key in data) {
            if (Object.prototype.hasOwnProperty.call(data, key)) {
                const value = data[key];
                html += `<dt>${toTitleCase(key)}</dt>`;
                html += '<dd>';
                if (value === null || value === undefined) {
                    html += 'N/A';
                } else if (Array.isArray(value)) {
                    if (value.length === 0) {
                        html += '<i>(empty list)</i>';
                    } else {
                        html += '<ul class="nested-list">';
                        value.forEach(item => {
                            html += `<li>${formatDataAsHtml({ step: step, data: item, name: name })}</li>`;
                        });
                        html += '</ul>';
                    }
                } else if (typeof value === 'object') {
                    html += formatDataAsHtml({ step: step, data: value, name: name });
                } else {
                    html += String(value).replace(/\n/g, '<br>');
                }
                html += '</dd>';
            }
        }
        html += '</dl>';
        return html;
    }
    
    // Reset UI elements
    function resetUI() {
        submitButton.disabled = false;
        submitButton.textContent = 'Analyze';
    }

    // Reset the generate criteria button state
    function resetGenerateButton() {
        // Find all generate criteria buttons and reset them
        const generateButtons = document.querySelectorAll('#embedded-generate-criteria-button');
        generateButtons.forEach(button => {
            button.classList.remove('loading');
            button.textContent = 'Generated';
            button.disabled = true; // Keep it disabled since generation is complete
        });
    }

    // Helper function to render a criteria set with reasoning
    function renderCriteriaSet(data, mode) {
        // 简化数据提取逻辑
        let criteriaData = data;
        let reasoningContent = null;
        
        // 尝试从响应中提取推理过程
        if (data && data.Reasoning) {
            reasoningContent = data.Reasoning;
            console.log("Found Reasoning content in field format");
        }
        
        let html = '';
        
        // 添加推理过程显示区域（如果有）
        if (reasoningContent) {
            html += `<div class="reasoning-container">
                <h3 class="reasoning-title">🤔 Reasoning Process</h3>
                <div class="reasoning-content">`;
            
            // 新的 formatReasoningSections 函数能处理所有情况
            html += formatReasoningSections(reasoningContent);
            
            html += `</div></div>`;
        }
        
        // 添加最终标准区域
        html += `<div class="criteria-set final-criteria-set">
            <div class="criteria-container">`;
        
        // Process Inclusion Criteria
        const inclusionCriteria = criteriaData.InclusionCriteria || [];
        if (inclusionCriteria.length > 0) {
            html += '<div class="criteria-section inclusion-criteria">';
            html += '<h3>Inclusion Criteria</h3>';
            html += '<ul>';
            
            inclusionCriteria.forEach((item, idx) => {
                let criterionText = '';
                
                if (typeof item === 'object') {
                    criterionText = item.Criterion || item.criterion || '';
                } else if (typeof item === 'string') {
                    criterionText = item;
                }
                
                html += `<li>${criterionText}</li>`;
            });
            
            html += '</ul></div>';
        }
        
        // Process Exclusion Criteria
        const exclusionCriteria = criteriaData.ExclusionCriteria || [];
        if (exclusionCriteria.length > 0) {
            html += '<div class="criteria-section exclusion-criteria">';
            html += '<h3>Exclusion Criteria</h3>';
            html += '<ul>';
            
            exclusionCriteria.forEach((item, idx) => {
                let criterionText = '';
                
                if (typeof item === 'object') {
                    criterionText = item.Criterion || item.criterion || '';
                } else if (typeof item === 'string') {
                    criterionText = item;
                }
                
                html += `<li>${criterionText}</li>`;
            });
            
            html += '</ul></div>';
        }
        
        html += '</div>'; // End of criteria-container
        html += '</div>'; // End of criteria-set
        
        // Auto-scroll to the bottom to show the criteria
        html += `
        <script>
            setTimeout(() => {
                window.scrollTo({
                    top: document.body.scrollHeight,
                    behavior: 'smooth'
                });
                
                // 初始化折叠/展开功能和高亮功能
                if (typeof initializeCollapsibleElements === 'function') {
                    initializeCollapsibleElements();
                }
                if (typeof initializeCriterionHighlighting === 'function') {
                    initializeCriterionHighlighting();
                }
            }, 300);
        </script>
        `;
        
        return html;
    }

    // 新的核心解析函数，取代旧的 formatAspectSections, parseReasoningSections, 等
    function formatReasoningSections(content) {
        let html = '';
        
        // Regex to split by "### Aspect..." or "### Draft..."
        // This is the updated, more robust splitter regex.
        const splitter = /(?=#{3}\s*(?:Aspect\s+\d+:|Draft))/i;
        const sections = content.split(splitter).filter(s => s.trim());

        if (sections.length === 0 && content.trim()) {
            // No sections found, just render the whole content
            return renderSectionInnerContent(content);
        }
        
        for (let i = 0; i < sections.length; i++) {
            const sectionContent = sections[i].trim();

            // Check if the section starts with a title.
            // This is the updated, more robust title-matching regex.
            const titleMatch = sectionContent.match(/^(#{3}\s*(?:Aspect\s+\d+:.*|Draft.*))(?=\n|$)/i);

            if (titleMatch) {
                // This is a titled section
                const title = titleMatch[1].trim();
                const innerContent = sectionContent.substring(titleMatch[0].length).trim();
                
                html += `<div class="reasoning-section">`;
                html += `<div class="reasoning-section-title">${title}</div>`;
                html += renderSectionInnerContent(innerContent);
                html += `</div>`;
            } else {
                // This is content without a title (should be the first part)
                html += renderSectionInnerContent(sectionContent);
            }
        }
        
        return html;
    }

    // 新的辅助函数，用于渲染一个section的内部内容
    function renderSectionInnerContent(content) {
        if (!content) return '';

        const splitter = /(💭\s*Rationale:|🔎\s*Search:|🧩\s*Deduce:|📝\s*Draft:)/i;
        const parts = content.split(splitter);

        let html = '';
        // The first part is any text before the first keyword
        if (parts[0] && parts[0].trim()) {
            html += formatReasoningText(parts[0].trim());
        }

        for (let i = 1; i < parts.length; i += 2) {
            const keyword = parts[i];
            const text = parts[i+1] ? parts[i+1].trim() : '';

            let keywordClass = '';
            if (/Rationale/i.test(keyword)) keywordClass = 'rationale';
            else if (/Search/i.test(keyword)) keywordClass = 'search';
            else if (/Deduce/i.test(keyword)) keywordClass = 'deduce';
            else if (/Draft/i.test(keyword)) keywordClass = 'draft';

            if (keywordClass) {
                 html += `<div class="reasoning-keyword ${keywordClass}">${keyword.trim()}</div>`;
            }
            if (text) {
                html += formatReasoningText(text);
            }
        }
        
        return html;
    }
    
    // 格式化推理文本为HTML (只处理转义和换行)
    function formatReasoningText(text) {
        if (!text) return '';
        
        return '<div>' + text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;')
            .replace(/\n/g, '<br>')
            + '</div>';
    }

    // 专门解析确定格式的JSON
    function renderExactCriteriaFormat(data, mode) {
        // Direct access to criteria data
        const inclusionCriteria = data["Inclusion Criteria"] || [];
        const exclusionCriteria = data["Exclusion Criteria"] || [];
        const traceability = data["Traceability Appendix"] || {};
        
        // Simplified direct layout without nested containers
        let html = `<div class="criteria-container-wrapper">
            <div class="criteria-container">`;
        
        // Left column: Inclusion Criteria
        html += '<div class="criteria-section inclusion-criteria">';
        html += '<h3>Inclusion Criteria</h3>';
        html += '<ul>';
        
        inclusionCriteria.forEach((item) => {
            // Extract citation references if present
            const parts = item.split(/(\[\d+\])/g);
            let criterionText = '';
            let citations = '';
            
            if (parts.length > 1) {
                criterionText = parts[0].trim();
                citations = parts.slice(1).join('');
            } else {
                criterionText = item;
            }
            
            if (citations) {
                html += `<li>${criterionText} <span class="citation" title="Click to view evidence">${citations}</span></li>`;
            } else {
                html += `<li>${criterionText}</li>`;
            }
        });
        
        html += '</ul></div>';
        
        // Right column: Exclusion Criteria
        html += '<div class="criteria-section exclusion-criteria">';
        html += '<h3>Exclusion Criteria</h3>';
        html += '<ul>';
        
        exclusionCriteria.forEach((item) => {
            // Extract citation references if present
            const parts = item.split(/(\[\d+\])/g);
            let criterionText = '';
            let citations = '';
            
            if (parts.length > 1) {
                criterionText = parts[0].trim();
                citations = parts.slice(1).join('');
            } else {
                criterionText = item;
            }
            
            if (citations) {
                html += `<li>${criterionText} <span class="citation" title="Click to view evidence">${citations}</span></li>`;
            } else {
                html += `<li>${criterionText}</li>`;
            }
        });
        
        html += '</ul></div>';
        
        html += '</div>'; // End of criteria-container
        
        // Process Traceability Appendix
        if (Object.keys(traceability).length > 0) {
            html += '<div class="traceability-items">';
            
            for (const [tag, info] of Object.entries(traceability)) {
                const cleanTag = tag.replace(/[\[\]]/g, '');
                html += `<div class="traceability-item" id="evidence-${cleanTag}">`;
                html += `<div class="traceability-tag">${tag}</div>`;
                html += `<div class="traceability-content">`;
                
                // Sources
                if (info.EvidenceSources) {
                    const sources = Array.isArray(info.EvidenceSources) 
                        ? info.EvidenceSources.join(', ') 
                        : info.EvidenceSources;
                    html += `<div><strong>Sources:</strong> ${sources}</div>`;
                }
                
                // Snippets - Simplified
                if (info.EvidenceSnippets) {
                    html += `<div class="evidence-container">`;
                    if (Array.isArray(info.EvidenceSnippets)) {
                        info.EvidenceSnippets.forEach(snippet => {
                            html += `<div class="evidence-snippet">${snippet}</div>`;
                        });
                    } else {
                        html += `<div class="evidence-snippet">${info.EvidenceSnippets}</div>`;
                    }
                    html += `</div>`;
                }
                
                // Rationale
                if (info.Rationale) {
                    html += `<div><strong>Rationale:</strong> ${info.Rationale}</div>`;
                }
                
                html += `</div>`; // End of traceability-content
                html += `</div>`; // End of traceability-item
            }
            
            html += '</div>'; // End of traceability-items
        }
        
        html += '</div>'; // End criteria-container-wrapper
        
        // Citation interaction JavaScript
        html += `
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                document.querySelectorAll('.citation').forEach(citation => {
                    citation.addEventListener('click', function(e) {
                        // Get the tag number from the citation text
                        const tagMatch = citation.textContent.match(/\\[(\\d+)\\]/);
                        if (tagMatch) {
                            e.preventDefault();
                            const tag = tagMatch[1];
                            
                            // Find the corresponding evidence element
                            const evidenceSelector = \`#evidence-\${tag}\`;
                            const evidenceElement = document.querySelector(evidenceSelector);
                            
                            if (evidenceElement) {
                                // Add click animation to citation
                                citation.classList.add('clicked');
                                setTimeout(() => {
                                    citation.classList.remove('clicked');
                                }, 600);
                                
                                // Scroll to the evidence and highlight it
                                evidenceElement.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'center'
                                });
                                
                                evidenceElement.classList.add('highlight-evidence');
                                setTimeout(() => {
                                    evidenceElement.classList.remove('highlight-evidence');
                                }, 3000);
                            }
                        }
                    });
                });
            });
        </script>
        `;
        
        return html;
    }
});

// =======================================
// 增强型临床试验标准展示功能
// =======================================

// 在页面加载完成后初始化额外功能
document.addEventListener('DOMContentLoaded', function() {
    // 初始化折叠功能
    initializeCollapsibleElements();
    initializeCriterionHighlighting();
    
    // 监听动态添加的内容
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                initializeCollapsibleElements();
                initializeCriterionHighlighting();
            }
        });
    });
    
    // 观察DOM变化
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});

// Initialize collapsible functionality
function initializeCollapsibleElements() {
    console.log("Initializing collapsible elements...");
    // Find all reasoning containers that haven't been initialized
    const reasoningContainers = document.querySelectorAll('.reasoning-container:not(.collapsible-initialized)');
    
    reasoningContainers.forEach(container => {
        // Mark as initialized
        container.classList.add('collapsible-initialized');
        
        // Get content element
        const content = container.querySelector('.reasoning-content');
        
        // Default to collapsed state
        content.style.display = 'none';
        
        // Create toggle button with collapsed state
        const toggleButton = document.createElement('button');
        toggleButton.className = 'collapsible-toggle';
        toggleButton.innerHTML = '<span class="toggle-icon">▼</span> Expand Reasoning';
        toggleButton.setAttribute('aria-expanded', 'false');
        
        // Insert button
        const title = container.querySelector('.reasoning-title');
        if (title) {
            title.appendChild(toggleButton);
        } else {
            container.insertBefore(toggleButton, content);
        }
        
        // Log for debugging
        console.log("Added collapse button to", container);
        
        // Add click event with direct DOM manipulation
        toggleButton.addEventListener('click', function(event) {
            console.log("Toggle button clicked");
            event.preventDefault();
            event.stopPropagation();
            
            const isExpanded = toggleButton.getAttribute('aria-expanded') === 'true';
            console.log("Current state:", isExpanded ? "expanded" : "collapsed");
            
            if (isExpanded) {
                // Collapse content
                console.log("Collapsing content");
                content.style.display = 'none';
                toggleButton.setAttribute('aria-expanded', 'false');
                toggleButton.innerHTML = '<span class="toggle-icon">▼</span> Expand Reasoning';
            } else {
                // Expand content
                console.log("Expanding content");
                // 确保内容元素存在并且可见
                if (content) {
                content.style.display = 'block';
                content.style.opacity = '0';
                toggleButton.setAttribute('aria-expanded', 'true');
                toggleButton.innerHTML = '<span class="toggle-icon">▲</span> Collapse Reasoning';
                
                // Force reflow for animation
                void content.offsetWidth;
                content.style.opacity = '1';
                    
                    // 向下滚动以显示内容
                    setTimeout(() => {
                        content.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 100);
                } else {
                    console.error("Reasoning content element not found!");
                }
            }
            
            return false;
        });
    });
    
    console.log(`Initialized ${reasoningContainers.length} collapsible containers`);
}

// 初始化标准与推理高亮功能
function initializeCriterionHighlighting() {
    // Find all criteria items
    const criteriaItems = document.querySelectorAll('.criteria-section li:not(.highlight-initialized)');
    
    criteriaItems.forEach(item => {
        // Mark as initialized but don't add click handler
        item.classList.add('highlight-initialized');
        
        // Optional: Add a visual indication that these are not clickable
        item.style.cursor = 'default';
    });
    
    // Remove the automatic expansion of reasoning content
    console.log("Criterion highlighting initialized");
}

// 查找并高亮匹配的推理部分
function findAndHighlightMatchingReasoning(criterionText) {
    // 确保推理部分可见
    const reasoningContainer = document.querySelector('.reasoning-container');
    const toggleButton = reasoningContainer?.querySelector('.collapsible-toggle');
    const reasoningContent = reasoningContainer?.querySelector('.reasoning-content');
    
    if (toggleButton && reasoningContent && reasoningContent.style.display === 'none') {
        toggleButton.click(); // 展开推理部分
    }
    
    // 搜索可能匹配的部分
    const reasoningSections = document.querySelectorAll('.reasoning-section');
    let bestMatch = null;
    let bestScore = -1;
    
    // 遍历所有推理部分，找到最佳匹配
    reasoningSections.forEach(section => {
        const draftEl = section.querySelector('.reasoning-keyword.draft + br + *');
        if (draftEl) {
            const draftText = draftEl.textContent.trim().replace(/[""]/g, '');
            const score = calculateSimilarity(criterionText, draftText);
            
            if (score > bestScore) {
                bestScore = score;
                bestMatch = section;
            }
        }
    });
    
    // 如果找到匹配的推理部分，添加高亮并滚动到视图
    if (bestMatch && bestScore > 0.5) { // 设置一个相似度阈值
        bestMatch.classList.add('highlighted');
        bestMatch.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // 添加脉冲动画
        bestMatch.classList.add('pulse-highlight');
        setTimeout(() => {
            bestMatch.classList.remove('pulse-highlight');
        }, 2000);
    }
}

// 计算两个字符串的相似度 (简化的Levenshtein距离)
function calculateSimilarity(str1, str2) {
    // 将字符串转换为小写并移除特殊字符以进行比较
    const normalize = s => s.toLowerCase().replace(/[^\w\s]/g, '');
    const s1 = normalize(str1);
    const s2 = normalize(str2);
    
    // 如果一个字符串是另一个的子串，给予较高分数
    if (s1.includes(s2) || s2.includes(s1)) {
        const ratio = Math.min(s1.length, s2.length) / Math.max(s1.length, s2.length);
        return 0.7 + (ratio * 0.3); // 最低0.7分，最高1.0分
    }
    
    // 计算单词重叠
    const words1 = new Set(s1.split(/\s+/).filter(Boolean));
    const words2 = new Set(s2.split(/\s+/).filter(Boolean));
    
    if (words1.size === 0 || words2.size === 0) return 0;
    
    // 计算交集大小
    let intersection = 0;
    words1.forEach(word => {
        if (words2.has(word)) intersection++;
    });
    
    // Jaccard相似系数
    return intersection / (words1.size + words2.size - intersection);
}