// 增强型临床试验标准展示系统
document.addEventListener('DOMContentLoaded', () => {
    // 初始化折叠功能
    initializeCollapsibleElements();
    
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

// 初始化折叠功能
function initializeCollapsibleElements() {
    // 寻找所有reasoning容器并添加折叠功能
    const reasoningContainers = document.querySelectorAll('.reasoning-container:not(.collapsible-initialized)');
    
    reasoningContainers.forEach(container => {
        // 标记为已初始化
        container.classList.add('collapsible-initialized');
        
        // 创建折叠按钮
        const toggleButton = document.createElement('button');
        toggleButton.className = 'collapsible-toggle';
        toggleButton.innerHTML = '<span class="toggle-icon">▼</span> 展开推理过程';
        toggleButton.setAttribute('aria-expanded', 'false');
        
        // 获取内容元素
        const content = container.querySelector('.reasoning-content');
        content.style.display = 'none'; // 默认隐藏
        
        // 插入按钮
        const title = container.querySelector('.reasoning-title');
        if (title) {
            title.appendChild(toggleButton);
        } else {
            container.insertBefore(toggleButton, content);
        }

        // 添加点击事件
        toggleButton.addEventListener('click', () => {
            const isExpanded = toggleButton.getAttribute('aria-expanded') === 'true';
            
            // 切换显示状态
            content.style.display = isExpanded ? 'none' : 'block';
            toggleButton.setAttribute('aria-expanded', !isExpanded);
            toggleButton.innerHTML = isExpanded 
                ? '<span class="toggle-icon">▼</span> 展开推理过程' 
                : '<span class="toggle-icon">▲</span> 收起推理过程';
            
            // 添加淡入动画
            if (!isExpanded) {
                content.style.opacity = '0';
                setTimeout(() => {
                    content.style.opacity = '1';
                }, 10);
            }
        });
    });
        }

// 初始化标准与推理高亮功能
function initializeCriterionHighlighting() {
    // 寻找所有标准项
    const criteriaItems = document.querySelectorAll('.criteria-section li:not(.highlight-initialized)');
    
    criteriaItems.forEach(item => {
        // 标记为已初始化
        item.classList.add('highlight-initialized');

        // 添加点击事件
        item.addEventListener('click', () => {
            // 移除所有已有高亮
            document.querySelectorAll('.criteria-section li.highlighted').forEach(el => {
                el.classList.remove('highlighted');
            });
            document.querySelectorAll('.reasoning-section.highlighted').forEach(el => {
                el.classList.remove('highlighted');
            });
            
            // 添加高亮到当前标准
            item.classList.add('highlighted');
            
            // 提取标准文本
            const criterionText = item.textContent.trim();
            
            // 查找匹配的推理部分
            findAndHighlightMatchingReasoning(criterionText);
        });
    });
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