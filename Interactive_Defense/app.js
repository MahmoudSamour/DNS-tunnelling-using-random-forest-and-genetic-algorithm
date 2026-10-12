document.addEventListener('DOMContentLoaded', () => {
    const nodes = document.querySelectorAll('.node');
    const universe = document.getElementById('universe');
    const canvas = document.getElementById('canvas');
    const slideCounter = document.getElementById('slide-counter');
    
    const btnNext = document.getElementById('btn-next');
    const btnPrev = document.getElementById('btn-prev');
    const btnMap = document.getElementById('btn-map');
    const btnNotes = document.getElementById('btn-notes');

    let currentActiveIndex = -1; // -1 means Map View
    let isTransitioning = false;
    let showNotes = false;
    
    // Pan and Zoom State for Map View
    let isDragging = false;
    let startX = 0, startY = 0;
    let currentPanX = 0;
    let currentPanY = -50;
    
    // Layout Config: Dynamic Circular Layout
    const RADIUS = Math.max(7000, nodes.length * 380);
    let currentZoom = Math.min(0.06, 500 / RADIUS);
    
    const svg = document.getElementById('network-lines');
    let pathData = '';

    // Position nodes in a constellation ring
    nodes.forEach((node, index) => {
        const angle = (index / nodes.length) * Math.PI * 2 - (Math.PI / 2);
        
        const x = RADIUS * Math.cos(angle);
        const y = RADIUS * Math.sin(angle);
        const z = (Math.random() * 300) - 150;

        node.dataset.x = x;
        node.dataset.y = y;
        node.dataset.z = z;

        node.style.transform = `translate3d(${x}px, ${y}px, ${z}px)`;
        
        if (index === 0) {
            pathData += `M ${x} ${y} `;
        } else {
            pathData += `L ${x} ${y} `;
        }

        node.addEventListener('click', (e) => {
            if (isDragging) return;
            if (currentActiveIndex !== index && !isTransitioning) {
                goToSlide(index);
            }
        });
    });

    if (nodes.length > 0) {
        pathData += 'Z'; // Close ring
    }

    // Render the SVG constellation path
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.setAttribute('stroke', 'rgba(0, 240, 255, 0.4)');
    path.setAttribute('stroke-width', '24');
    path.setAttribute('fill', 'none');
    path.style.filter = 'drop-shadow(0 0 30px rgba(0, 240, 255, 0.8))';
    path.style.strokeLinecap = 'round';
    path.style.strokeLinejoin = 'round';
    svg.appendChild(path);

    // Pan and Zoom dragging
    document.addEventListener('mousedown', (e) => {
        if (currentActiveIndex !== -1 || isTransitioning) return;
        if (e.target.closest('#ui-overlay') || e.target.tagName.toLowerCase() === 'button' || e.target.tagName.toLowerCase() === 'a') return;

        isDragging = true;
        startX = e.clientX - currentPanX;
        startY = e.clientY - currentPanY;
        
        universe.style.transition = 'none';
        document.body.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || currentActiveIndex !== -1) return;
        
        currentPanX = e.clientX - startX;
        currentPanY = e.clientY - startY;
        
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(12deg)`;
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            universe.style.transition = 'transform 0.7s cubic-bezier(0.65, 0, 0.35, 1)';
            document.body.style.cursor = 'default';
        }
    });

    document.addEventListener('wheel', (e) => {
        if (currentActiveIndex !== -1 || isTransitioning) return;
        e.preventDefault();
        
        const zoomSensitivity = 0.0008;
        currentZoom -= e.deltaY * zoomSensitivity;
        currentZoom = Math.max(0.015, Math.min(currentZoom, 0.5)); 
        
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(12deg)`;
    }, { passive: false });

    function goToSlide(index) {
        if (index < 0 || index >= nodes.length || isTransitioning) return;
        isTransitioning = true;
        
        nodes.forEach(n => n.classList.remove('active'));
        const activeNode = nodes[index];
        currentActiveIndex = index;
        
        document.body.classList.add('slide-active');
        
        const targetX = -parseFloat(activeNode.dataset.x);
        const targetY = -parseFloat(activeNode.dataset.y);
        const targetZ = -parseFloat(activeNode.dataset.z);

        // Prezi Zoom out slightly, pan, then zoom in
        universe.style.transition = 'transform 0.6s cubic-bezier(0.65, 0, 0.35, 1)';
        universe.style.transform = `translate(0px, 0px) scale(0.2) rotateX(5deg)`;
        
        setTimeout(() => {
            canvas.style.transform = `translate3d(${targetX}px, ${targetY}px, ${targetZ}px)`;
            
            setTimeout(() => {
                universe.style.transform = `translate(0px, 0px) scale(1.0) rotateX(0deg)`;
                activeNode.classList.add('active');
                updateUI();
                
                setTimeout(() => {
                    isTransitioning = false;
                }, 700);
            }, 600);
        }, 500);
    }

    function goToMap() {
        if (isTransitioning) return;
        isTransitioning = true;
        
        currentActiveIndex = -1;
        nodes.forEach(n => n.classList.remove('active'));
        document.body.classList.remove('slide-active');
        
        canvas.style.transform = `translate3d(0px, 0px, 0px)`;
        
        currentPanX = 0;
        currentPanY = -50;
        currentZoom = Math.min(0.06, 500 / RADIUS);
        
        universe.style.transition = 'transform 0.8s cubic-bezier(0.65, 0, 0.35, 1)';
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(12deg)`;
        
        updateUI();
        setTimeout(() => { isTransitioning = false; }, 900);
    }

    function updateUI() {
        if (currentActiveIndex === -1) {
            slideCounter.textContent = 'Global Map View';
            btnPrev.style.opacity = '0.5';
            btnNext.style.opacity = '1';
        } else {
            const currentTitle = nodes[currentActiveIndex].dataset.title || '';
            const shortTitle = currentTitle.length > 25 ? currentTitle.substring(0, 25) + '...' : currentTitle;
            slideCounter.textContent = `Slide ${currentActiveIndex + 1} of ${nodes.length}`;
            btnPrev.style.opacity = currentActiveIndex > 0 ? '1' : '0.5';
            btnNext.style.opacity = currentActiveIndex < nodes.length - 1 ? '1' : '0.5';
        }
    }

    function toggleNotes() {
        showNotes = !showNotes;
        if (showNotes) {
            document.body.classList.add('show-notes');
            btnNotes.style.background = '#f59e0b';
            btnNotes.style.color = '#000000';
        } else {
            document.body.classList.remove('show-notes');
            btnNotes.style.background = '';
            btnNotes.style.color = '';
        }
    }

    btnNext.addEventListener('click', () => {
        if (currentActiveIndex < nodes.length - 1) goToSlide(currentActiveIndex + 1);
    });

    btnPrev.addEventListener('click', () => {
        if (currentActiveIndex > 0) goToSlide(currentActiveIndex - 1);
    });

    btnMap.addEventListener('click', goToMap);
    btnNotes.addEventListener('click', toggleNotes);

    document.addEventListener('keydown', (e) => {
        if (isTransitioning) return;
        if (e.key === 'ArrowRight' || e.key === ' ') {
            if (currentActiveIndex < nodes.length - 1) goToSlide(currentActiveIndex + 1);
        } else if (e.key === 'ArrowLeft') {
            if (currentActiveIndex > 0) goToSlide(currentActiveIndex - 1);
        } else if (e.key === 'Escape') {
            goToMap();
        } else if (e.key === 'n' || e.key === 'N') {
            toggleNotes();
        }
    });

    setTimeout(goToMap, 100);
});
