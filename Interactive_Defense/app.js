document.addEventListener('DOMContentLoaded', () => {
    const nodes = document.querySelectorAll('.node');
    const universe = document.getElementById('universe'); // Handles Zoom / Depth / Pan
    const canvas = document.getElementById('canvas');     // Handles X / Y panning for slide focus
    const slideCounter = document.getElementById('slide-counter');
    
    const btnNext = document.getElementById('btn-next');
    const btnPrev = document.getElementById('btn-prev');
    const btnMap = document.getElementById('btn-map');

    let currentActiveIndex = -1; // -1 means Map View
    let isTransitioning = false;
    
    // Pan and Zoom State for Map View
    let isDragging = false;
    let startX = 0, startY = 0;
    let currentPanX = 0;
    let currentPanY = -100;
    let currentZoom = 0.07;
    
    // Layout Config: Circular Layout
    const RADIUS = 6000; 
    
    const svg = document.getElementById('network-lines');
    let pathData = '';

    // Position nodes in a massive ring
    nodes.forEach((node, index) => {
        // Calculate angle (starts at top, -90 degrees, goes clockwise)
        const angle = (index / nodes.length) * Math.PI * 2 - (Math.PI / 2);
        
        const x = RADIUS * Math.cos(angle);
        const y = RADIUS * Math.sin(angle);
        const z = (Math.random() * 400) - 200; // Slight depth variation

        node.dataset.x = x;
        node.dataset.y = y;
        node.dataset.z = z;

        node.style.transform = `translate3d(${x}px, ${y}px, ${z}px)`;
        
        // SVG path
        if (index === 0) {
            pathData += `M ${x} ${y} `;
        } else {
            pathData += `L ${x} ${y} `;
        }

        // Click event to zoom into node
        node.addEventListener('click', (e) => {
            // Prevent click if we were dragging
            if (isDragging) return;
            if (currentActiveIndex !== index && !isTransitioning) {
                goToSlide(index);
            }
        });
    });

    // Render the SVG
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.setAttribute('stroke', 'rgba(0, 240, 255, 0.8)');
    path.setAttribute('stroke-width', '20');
    path.setAttribute('fill', 'none');
    path.style.filter = 'drop-shadow(0 0 25px rgba(0, 240, 255, 1))';
    path.style.strokeLinecap = 'round';
    path.style.strokeLinejoin = 'round';
    svg.appendChild(path);

    // --- INTERACTIVE PAN AND ZOOM ---
    document.addEventListener('mousedown', (e) => {
        // Only allow drag in Map View
        if (currentActiveIndex !== -1 || isTransitioning) return;
        
        // Don't drag if clicking a button
        if (e.target.tagName.toLowerCase() === 'button') return;

        isDragging = true;
        startX = e.clientX - currentPanX;
        startY = e.clientY - currentPanY;
        
        // Disable CSS transition for smooth frame-by-frame dragging
        universe.style.transition = 'none';
        document.body.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || currentActiveIndex !== -1) return;
        
        currentPanX = e.clientX - startX;
        currentPanY = e.clientY - startY;
        
        // Apply transform immediately
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(15deg)`;
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            // Restore CSS transition
            universe.style.transition = 'transform 0.7s cubic-bezier(0.65, 0, 0.35, 1)';
            document.body.style.cursor = 'default';
        }
    });

    document.addEventListener('wheel', (e) => {
        if (currentActiveIndex !== -1 || isTransitioning) return;
        e.preventDefault();
        
        const zoomSensitivity = 0.001;
        currentZoom -= e.deltaY * zoomSensitivity;
        
        // Clamp zoom between 0.03 (very far) and 1.0 (close)
        currentZoom = Math.max(0.03, Math.min(currentZoom, 1)); 
        
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(15deg)`;
    }, { passive: false });
    // --------------------------------

    function goToSlide(index) {
        if (index < 0 || index >= nodes.length || isTransitioning) return;
        isTransitioning = true;
        
        nodes.forEach(n => n.classList.remove('active'));
        const activeNode = nodes[index];
        currentActiveIndex = index;
        
        // Hide network lines
        document.body.classList.add('slide-active');
        
        const targetX = -parseFloat(activeNode.dataset.x);
        const targetY = -parseFloat(activeNode.dataset.y);
        const targetZ = -parseFloat(activeNode.dataset.z);

        // Prezi-style Transition:
        // 1. Zoom Out slightly to give a sense of movement
        universe.style.transition = 'transform 0.7s cubic-bezier(0.65, 0, 0.35, 1)';
        universe.style.transform = `translate(0px, 0px) scale(0.25) rotateX(5deg)`;
        
        setTimeout(() => {
            // 2. Pan canvas to the new slide
            canvas.style.transform = `translate3d(${targetX}px, ${targetY}px, ${targetZ}px)`;
            
            setTimeout(() => {
                // 3. Zoom In to the new slide
                universe.style.transform = `translate(0px, 0px) scale(1.0) rotateX(0deg)`;
                activeNode.classList.add('active');
                updateUI();
                
                setTimeout(() => {
                    isTransitioning = false;
                }, 800);
            }, 800);
        }, 600);
    }

    function goToMap() {
        if (isTransitioning) return;
        isTransitioning = true;
        
        currentActiveIndex = -1;
        nodes.forEach(n => n.classList.remove('active'));
        
        // Show network lines
        document.body.classList.remove('slide-active');
        
        // Center the canvas
        canvas.style.transform = `translate3d(0px, 0px, 0px)`;
        
        // Reset pan/zoom to ensure the entire 6000px radius circle is visible
        currentPanX = 0;
        currentPanY = -100;
        currentZoom = 0.07;
        
        universe.style.transition = 'transform 0.7s cubic-bezier(0.65, 0, 0.35, 1)';
        universe.style.transform = `translate(${currentPanX}px, ${currentPanY}px) scale(${currentZoom}) rotateX(15deg)`;
        
        updateUI();
        setTimeout(() => { isTransitioning = false; }, 1000);
    }

    function updateUI() {
        if (currentActiveIndex === -1) {
            slideCounter.textContent = 'Map View';
            btnPrev.style.opacity = '0.5';
            btnNext.style.opacity = '1';
        } else {
            slideCounter.textContent = `Slide ${currentActiveIndex + 1} of ${nodes.length}`;
            btnPrev.style.opacity = currentActiveIndex > 0 ? '1' : '0.5';
            btnNext.style.opacity = currentActiveIndex < nodes.length - 1 ? '1' : '0.5';
        }
    }

    btnNext.addEventListener('click', () => {
        if (currentActiveIndex < nodes.length - 1) goToSlide(currentActiveIndex + 1);
    });

    btnPrev.addEventListener('click', () => {
        if (currentActiveIndex > 0) goToSlide(currentActiveIndex - 1);
    });

    btnMap.addEventListener('click', goToMap);

    document.addEventListener('keydown', (e) => {
        if (isTransitioning) return;
        if (e.key === 'ArrowRight' || e.key === ' ') {
            if (currentActiveIndex < nodes.length - 1) goToSlide(currentActiveIndex + 1);
        } else if (e.key === 'ArrowLeft') {
            if (currentActiveIndex > 0) goToSlide(currentActiveIndex - 1);
        } else if (e.key === 'Escape') {
            goToMap();
        }
    });

    // Initialize Map View
    setTimeout(goToMap, 100);
});
