document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide-panel');
    const totalSlides = slides.length;
    
    const currNumDisplay = document.getElementById('curr-num');
    const progressBar = document.getElementById('progress-bar');
    const slidePicker = document.getElementById('slide-picker');
    
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    
    const btnGrid = document.getElementById('btn-grid');
    const gridModal = document.getElementById('grid-modal');
    const btnCloseGrid = document.getElementById('btn-close-grid');
    const modalBackdrop = gridModal.querySelector('.modal-backdrop');
    
    const btnNotes = document.getElementById('btn-notes');
    const notesDrawer = document.getElementById('notes-drawer');
    const btnCloseNotes = document.getElementById('btn-close-notes');
    const notesBody = document.getElementById('notes-body');
    
    const thumbnails = document.querySelectorAll('.grid-thumb');
    
    let currentSlide = 1;
    let isNotesOpen = false;

    function goToSlide(n, updateHash = true) {
        if (n < 1) n = 1;
        if (n > totalSlides) n = totalSlides;
        currentSlide = n;

        // Active slide switch
        slides.forEach((s, idx) => {
            const slideNum = idx + 1;
            if (slideNum === currentSlide) {
                s.classList.add('active');
            } else {
                s.classList.remove('active');
            }
        });

        // Top and bottom controls sync
        currNumDisplay.textContent = String(currentSlide).padStart(2, '0');
        slidePicker.value = currentSlide;
        progressBar.style.width = `${(currentSlide / totalSlides) * 100}%`;

        btnPrev.disabled = currentSlide === 1;
        btnPrev.style.opacity = currentSlide === 1 ? '0.4' : '1';
        btnNext.disabled = currentSlide === totalSlides;
        btnNext.style.opacity = currentSlide === totalSlides ? '0.4' : '1';

        // Thumbnails active state
        thumbnails.forEach((t) => {
            const target = parseInt(t.dataset.slideTarget, 10);
            if (target === currentSlide) {
                t.classList.add('active-thumb');
            } else {
                t.classList.remove('active-thumb');
            }
        });

        // Update speaker notes content
        const activeSlide = slides[currentSlide - 1];
        const notes = activeSlide ? activeSlide.dataset.notes : '';
        if (notes && notes.trim()) {
            notesBody.innerHTML = notes.replace(/\n/g, '<br>');
            btnNotes.style.borderColor = 'rgba(245, 158, 11, 0.6)';
            btnNotes.style.color = '#f59e0b';
        } else {
            notesBody.innerHTML = '<em>No specific examiner directives or speaking notes for this slide.</em>';
            btnNotes.style.borderColor = '';
            btnNotes.style.color = '';
        }

        if (updateHash) {
            window.location.hash = `slide-${currentSlide}`;
        }
    }

    // Next / Prev buttons
    btnNext.addEventListener('click', () => goToSlide(currentSlide + 1));
    btnPrev.addEventListener('click', () => goToSlide(currentSlide - 1));

    // Dropdown picker
    slidePicker.addEventListener('change', (e) => {
        goToSlide(parseInt(e.target.value, 10));
    });

    // Grid Modal
    function openGrid() {
        gridModal.classList.remove('modal-hidden');
        const activeThumb = gridModal.querySelector(`.grid-thumb[data-slide-target="${currentSlide}"]`);
        if (activeThumb) {
            activeThumb.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function closeGrid() {
        gridModal.classList.add('modal-hidden');
    }

    btnGrid.addEventListener('click', openGrid);
    btnCloseGrid.addEventListener('click', closeGrid);
    modalBackdrop.addEventListener('click', closeGrid);

    thumbnails.forEach((thumb) => {
        thumb.addEventListener('click', () => {
            const target = parseInt(thumb.dataset.slideTarget, 10);
            goToSlide(target);
            closeGrid();
        });
    });

    // Notes Drawer
    function toggleNotes() {
        isNotesOpen = !isNotesOpen;
        if (isNotesOpen) {
            notesDrawer.classList.remove('drawer-hidden');
            btnNotes.style.background = '#f59e0b';
            btnNotes.style.color = '#000000';
        } else {
            notesDrawer.classList.add('drawer-hidden');
            btnNotes.style.background = '';
            btnNotes.style.color = '';
        }
    }

    function closeNotes() {
        isNotesOpen = false;
        notesDrawer.classList.add('drawer-hidden');
        btnNotes.style.background = '';
        btnNotes.style.color = '';
    }

    btnNotes.addEventListener('click', toggleNotes);
    btnCloseNotes.addEventListener('click', closeNotes);

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName.toLowerCase() === 'input' || e.target.tagName.toLowerCase() === 'textarea' || e.target.tagName.toLowerCase() === 'select') {
            return;
        }

        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
            e.preventDefault();
            goToSlide(currentSlide + 1);
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            e.preventDefault();
            goToSlide(currentSlide - 1);
        } else if (e.key === 'Home') {
            e.preventDefault();
            goToSlide(1);
        } else if (e.key === 'End') {
            e.preventDefault();
            goToSlide(totalSlides);
        } else if (e.key === 'g' || e.key === 'G') {
            e.preventDefault();
            if (gridModal.classList.contains('modal-hidden')) openGrid();
            else closeGrid();
        } else if (e.key === 'n' || e.key === 'N') {
            e.preventDefault();
            toggleNotes();
        } else if (e.key === 'Escape') {
            closeGrid();
            closeNotes();
        }
    });

    // Touch / Swipe
    let touchStartX = 0;
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        const touchEndX = e.changedTouches[0].screenX;
        const diff = touchEndX - touchStartX;
        if (Math.abs(diff) > 50) {
            if (diff < 0) goToSlide(currentSlide + 1);
            else goToSlide(currentSlide - 1);
        }
    }, { passive: true });

    // Initial slide from hash or default to 1
    const hash = window.location.hash;
    const match = hash.match(/slide-(\d+)/);
    const initialSlide = match ? parseInt(match[1], 10) : 1;
    goToSlide(initialSlide, false);
});
