import React, { useEffect, useRef, useState } from 'react';

const TOTAL_FRAMES = 400;
const FRAME_RATE = 24; // fps

export default function EarthBackground({
  overlayOpacity = 0.38,
  glowOpacity = 0.55
}) {
  const canvasRef = useRef(null);
  const [loadedPercent, setLoadedPercent] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });

    let animationFrameId;
    let currentFrameIndex = 0;
    let lastTimestamp = 0;
    const frameInterval = 1000 / FRAME_RATE;

    // Cache array for preloaded Image objects
    const images = new Array(TOTAL_FRAMES);
    let loadedCount = 0;

    // Helper to format frame path: frame_0001.jpg -> frame_0400.jpg
    const getFrameUrl = (index) => {
      const numStr = String(index + 1).padStart(4, '0');
      return `/earth-frames/frame_${numStr}.jpg`;
    };

    // Preload high-priority initial chunk first, then load the rest asynchronously
    const INITIAL_CHUNK = 40;

    const loadSingleImage = (index) => {
      if (images[index]) return;
      const img = new Image();
      img.src = getFrameUrl(index);
      img.onload = () => {
        images[index] = img;
        loadedCount++;
        if (loadedCount % 10 === 0 || loadedCount === TOTAL_FRAMES) {
          setLoadedPercent(Math.round((loadedCount / TOTAL_FRAMES) * 100));
        }
      };
    };

    // Load first batch immediately
    for (let i = 0; i < Math.min(INITIAL_CHUNK, TOTAL_FRAMES); i++) {
      loadSingleImage(i);
    }

    // Schedule remaining frames in background so initial render is instantaneous
    const idleId = requestIdleCallback
      ? requestIdleCallback(() => {
          for (let i = INITIAL_CHUNK; i < TOTAL_FRAMES; i++) {
            loadSingleImage(i);
          }
        })
      : setTimeout(() => {
          for (let i = INITIAL_CHUNK; i < TOTAL_FRAMES; i++) {
            loadSingleImage(i);
          }
        }, 300);

    // Resize canvas to match display size & DPR
    const handleResize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    // Render loop
    const render = (timestamp) => {
      if (!lastTimestamp) lastTimestamp = timestamp;
      const elapsed = timestamp - lastTimestamp;

      if (elapsed >= frameInterval) {
        lastTimestamp = timestamp - (elapsed % frameInterval);

        // Find best available frame (current, or nearest previously loaded)
        let frameToDraw = images[currentFrameIndex];
        if (!frameToDraw) {
          // Look backwards for latest loaded frame to prevent any flicker
          for (let k = currentFrameIndex - 1; k >= 0; k--) {
            if (images[k]) {
              frameToDraw = images[k];
              break;
            }
          }
        }

        if (frameToDraw && frameToDraw.complete && frameToDraw.naturalWidth) {
          const cw = canvas.width;
          const ch = canvas.height;
          const iw = frameToDraw.naturalWidth;
          const ih = frameToDraw.naturalHeight;

          // Object-fit: cover scaling
          const scale = Math.max(cw / iw, ch / ih);
          const dw = iw * scale;
          const dh = ih * scale;
          const dx = (cw - dw) / 2;
          const dy = (ch - dh) / 2;

          ctx.drawImage(frameToDraw, dx, dy, dw, dh);
        }

        // Advance to next frame
        currentFrameIndex = (currentFrameIndex + 1) % TOTAL_FRAMES;
      }

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      if (requestIdleCallback && idleId) cancelIdleCallback(idleId);
      else clearTimeout(idleId);
    };
  }, []);

  return (
    <div className="earth-bg-root">
      {/* Canvas rendering sequential Earth frames with violet planetary color-grading */}
      <canvas
        ref={canvasRef}
        className="earth-bg-canvas"
        style={{
          filter: 'hue-rotate(242deg) saturate(135%) brightness(0.68) contrast(112%)',
        }}
      />

      {/* Atmospheric space vignette overlay for high contrast text readability */}
      <div
        className="earth-bg-overlay"
        style={{
          background: `radial-gradient(circle at 50% 35%, rgba(16, 12, 26, ${overlayOpacity * 0.4}) 0%, rgba(8, 6, 14, ${overlayOpacity * 1.6}) 100%)`,
        }}
      />

      {/* Glowing cosmic violet / lavender atmospheric nebula glow in top-left matching screenshot */}
      <div
        className="earth-bg-glow"
        style={{
          background:
            'radial-gradient(circle at 20% 20%, rgba(168, 85, 247, 0.28) 0%, rgba(124, 58, 237, 0.16) 35%, rgba(9, 7, 16, 0) 70%)',
          filter: 'blur(50px)',
          opacity: glowOpacity
        }}
      />

      {/* Ambient specular corner light accents */}
      <div
        className="earth-bg-accent-tr"
        style={{
          background: 'radial-gradient(circle, rgba(147, 51, 234, 0.2) 0%, transparent 65%)',
          filter: 'blur(55px)'
        }}
      />
      <div
        className="earth-bg-accent-bl"
        style={{
          background: 'radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, transparent 65%)',
          filter: 'blur(60px)'
        }}
      />
    </div>
  );
}
