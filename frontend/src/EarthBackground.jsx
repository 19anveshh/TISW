import React, { useEffect, useRef } from 'react';

export default function EarthBackground({
  overlayOpacity = 0.38,
  glowOpacity = 0.55
}) {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;

    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const saveData = navigator.connection?.saveData;
    if (reducedMotion || saveData) {
      video.pause();
      return undefined;
    }

    const resume = () => video.play().catch(() => {});
    resume();

    const handleVisibility = () => {
      if (document.hidden) {
        video.pause();
      } else {
        resume();
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);

  return (
    <div className="earth-bg-root">
      {/* Looping Earth footage; poster frame stands in until playback starts */}
      <video
        ref={videoRef}
        className="earth-bg-video"
        poster="/earth-poster.jpg"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        aria-hidden="true"
        tabIndex={-1}
        style={{
          filter: 'hue-rotate(242deg) saturate(135%) brightness(0.68) contrast(112%)',
        }}
      >
        <source src="/earth-loop.mp4" type="video/mp4" />
      </video>

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
