import React from 'react';

/**
 * Toast — Non-blocking notification component.
 * Slides up from the bottom with glassmorphic styling.
 */
export default function Toast({ message, type = 'info' }) {
  if (!message) return null;

  const typeStyles = {
    info: 'bg-white/80 text-slate-800 border-white/90',
    error: 'bg-red-50/90 text-red-800 border-red-200/80',
    success: 'bg-emerald-50/90 text-emerald-800 border-emerald-200/80',
    warning: 'bg-amber-50/90 text-amber-800 border-amber-200/80',
  };

  const iconMap = {
    info: '💡',
    error: '⚠️',
    success: '✅',
    warning: '⏳',
  };

  return (
    <div
      className="toast-container"
      style={{
        position: 'fixed',
        bottom: '24px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 9999,
        animation: 'toastSlideUp 0.3s ease-out forwards',
      }}
    >
      <div
        className={`toast-pill ${typeStyles[type] || typeStyles.info}`}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '10px 20px',
          borderRadius: '9999px',
          fontSize: '13px',
          fontWeight: 600,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.15)',
          whiteSpace: 'nowrap',
        }}
      >
        <span>{iconMap[type] || '💡'}</span>
        <span>{message}</span>
      </div>
    </div>
  );
}
