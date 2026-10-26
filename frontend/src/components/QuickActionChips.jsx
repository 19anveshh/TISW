import React from 'react';

/**
 * QuickActionChips — Floating glass pills for teaching mode shortcuts.
 * Maps directly to the tutor's Adaptive Overrides:
 *   - Explain briefly (TL;DR)
 *   - Go deeper / under the hood
 *   - Teach me through questions (Socratic)
 *   - I don't understand (Reset analogy)
 */

const CHIPS = [
  {
    id: 'brief',
    label: 'Explain briefly (TL;DR)',
    icon: '⚡',
    message: 'Explain briefly — give me a TL;DR with 1-sentence Why, 1-sentence What, and 3-bullet How.',
    mode: 'brief',
  },
  {
    id: 'deep',
    label: 'Go deeper / Under the hood',
    icon: '🔬',
    message: 'Go deeper — skip toy examples, dive into internal execution models, mathematical formulations, edge-case failure modes, and production bottlenecks.',
    mode: 'deep',
  },
  {
    id: 'socratic',
    label: 'Teach me through questions',
    icon: '🧠',
    message: 'Teach me through Socratic questions — present a concrete scenario or a buggy approach, and ask me a targeted question to discover the principle myself.',
    mode: 'socratic',
  },
  {
    id: 'confused',
    label: "I don't understand (Reset analogy)",
    icon: '🔄',
    message: "I don't understand — step back, drop abstraction down one level, switch to a completely fresh everyday analogy, and trace step-by-step with tiny input (x = 2).",
    mode: 'confused',
  },
];

export default function QuickActionChips({ onChipClick, activeMode, disabled }) {
  return (
    <div className="quick-chips-container">
      {CHIPS.map((chip) => {
        const isActive = activeMode === chip.id;
        return (
          <button
            key={chip.id}
            type="button"
            disabled={disabled}
            onClick={() => onChipClick(chip)}
            className={`quick-chip ${isActive ? 'quick-chip--active' : ''}`}
            title={chip.label}
          >
            <span className="quick-chip-icon">{chip.icon}</span>
            <span className="quick-chip-label">{chip.label}</span>
          </button>
        );
      })}
    </div>
  );
}

export { CHIPS };
