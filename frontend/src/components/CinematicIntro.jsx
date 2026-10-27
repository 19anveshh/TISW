import React, { useEffect, useRef, useState } from 'react';
import './CinematicIntro.css';

const CORE_WORDS = [
  { word: 'Teaching', key: 'teaching', left: '11%', top: '29%', size: 'clamp(2.6rem, 5.8vw, 5rem)', opacity: 0.92, delay: '80ms' },
  { word: 'in', key: 'in', left: '46%', top: '17%', size: 'clamp(1.45rem, 2.8vw, 2.4rem)', opacity: 0.62, delay: '220ms' },
  { word: 'a', key: 'article', left: '78%', top: '30%', size: 'clamp(1.1rem, 2vw, 1.7rem)', opacity: 0.38, delay: '360ms' },
  { word: 'Structured', key: 'structured', left: '10%', top: '59%', size: 'clamp(2.3rem, 4.8vw, 4.1rem)', opacity: 0.8, delay: '180ms' },
  { word: 'Way', key: 'way', left: '71%', top: '60%', size: 'clamp(2.4rem, 5.1vw, 4.4rem)', opacity: 0.86, delay: '300ms' },
];

const ATMOSPHERE_WORDS = [
  { word: 'Knowledge', left: '6%', top: '16%', size: 'clamp(0.9rem, 1.7vw, 1.35rem)', opacity: 0.34, delay: '40ms' },
  { word: 'Curiosity', left: '75%', top: '17%', size: 'clamp(1rem, 1.9vw, 1.5rem)', opacity: 0.42, delay: '260ms' },
  { word: 'Thinking', left: '39%', top: '31%', size: 'clamp(0.8rem, 1.45vw, 1.15rem)', opacity: 0.28, delay: '480ms' },
  { word: 'Discovery', left: '72%', top: '43%', size: 'clamp(0.95rem, 1.8vw, 1.4rem)', opacity: 0.34, delay: '140ms' },
  { word: 'Understanding', left: '3%', top: '47%', size: 'clamp(0.75rem, 1.35vw, 1.05rem)', opacity: 0.24, delay: '600ms' },
  { word: 'Clarity', left: '44%', top: '74%', size: 'clamp(0.85rem, 1.55vw, 1.2rem)', opacity: 0.33, delay: '340ms' },
  { word: 'Ideas', left: '83%', top: '77%', size: 'clamp(0.95rem, 1.8vw, 1.4rem)', opacity: 0.36, delay: '520ms' },
  { word: 'Explore', left: '19%', top: '83%', size: 'clamp(0.8rem, 1.45vw, 1.15rem)', opacity: 0.28, delay: '720ms' },
  { word: 'Reason', left: '60%', top: '85%', size: 'clamp(0.9rem, 1.65vw, 1.3rem)', opacity: 0.3, delay: '400ms' },
  { word: 'Insight', left: '86%', top: '54%', size: 'clamp(0.75rem, 1.35vw, 1.05rem)', opacity: 0.26, delay: '660ms' },
  { word: 'Learning', left: '31%', top: '47%', size: 'clamp(0.8rem, 1.5vw, 1.2rem)', opacity: 0.24, delay: '560ms' },
];

const LETTER_KEYS = ['teaching', 'in', 'structured', 'way'];
const INTRO_DURATION_MS = 6500;

function KineticWord({ item, core = false, registerLetterRef, registerWordRef, convergeTransform, letterTransforms = [] }) {
  const firstLetter = item.word[0];
  const letterIndex = LETTER_KEYS.indexOf(item.key);
  return (
    <span
      ref={core ? (node) => {
        if (node) registerWordRef(item.key, node);
      } : undefined}
      className={`cinematic-word ${core ? 'cinematic-word--core' : 'cinematic-word--atmosphere'} cinematic-word--${item.key || item.word.toLowerCase()}`}
      style={{
        left: item.left,
        top: item.top,
        fontSize: item.size,
        '--converge-transform': convergeTransform || 'translate3d(0, 0, 0)',
        '--word-opacity': item.opacity,
        '--word-delay': item.delay,
      }}
    >
      {core ? (
        <>
          <span
            ref={(node) => {
              if (node) registerLetterRef(item.key, node);
            }}
            className="cinematic-initial"
            style={{ transform: letterTransforms[letterIndex] || undefined }}
          >
            {firstLetter}
          </span>
          <span className="cinematic-rest">{item.word.slice(1)}</span>
        </>
      ) : item.word}
    </span>
  );
}

function getTargetLetterPositions(targetElement, sourceFontSizes) {
  if (!targetElement) return null;

  const rect = targetElement.getBoundingClientRect();
  const styles = window.getComputedStyle(targetElement);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return null;

  context.font = `${styles.fontWeight} ${styles.fontSize} ${styles.fontFamily}`;
  const letterSpacing = Number.parseFloat(styles.letterSpacing) || 0;
  const widths = LETTER_KEYS.map((key) => context.measureText(key[0].toUpperCase()).width);
  const totalWidth = widths.reduce((sum, width) => sum + width, 0) + letterSpacing * (widths.length - 1);
  let cursor = rect.left + (rect.width - totalWidth) / 2;

  return LETTER_KEYS.map((_key, index) => {
    const width = widths[index];
    const position = {
      x: cursor + width / 2,
      y: rect.top + rect.height / 2,
      scale: Number.parseFloat(styles.fontSize) / sourceFontSizes[index],
    };
    cursor += width + letterSpacing;
    return position;
  });
}

function getLetterWidths(element, fontSize) {
  const styles = window.getComputedStyle(element);
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return null;

  context.font = `${styles.fontWeight} ${fontSize}px ${styles.fontFamily}`;
  return LETTER_KEYS.map((key) => context.measureText(key[0].toUpperCase()).width);
}

export default function CinematicIntro({ onReveal, onComplete }) {
  const [scene, setScene] = useState('void');
  const [complete, setComplete] = useState(false);
  const [letterTransforms, setLetterTransforms] = useState([]);
  const [convergeTransforms, setConvergeTransforms] = useState({});
  const letterRefs = useRef({});
  const wordRefs = useRef({});
  const formationRectsRef = useRef([]);
  const registerLetterRef = (key, node) => {
    letterRefs.current[key] = node;
  };
  const registerWordRef = (key, node) => {
    wordRefs.current[key] = node;
  };

  useEffect(() => {
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) {
      onReveal();
      onComplete();
      setComplete(true);
      return undefined;
    }

    const timers = [];
    const frames = [];
    let travelActive = false;
    const scheduleFrame = (callback) => {
      const frame = window.requestAnimationFrame(callback);
      frames.push(frame);
    };
    const updateTravelTransform = () => {
      if (!travelActive) return;

      const target = document.querySelector('.brand-logo-text');
      const sourceFontSizes = LETTER_KEYS.map((key) => (
        Number.parseFloat(window.getComputedStyle(letterRefs.current[key]).fontSize)
      ));
      const targetPositions = getTargetLetterPositions(target, sourceFontSizes);
      const sourceRects = formationRectsRef.current;
      if (!targetPositions || sourceRects.some((rect) => !rect)) return;

      setLetterTransforms(targetPositions.map((position, index) => {
        const rect = sourceRects[index];
        const targetWidth = rect.width * position.scale;
        const targetHeight = rect.height * position.scale;
        const targetLeft = position.x - targetWidth / 2;
        const targetTop = position.y - targetHeight / 2;
        return `translate3d(${targetLeft - rect.left}px, ${targetTop - rect.top}px, 0) scale(${position.scale})`;
      }));
    };
    window.addEventListener('resize', updateTravelTransform);

    timers.push(window.setTimeout(() => setScene('words'), 520));
    timers.push(window.setTimeout(() => {
      setScene('converge');
      scheduleFrame(() => {
        const wordRects = LETTER_KEYS.map((key) => wordRefs.current[key]?.getBoundingClientRect());
        if (wordRects.some((rect) => !rect)) return;

        const layout = [
          { x: 0.34, y: 0.39 },
          { x: 0.66, y: 0.39 },
          { x: 0.34, y: 0.61 },
          { x: 0.66, y: 0.61 },
        ];
        const transforms = {};

        LETTER_KEYS.forEach((key, index) => {
          const rect = wordRects[index];
          const targetLeft = window.innerWidth * layout[index].x - rect.width / 2;
          const targetTop = window.innerHeight * layout[index].y - rect.height / 2;
          transforms[key] = `translate3d(${targetLeft - rect.left}px, ${targetTop - rect.top}px, 0)`;
        });

        setConvergeTransforms(transforms);
      });
    }, 1800));
    timers.push(window.setTimeout(() => setScene('formation'), 3000));
    timers.push(window.setTimeout(() => {
      scheduleFrame(() => {
        scheduleFrame(() => {
          const measured = LETTER_KEYS.map((key) => letterRefs.current[key]?.getBoundingClientRect());
          const validRects = measured.every(Boolean) ? measured : null;
          if (!validRects) return;
          formationRectsRef.current = validRects;

          const rootFontSize = Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize) || 16;
          const fluidFormationSize = window.innerWidth <= 700
            ? window.innerWidth * 0.17
            : window.innerWidth * 0.1;
          const formationFontSize = window.innerWidth <= 700
            ? Math.min(Math.max(rootFontSize * 3.6, fluidFormationSize), rootFontSize * 6.5)
            : Math.min(Math.max(rootFontSize * 4.2, fluidFormationSize), rootFontSize * 8);
          const sourceFontSizes = LETTER_KEYS.map((key) => (
            Number.parseFloat(window.getComputedStyle(letterRefs.current[key]).fontSize)
          ));
          const widths = getLetterWidths(letterRefs.current.teaching, formationFontSize);
          if (!widths) return;
          const gap = Math.max(16, formationFontSize * 0.14);
          const totalWidth = widths.reduce((sum, width) => sum + width, 0) + gap * (widths.length - 1);
          let cursor = window.innerWidth / 2 - totalWidth / 2;
          const centerY = window.innerHeight * 0.46;
          const transforms = validRects.map((rect, index) => {
            const targetX = cursor + widths[index] / 2;
            cursor += widths[index] + gap;
            const scale = formationFontSize / sourceFontSizes[index];
            const targetTop = centerY - (rect.height * scale) / 2;
            return `translate3d(${targetX - widths[index] / 2 - rect.left}px, ${targetTop - rect.top}px, 0) scale(${scale})`;
          });
          setLetterTransforms(transforms);
        });
      });
    }, 3650));
    timers.push(window.setTimeout(() => {
      setScene('travel');
      travelActive = true;
      onReveal();
      scheduleFrame(updateTravelTransform);
    }, 4700));
    timers.push(window.setTimeout(() => {
      travelActive = false;
      window.removeEventListener('resize', updateTravelTransform);
      setComplete(true);
      onComplete();
    }, INTRO_DURATION_MS));

    return () => {
      timers.forEach((timer) => window.clearTimeout(timer));
      frames.forEach((frame) => window.cancelAnimationFrame(frame));
      window.removeEventListener('resize', updateTravelTransform);
    };
  }, [onComplete, onReveal]);

  if (complete) return null;

  return (
    <div className={`cinematic-intro cinematic-intro--${scene}`} aria-hidden="true">
      <div className="cinematic-intro__grain" />
      <div className="cinematic-intro__words">
        {CORE_WORDS.map((item) => (
          <KineticWord
            key={item.key}
            item={item}
            core={LETTER_KEYS.includes(item.key)}
            registerLetterRef={registerLetterRef}
            registerWordRef={registerWordRef}
            convergeTransform={convergeTransforms[item.key]}
            letterTransforms={letterTransforms}
          />
        ))}
        {ATMOSPHERE_WORDS.map((item) => (
          <KineticWord key={item.word} item={item} />
        ))}
      </div>
    </div>
  );
}
