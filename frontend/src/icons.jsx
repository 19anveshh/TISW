import React from 'react';

const createIcon = (d, extra = null) => ({ className = "w-4 h-4", ...props }) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
    {...props}
  >
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
    {extra}
  </svg>
);

export const Sparkles = createIcon([
  "m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"
]);

export const Search = ({ className = "w-4 h-4", ...props }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} {...props}>
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

export const Bell = createIcon([
  "M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9",
  "M10.3 21a1.94 1.94 0 0 0 3.4 0"
]);

export const Paperclip = createIcon("m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48");

export const Globe = ({ className = "w-4 h-4", ...props }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} {...props}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    <path d="M2 12h20" />
  </svg>
);

export const RefreshCw = createIcon([
  "M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8",
  "M21 3v5h-5",
  "M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16",
  "M3 21v-5h5"
]);

export const Send = createIcon("m22 2-7 20-4-9-9-4Z", <path d="M22 2 11 13" />);

export const Zap = createIcon("M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z");

export const Lightbulb = createIcon([
  "M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5",
  "M9 18h6",
  "M10 22h4"
]);

export const Wand2 = createIcon([
  "m19 11-4-4",
  "m2 22 10-10",
  "M14 6l1-3 1 3 3 1-3 1-1 3-1-3-3-1z"
]);

export const Box = createIcon([
  "M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z",
  "m3.3 7 8.7 5 8.7-5",
  "M12 22V12"
]);

export const Sliders = createIcon([
  "M4 21v-7",
  "M4 10V3",
  "M12 21v-9",
  "M12 8V3",
  "M20 21v-5",
  "M20 12V3",
  "M1 14h6",
  "M9 8h6",
  "M17 16h6"
]);

export const Plus = createIcon(["M5 12h14", "M12 5v14"]);

export const Check = createIcon("M20 6 9 17l-5-5");

export const Terminal = createIcon([
  "m4 17 6-6-6-6",
  "M12 19h8"
]);

export const Copy = createIcon([
  "rect width='14' height='14' x='8' y='8' rx='2' ry='2'",
  "M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"
]);

export const Layers = createIcon([
  "m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z",
  "m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65",
  "m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"
]);

export const BookOpen = createIcon([
  "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z",
  "M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"
]);

export const Cpu = createIcon([
  "M12 2v2M12 20v2M2 12h2M20 12h2",
  "rect width='12' height='12' x='6' y='6' rx='2'"
]);

export const MessageSquare = createIcon("M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z");

export const ArrowRight = createIcon(["M5 12h14", "m12 5 7 7-7 7"]);

export const ChevronRight = createIcon("m9 18 6-6-6-6");

export const ShieldCheck = createIcon([
  "M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z",
  "m9 12 2 2 4-4"
]);

export const X = createIcon(["M18 6 6 18", "m6 6 12 12"]);

export const Volume2 = createIcon([
  "polygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'",
  "M15.54 8.46a5 5 0 0 1 0 7.07",
  "M19.07 4.93a10 10 0 0 1 0 14.14"
]);

export const HelpCircle = ({ className = "w-4 h-4", ...props }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className} {...props}>
    <circle cx="12" cy="12" r="10" />
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
    <path d="M12 17h.01" />
  </svg>
);

export const ChevronDown = createIcon("m6 9 6 6 6-6");
export const ArrowUpRight = createIcon(["M7 17L17 7", "M7 7h10v10"]);
export const Activity = createIcon(["M3 12h4l3-8 4 16 3-8h4"]);

export const FileText = createIcon([
  "M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z",
  "M14 2v6h6",
  "M16 13H8",
  "M16 17H8",
  "M10 9H8"
]);

export const Trash2 = createIcon([
  "M3 6h18",
  "M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6",
  "M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2",
  "M10 11v6",
  "M14 11v6"
]);

export const Database = createIcon([
  "M3 5a9 3 0 1 0 18 0a9 3 0 1 0-18 0",
  "M3 5v14a9 3 0 0 0 18 0V5",
  "M3 12a9 3 0 0 0 18 0"
]);

export const KeyRound = createIcon([
  "M2.586 17.414A2 2 0 0 0 2 18.828V21a1 1 0 0 0 1 1h3a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h1a1 1 0 0 0 1-1v-1a1 1 0 0 1 1-1h.172a2 2 0 0 0 1.414-.586l.814-.814a6.5 6.5 0 1 0-4-4z",
  "M16.5 7.5h.01"
]);

export const Menu = createIcon(["M4 6h16", "M4 12h16", "M4 18h16"]);

export const Upload = createIcon([
  "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",
  "m17 8-5-5-5 5",
  "M12 3v12"
]);

export const History = createIcon([
  "M3 3v5h5",
  "M3.05 13A9 9 0 1 0 6 5.3L3 8",
  "M12 7v5l4 2"
]);

export const ExternalLink = createIcon([
  "M15 3h6v6",
  "M10 14 21 3",
  "M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"
]);

export const GitHub = ({ className = "w-4 h-4", ...props }) => (
  <svg
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
    focusable="false"
    {...props}
  >
    <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.55v-2.1c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.75 2.69 1.25 3.35.96.1-.75.4-1.25.72-1.54-2.55-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.18-3.1-.12-.29-.51-1.46.11-3.03 0 0 .96-.31 3.15 1.18a10.9 10.9 0 0 1 2.87-.39c.97 0 1.96.13 2.87.39 2.19-1.49 3.15-1.18 3.15-1.18.62 1.57.23 2.74.11 3.03.74.81 1.18 1.84 1.18 3.1 0 4.43-2.69 5.4-5.25 5.69.41.35.78 1.05.78 2.12v3.14c0 .3.21.66.8.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
  </svg>
);

