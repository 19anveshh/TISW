import React from 'react';
import GlassSheet from './GlassSheet';

export default function AboutSheet({ onClose }) {
  return (
    <GlassSheet title="About Our Platform" onClose={onClose} closeLabel="Close About">
      <div className="sheet-prose">
        <p>
          TISW turns your documents into an interactive learning and knowledge platform.
          Instead of spending hours searching through lengthy notes, textbooks, or PDFs, you
          can ask questions in natural language and get clear, relevant answers based on your
          study materials.
        </p>
        <p>
          What makes TISW different is its combination of AI-powered document search,
          intelligent retrieval, and structured study evaluation. Beyond answering questions,
          it helps students evaluate their understanding through structured assessments based
          on their learning content, making it easier to identify knowledge gaps and focus on
          areas that need improvement.
        </p>
        <p>
          Whether you're preparing for exams, revising complex topics, or learning something
          new, TISW helps you study smarter, save time, and understand concepts more
          effectively.
        </p>
      </div>
    </GlassSheet>
  );
}
