import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';

/**
 * MarkdownRenderer — Rich markdown rendering for tutor messages.
 * Supports: GFM tables, code highlighting, LaTeX math, headers, lists.
 */
export default function MarkdownRenderer({ content }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={{
          // Headers with accent dot
          h1: ({ children }) => (
            <h1 className="md-h1">
              <span className="md-heading-dot" />
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="md-h2">
              <span className="md-heading-dot" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="md-h3">
              <span className="md-heading-dot" />
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="md-h4">
              <span className="md-heading-dot" />
              {children}
            </h4>
          ),

          // Code blocks with copy button and language label
          pre: ({ children }) => (
            <div className="md-code-block-wrapper">
              <pre className="md-code-block">{children}</pre>
            </div>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return <code className="md-inline-code" {...props}>{children}</code>;
            }
            const lang = className?.replace('hljs language-', '').replace('language-', '') || '';
            return (
              <>
                {lang && <div className="md-code-lang">{lang}</div>}
                <code className={`${className || ''} md-code-content`} {...props}>
                  {children}
                </code>
              </>
            );
          },

          // Styled blockquotes
          blockquote: ({ children }) => (
            <blockquote className="md-blockquote">{children}</blockquote>
          ),

          // Lists
          ul: ({ children }) => <ul className="md-ul">{children}</ul>,
          ol: ({ children }) => <ol className="md-ol">{children}</ol>,
          li: ({ children }) => <li className="md-li">{children}</li>,

          // Links
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer" className="md-link">
              {children}
            </a>
          ),

          // Tables
          table: ({ children }) => (
            <div className="md-table-wrapper">
              <table className="md-table">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="md-th">{children}</th>,
          td: ({ children }) => <td className="md-td">{children}</td>,

          // Emphasis
          strong: ({ children }) => <strong className="md-strong">{children}</strong>,
          em: ({ children }) => <em className="md-em">{children}</em>,

          // Horizontal rule
          hr: () => <hr className="md-hr" />,

          // Paragraph
          p: ({ children }) => <p className="md-p">{children}</p>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
