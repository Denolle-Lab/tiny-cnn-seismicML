import { useEffect } from 'react';
import { glossary } from '../data/glossary';
import './GlossaryModal.css';

export function GlossaryModal({ termKey, onClose }) {
  const entry = termKey ? glossary[termKey] : null;

  useEffect(() => {
    const handleKey = (e) => {
      if (termKey && e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [termKey, onClose]);

  if (!entry) return null;

  return (
    <div className="glossary-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-label={`${entry.title} definition`}>
      <div className="glossary-modal" onClick={(e) => e.stopPropagation()}>
        <div className="glossary-modal-header">
          <h3>{entry.title}</h3>
          <button type="button" className="glossary-close" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="glossary-modal-body" dangerouslySetInnerHTML={{ __html: entry.body }} />
      </div>
    </div>
  );
}

export function GlossaryTerm({ termKey, children, onOpen }) {
  return (
    <button type="button" className="glossary-term" onClick={() => onOpen(termKey)}>
      {children}
    </button>
  );
}

