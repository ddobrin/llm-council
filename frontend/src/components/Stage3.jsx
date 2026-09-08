import ReactMarkdown from 'react-markdown';
import StageHeader from './StageHeader';
import { formatModelName } from '../utils/councilUtils';
import './StageHeader.css';
import './Stage3.css';

export default function Stage3({ finalResponse, isLoading = false }) {
  if (!finalResponse && !isLoading) {
    return null;
  }

  return (
    <section className="stage-panel stage3-panel">
      <StageHeader
        stageNumber={3}
        title="Final Synthesis"
        isLoading={isLoading}
        loadingText="Council Chairman synthesizing..."
      />

      {finalResponse && (
        <div className="final-response">
          <div className="final-response-content markdown-content">
            <ReactMarkdown>{finalResponse.response || ''}</ReactMarkdown>
          </div>
          <div className="chairman-badge">
            <span>Synthesized by {formatModelName(finalResponse.model)}</span>
          </div>
        </div>
      )}
    </section>
  );
}
