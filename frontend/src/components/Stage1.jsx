import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import StageHeader from './StageHeader';
import { formatModelName, getAvatarColor, getModelLabel } from '../utils/councilUtils';
import './StageHeader.css';
import './Stage1.css';

export default function Stage1({
  responses = [],
  isLoading = false,
  totalModels = 3,
  labelToModel = {},
}) {
  const [activeTab, setActiveTab] = useState(0);

  if ((!responses || responses.length === 0) && !isLoading) {
    return null;
  }

  const selectedResponse = responses && responses[activeTab];

  return (
    <section className="stage-panel stage1-panel">
      <StageHeader
        stageNumber={1}
        title="Individual Responses"
        isLoading={isLoading}
        loadingText={`Collecting responses (${responses ? responses.length : 0}/${totalModels})`}
      />

      {responses && responses.length > 0 && (
        <>
          <div className="response-tabs">
            {responses.map((resp, index) => {
              const modelLabel = getModelLabel(resp.model, labelToModel);
              return (
                <button
                  key={resp.model || index}
                  type="button"
                  className={`response-tab ${activeTab === index ? 'active' : ''}`}
                  onClick={() => setActiveTab(index)}
                >
                  <span
                    className="model-avatar"
                    style={{ backgroundColor: getAvatarColor(resp.model, index) }}
                  />
                  {modelLabel && (
                    <span className="model-label-badge">{modelLabel}</span>
                  )}
                  <span className="tab-model-name">
                    {formatModelName(resp.model)}
                  </span>
                </button>
              );
            })}
          </div>

          {selectedResponse && (
            <div className="response-display">
              <div className="response-content markdown-content">
                <ReactMarkdown>{selectedResponse.response || ''}</ReactMarkdown>
              </div>
              <div className="response-meta">
                <span>Model: {selectedResponse.model}</span>
                {selectedResponse.effort && (
                  <span className="response-effort-badge" title={`Reasoning effort: ${selectedResponse.effort}`}>
                    Effort: {selectedResponse.effort}
                  </span>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
