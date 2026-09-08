import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import StageHeader from './StageHeader';
import {
  formatModelName,
  getAvatarColor,
  getEvaluationText,
  deAnonymizeText,
} from '../utils/councilUtils';
import './StageHeader.css';
import './Stage2.css';

export default function Stage2({
  rankings = [],
  labelToModel = {},
  aggregateRankings = [],
  isLoading = false,
  totalModels = 3,
}) {
  const [activeTab, setActiveTab] = useState(0);

  if ((!rankings || rankings.length === 0) && !isLoading) {
    return null;
  }

  const getRankClass = (index) => {
    if (index === 0) return 'rank-1';
    if (index === 1) return 'rank-2';
    if (index === 2) return 'rank-3';
    return 'rank-other';
  };

  const selectedRanking = rankings && rankings[activeTab];

  return (
    <section className="stage-panel stage2-panel">
      <StageHeader
        stageNumber={2}
        title="Peer Review & Rankings"
        isLoading={isLoading}
        loadingText={`Collecting rankings (${rankings ? rankings.length : 0}/${totalModels})`}
      />

      {labelToModel && Object.keys(labelToModel).length > 0 && (
        <div className="model-legend">
          {Object.entries(labelToModel).map(([label, modelId]) => (
            <span key={label} className="legend-item">
              <strong>{label.replace('Response ', 'Model ')}</strong>: {formatModelName(modelId)}
            </span>
          ))}
        </div>
      )}

      {rankings && rankings.length > 0 && (
        <>
          <div className="response-tabs">
            {rankings.map((ranking, index) => (
              <button
                key={ranking.model || index}
                type="button"
                className={`response-tab ${activeTab === index ? 'active' : ''}`}
                onClick={() => setActiveTab(index)}
              >
                <span
                  className="model-avatar"
                  style={{ backgroundColor: getAvatarColor(ranking.model, index) }}
                />
                <span className="tab-model-name">
                  {formatModelName(ranking.model)}
                </span>
              </button>
            ))}
          </div>

          {selectedRanking && (
            <div className="ranking-display">
              <div className="evaluation-content markdown-content">
                <ReactMarkdown>
                  {deAnonymizeText(getEvaluationText(selectedRanking.ranking), labelToModel)}
                </ReactMarkdown>
              </div>

              {selectedRanking.parsed_ranking && selectedRanking.parsed_ranking.length > 0 && (
                <div className="ranking-section">
                  <h4 className="ranking-section-title">Final Ranking</h4>
                  <ol className="ranking-list">
                    {selectedRanking.parsed_ranking.map((label, index) => {
                      const actualModel = labelToModel && labelToModel[label];
                      return (
                        <li key={label || index}>
                          <strong>{label}</strong>
                          {actualModel && (
                            <span className="ranking-model-reference">
                              ({formatModelName(actualModel)})
                            </span>
                          )}
                        </li>
                      );
                    })}
                  </ol>
                </div>
              )}

              <div className="response-meta">
                <span>Model: {selectedRanking.model}</span>
              </div>
            </div>
          )}
        </>
      )}

      {aggregateRankings && aggregateRankings.length > 0 && (
        <div className="aggregate-rankings">
          <h4 className="aggregate-title">Aggregate Rankings</h4>
          <div className="aggregate-list">
            {aggregateRankings.map((ranking, index) => (
              <div key={ranking.model || index} className="aggregate-item">
                <span className={`rank-badge ${getRankClass(index)}`}>
                  {index + 1}
                </span>
                <span className="aggregate-model">
                  {formatModelName(ranking.model)}
                </span>
                <span className="aggregate-score">
                  Avg. rank: {ranking.average_rank ? ranking.average_rank.toFixed(2) : 'N/A'}
                </span>
                {ranking.rankings_count !== undefined && (
                  <span className="aggregate-votes">
                    ({ranking.rankings_count} votes)
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
