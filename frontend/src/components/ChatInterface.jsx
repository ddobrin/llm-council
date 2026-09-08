import { useState } from 'react';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import { computeLabelToModel } from '../utils/councilUtils';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  onSendMessage,
  onNewSession,
  isLoading,
  councilModels = [],
}) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Find the last assistant message to determine active progress & stage
  const messages = conversation?.messages || [];
  const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;
  const isLastAssistant = lastMessage?.role === 'assistant';

  let currentStageLabel = '';
  let progressPercent = 0;

  if (isLoading && isLastAssistant) {
    if (lastMessage.loading?.stage1) {
      currentStageLabel = 'Individual Review: Collecting Responses...';
      progressPercent = 33;
    } else if (lastMessage.loading?.stage2) {
      currentStageLabel = 'Peer Ranking: Evaluating Responses...';
      progressPercent = 66;
    } else if (lastMessage.loading?.stage3) {
      currentStageLabel = 'Final Synthesis: Chairman Deliberating...';
      progressPercent = 95;
    } else {
      currentStageLabel = 'Consulting the Council...';
      progressPercent = 15;
    }
  } else if (!isLoading && messages.length > 0 && isLastAssistant && lastMessage.stage3) {
    currentStageLabel = 'Complete';
    progressPercent = 100;
  }

  return (
    <div className="chat-layout">
      {/* Top Navbar matching Java MainLayout */}
      <header className="top-navbar">
        <h2 className="navbar-title">LLM Council</h2>
        <div className="view-switcher">
          <button type="button" className="view-switch-btn active">
            LLM Workflow Council
          </button>
        </div>
      </header>

      {/* Main scrollable content */}
      <main className="main-content-scroll">
        <div className="council-view">
          {/* Query Section */}
          <section className="query-section">
            <label className="query-label" htmlFor="council-query">
              Ask the Council
            </label>
            <textarea
              id="council-query"
              className="query-input"
              placeholder="Enter your question for the LLM Council..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={3}
            />

            <div className="mode-actions">
              <div className="mode-action-btn-wrapper">
                <button
                  type="button"
                  className="btn-primary-action"
                  onClick={handleSubmit}
                  disabled={!input.trim() || isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="btn-spinner" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    'Consult the Council'
                  )}
                </button>
                <div className="btn-tooltip">
                  3 stages. Every model answers, ranks the others anonymously, then the chairman writes one synthesized answer.
                </div>
              </div>
            </div>

            {messages.length > 0 && !isLoading && (
              <div className="new-session-action">
                <button
                  type="button"
                  className="btn-tertiary"
                  onClick={() => {
                    setInput('');
                    if (onNewSession) onNewSession();
                  }}
                >
                  Start New Session
                </button>
              </div>
            )}
          </section>

          {/* Progress Section */}
          {isLoading && (
            <section className="progress-section">
              <div className="progress-label">{currentStageLabel}</div>
              <div className="progress-bar-track">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </section>
          )}

          {/* Messages / Deliberation Flow */}
          {messages.map((msg, index) => {
            if (msg.role === 'user') {
              return (
                <div key={index} className="user-query-card">
                  <div className="user-query-header">
                    <span className="user-badge">Question</span>
                  </div>
                  <div className="user-query-text">{msg.content}</div>
                </div>
              );
            }

            if (msg.role === 'assistant') {
              // Ensure labelToModel is available
              const labelToModel =
                msg.metadata?.label_to_model ||
                computeLabelToModel(msg.stage1, councilModels);

              const aggregateRankings = msg.metadata?.aggregate_rankings;

              return (
                <div key={index} className="assistant-stages-group">
                  {/* Stage 1: Individual Review */}
                  <Stage1
                    responses={msg.stage1}
                    isLoading={msg.loading?.stage1}
                    totalModels={councilModels.length || 3}
                    labelToModel={labelToModel}
                  />

                  {/* Stage 2: Peer Review & Rankings */}
                  <Stage2
                    rankings={msg.stage2}
                    labelToModel={labelToModel}
                    aggregateRankings={aggregateRankings}
                    isLoading={msg.loading?.stage2}
                    totalModels={councilModels.length || 3}
                  />

                  {/* Stage 3: Final Synthesis */}
                  <Stage3
                    finalResponse={msg.stage3}
                    isLoading={msg.loading?.stage3}
                  />
                </div>
              );
            }

            return null;
          })}
        </div>
      </main>
    </div>
  );
}
