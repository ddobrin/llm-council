import './StageHeader.css';

export default function StageHeader({
  stageNumber,
  title,
  isLoading,
  loadingText,
}) {
  return (
    <header className="stage-header">
      <span className="stage-number">{stageNumber}</span>
      <h3 className="stage-title">{title}</h3>
      {isLoading && (
        <span className="stage-loading">
          <span className="loading-spinner" />
          {loadingText}
        </span>
      )}
    </header>
  );
}
