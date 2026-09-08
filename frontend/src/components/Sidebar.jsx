import DrawerRoster from './DrawerRoster';
import './Sidebar.css';

export default function Sidebar({
  conversations = [],
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  councilModels = [],
  chairmanModel = '',
}) {
  const formatDate = (isoStr) => {
    if (!isoStr) return '';
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return '';
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">LLM Council</h1>
        <p className="sidebar-subtitle">Collaborative AI Deliberation</p>
        <button
          type="button"
          className="new-session-btn"
          onClick={onNewConversation}
        >
          + New Session
        </button>
      </div>

      <nav className="sidebar-nav">
        <h3 className="session-list-header">Saved Sessions</h3>
        <div className="session-list">
          {conversations.length === 0 ? (
            <div className="no-sessions">No saved sessions</div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === currentConversationId;
              return (
                <button
                  key={conv.id}
                  type="button"
                  className={`session-item ${isActive ? 'active' : ''}`}
                  onClick={() => onSelectConversation(conv.id)}
                >
                  <span className="session-title">
                    {conv.title || 'Untitled Session'}
                  </span>
                  <div className="session-meta">
                    {conv.created_at && <span>{formatDate(conv.created_at)}</span>}
                    {conv.message_count !== undefined && (
                      <span>
                        {conv.created_at ? ' • ' : ''}
                        {conv.message_count} {conv.message_count === 1 ? 'msg' : 'msgs'}
                      </span>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>
      </nav>

      <DrawerRoster
        councilModels={councilModels}
        chairmanModel={chairmanModel}
      />
    </aside>
  );
}
