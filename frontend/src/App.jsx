import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [councilConfig, setCouncilConfig] = useState({
    council_models: [
      'gemini-3.6-flash',
      'gemini-3.7-flash',
      'gemini-3.8-flash',
    ],
    chairman_model: 'gemini-3.1-pro-preview',
  });
  const [isLoading, setIsLoading] = useState(false);

  const loadConversation = useCallback(async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  }, []);

  const loadConversations = useCallback(async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }, []);

  // Load council configuration & initial conversation on mount
  useEffect(() => {
    let isMounted = true;

    const init = async () => {
      try {
        const config = await api.getCouncil();
        if (config && isMounted) {
          setCouncilConfig(config);
        }
      } catch (e) {
        console.warn('Could not load council config:', e);
      }

      try {
        const convs = await api.listConversations();
        if (!isMounted) return;
        setConversations(convs);
        if (convs.length > 0) {
          const firstId = convs[0].id;
          setCurrentConversationId(firstId);
          const firstConv = await api.getConversation(firstId);
          if (isMounted) {
            setCurrentConversation(firstConv);
          }
        }
      } catch (e) {
        console.error('Failed to list conversations:', e);
      }
    };

    init();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations((prev) => [
        { id: newConv.id, created_at: newConv.created_at, message_count: 0, title: 'New Conversation' },
        ...prev,
      ]);
      setCurrentConversationId(newConv.id);
      setCurrentConversation(newConv);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = async (id) => {
    setCurrentConversationId(id);
    await loadConversation(id);
  };

  const handleSendMessage = async (content) => {
    let targetConvId = currentConversationId;

    setIsLoading(true);
    try {
      // If no conversation exists, create one first
      if (!targetConvId) {
        const newConv = await api.createConversation();
        targetConvId = newConv.id;
        setCurrentConversationId(targetConvId);
        setCurrentConversation(newConv);
        setConversations((prev) => [
          { id: newConv.id, created_at: newConv.created_at, message_count: 0, title: 'New Conversation' },
          ...prev,
        ]);
      }

      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: true,
          stage2: false,
          stage3: false,
        },
      };

      setCurrentConversation((prev) => {
        const existingMessages = prev ? prev.messages || [] : [];
        return {
          ...prev,
          id: targetConvId,
          messages: [...existingMessages, userMessage, assistantMessage],
        };
      });

      // Send message with streaming
      await api.sendMessageStream(targetConvId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.loading = { ...lastMsg.loading, stage1: true };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'stage1_complete':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage1 = event.data;
              lastMsg.loading = { ...lastMsg.loading, stage1: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'stage2_start':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.loading = { ...lastMsg.loading, stage2: true };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'stage2_complete':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage2 = event.data;
              lastMsg.metadata = event.metadata;
              lastMsg.loading = { ...lastMsg.loading, stage2: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'stage3_start':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.loading = { ...lastMsg.loading, stage3: true };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'stage3_complete':
            setCurrentConversation((prev) => {
              if (!prev || !prev.messages) return prev;
              const messages = [...prev.messages];
              const lastMsg = { ...messages[messages.length - 1] };
              lastMsg.stage3 = event.data;
              lastMsg.loading = { ...lastMsg.loading, stage3: false };
              messages[messages.length - 1] = lastMsg;
              return { ...prev, messages };
            });
            break;

          case 'title_complete':
            loadConversations();
            if (event.data?.title) {
              setCurrentConversation((prev) =>
                prev ? { ...prev, title: event.data.title } : prev
              );
            }
            break;

          case 'complete':
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          default:
            break;
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        councilModels={councilConfig.council_models}
        chairmanModel={councilConfig.chairman_model}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        onNewSession={handleNewConversation}
        isLoading={isLoading}
        councilModels={councilConfig.council_models}
      />
    </div>
  );
}

export default App;
