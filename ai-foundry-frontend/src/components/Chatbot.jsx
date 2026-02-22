import React, { useState, useRef, useEffect } from 'react'

const BOT_AVATAR = '🤖'
const USER_AVATAR = '👤'

/**
 * Floating BRD Chatbot widget.
 *
 * Props:
 *   brdMarkdown      – raw markdown of the generated BRD
 *   strategyMarkdown  – raw markdown of the strategic approach
 */
export default function Chatbot({ brdMarkdown, strategyMarkdown }) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        "Hi! I'm your **BRD Assistant**. Ask me anything about the generated Business Requirements Document or Strategy — I'll answer based on the documents.",
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, loading])

  // Focus input when chat opens
  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus()
    }
  }, [open])

  async function sendMessage(e) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    // Add user message
    const userMsg = { role: 'user', content: question }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          brd_markdown: brdMarkdown || null,
          strategy_markdown: strategyMarkdown || null,
          history: messages.slice(-10).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      })

      const data = await res.json()

      if (data.success) {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.answer },
        ])
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `⚠️ Sorry, something went wrong: ${data.error || 'Unknown error'}`,
          },
        ])
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Could not reach the server. Make sure the backend is running.`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // Simple markdown-ish rendering (bold, bullets, code)
  function renderMarkdown(text) {
    if (!text) return ''
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code style="background:rgba(139,92,246,0.15);padding:1px 5px;border-radius:4px;font-size:12px">$1</code>')
      .replace(/^- (.*)/gm, '<span style="display:block;padding-left:12px">• $1</span>')
      .replace(/\n/g, '<br/>')
  }

  const hasContext = !!(brdMarkdown || strategyMarkdown)

  return (
    <>
      {/* ─── Floating Action Button ─── */}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? 'Close chat' : 'Open BRD Assistant'}
        style={{
          position: 'fixed',
          bottom: 28,
          right: 28,
          zIndex: 9999,
          width: 60,
          height: 60,
          borderRadius: '50%',
          border: 'none',
          cursor: 'pointer',
          background: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)',
          boxShadow: '0 6px 24px rgba(139, 92, 246, 0.45)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'transform 0.25s, box-shadow 0.25s',
          transform: open ? 'rotate(90deg) scale(0.9)' : 'rotate(0) scale(1)',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.transform = open ? 'rotate(90deg) scale(0.95)' : 'scale(1.08)')}
        onMouseLeave={(e) => (e.currentTarget.style.transform = open ? 'rotate(90deg) scale(0.9)' : 'scale(1)')}
      >
        {open ? (
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        ) : (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        )}
      </button>

      {/* ─── Chat Panel ─── */}
      {open && (
        <div
          style={{
            position: 'fixed',
            bottom: 100,
            right: 28,
            zIndex: 9998,
            width: 400,
            maxWidth: 'calc(100vw - 40px)',
            height: 520,
            maxHeight: 'calc(100vh - 140px)',
            borderRadius: 20,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            background: '#fff',
            border: '1px solid #e5e7eb',
            boxShadow: '0 20px 60px rgba(0,0,0,0.18), 0 0 0 1px rgba(139,92,246,0.1)',
            animation: 'chatSlideUp 0.3s ease-out',
          }}
        >
          {/* Header */}
          <div
            style={{
              background: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)',
              padding: '16px 20px',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              flexShrink: 0,
            }}
          >
            <div
              style={{
                width: 38,
                height: 38,
                borderRadius: '50%',
                background: 'rgba(255,255,255,0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 20,
              }}
            >
              {BOT_AVATAR}
            </div>
            <div>
              <div style={{ color: '#fff', fontWeight: 700, fontSize: 15, fontFamily: 'Urbanist, sans-serif' }}>
                BRD Assistant
              </div>
              <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: 12 }}>
                {hasContext ? '● Grounded in your BRD' : '○ Waiting for BRD generation…'}
              </div>
            </div>
          </div>

          {/* Context banner */}
          {!hasContext && (
            <div
              style={{
                background: '#fef3c7',
                color: '#92400e',
                fontSize: 12,
                padding: '8px 16px',
                textAlign: 'center',
                flexShrink: 0,
              }}
            >
              ⏳ The BRD hasn't been generated yet. Responses won't be grounded until agents complete.
            </div>
          )}

          {/* Messages area */}
          <div
            ref={scrollRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '16px 16px 8px',
              display: 'flex',
              flexDirection: 'column',
              gap: 12,
              background: '#fafafa',
            }}
          >
            {messages.map((msg, i) => {
              const isUser = msg.role === 'user'
              return (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    gap: 8,
                    flexDirection: isUser ? 'row-reverse' : 'row',
                    alignItems: 'flex-end',
                  }}
                >
                  <div
                    style={{
                      width: 30,
                      height: 30,
                      borderRadius: '50%',
                      background: isUser ? '#e0e7ff' : 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 14,
                      flexShrink: 0,
                      color: isUser ? '#4338ca' : '#fff',
                    }}
                  >
                    {isUser ? USER_AVATAR : BOT_AVATAR}
                  </div>
                  <div
                    style={{
                      maxWidth: '78%',
                      padding: '10px 14px',
                      borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
                      background: isUser
                        ? 'linear-gradient(135deg, #8b5cf6, #7c3aed)'
                        : '#fff',
                      color: isUser ? '#fff' : '#1f2937',
                      fontSize: 13.5,
                      lineHeight: 1.55,
                      boxShadow: isUser ? 'none' : '0 1px 4px rgba(0,0,0,0.08)',
                      border: isUser ? 'none' : '1px solid #e5e7eb',
                    }}
                    dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                  />
                </div>
              )
            })}

            {/* Typing indicator */}
            {loading && (
              <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
                <div
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 14,
                    color: '#fff',
                    flexShrink: 0,
                  }}
                >
                  {BOT_AVATAR}
                </div>
                <div
                  style={{
                    padding: '12px 18px',
                    borderRadius: '16px 16px 16px 4px',
                    background: '#fff',
                    border: '1px solid #e5e7eb',
                    display: 'flex',
                    gap: 5,
                  }}
                >
                  <span className="chatbot-dot" style={{ animationDelay: '0s' }} />
                  <span className="chatbot-dot" style={{ animationDelay: '0.15s' }} />
                  <span className="chatbot-dot" style={{ animationDelay: '0.3s' }} />
                </div>
              </div>
            )}
          </div>

          {/* Input area */}
          <form
            onSubmit={sendMessage}
            style={{
              display: 'flex',
              gap: 8,
              padding: '12px 16px',
              borderTop: '1px solid #e5e7eb',
              background: '#fff',
              flexShrink: 0,
            }}
          >
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={hasContext ? 'Ask about your BRD…' : 'Ask a question…'}
              disabled={loading}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: 12,
                border: '1.5px solid #e5e7eb',
                outline: 'none',
                fontSize: 13.5,
                fontFamily: 'inherit',
                color: '#1f2937',
                background: '#fafafa',
                transition: 'border-color 0.2s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#8b5cf6')}
              onBlur={(e) => (e.target.style.borderColor = '#e5e7eb')}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              style={{
                width: 42,
                height: 42,
                borderRadius: 12,
                border: 'none',
                background:
                  loading || !input.trim()
                    ? '#d1d5db'
                    : 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)',
                color: '#fff',
                cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                transition: 'opacity 0.2s, transform 0.15s',
              }}
              onMouseEnter={(e) => { if (!loading && input.trim()) e.currentTarget.style.transform = 'scale(1.06)' }}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </form>
        </div>
      )}

      {/* Injected keyframe styles */}
      <style>{`
        @keyframes chatSlideUp {
          from { opacity: 0; transform: translateY(20px) scale(0.96); }
          to   { opacity: 1; transform: translateY(0)   scale(1);    }
        }
        .chatbot-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #a78bfa;
          animation: chatBounce 1.2s infinite ease-in-out;
          display: inline-block;
        }
        @keyframes chatBounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30%            { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>
    </>
  )
}
