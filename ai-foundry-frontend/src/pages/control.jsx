import React, { useState } from 'react'

export default function Control() {
  // KB Creation State
  const [kbUrl, setKbUrl] = useState('')
  const [kbLoading, setKbLoading] = useState(false)
  const [kbMessage, setKbMessage] = useState('')

  // Handle KB Creation
  const handleCreateKB = async () => {
    if (!kbUrl) {
      setKbMessage('❌ Please enter a URL')
      return
    }

    setKbLoading(true)
    setKbMessage('⏳ Creating knowledge base...')

    try {
      const response = await fetch('http://localhost:8002/create-knowledge-base', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: kbUrl })
      })

      if (response.ok) {
        setKbMessage('✅ Knowledge base creation started! (This may take a few minutes)')
        setKbUrl('')
      } else {
        const error = await response.json()
        setKbMessage(`❌ Failed: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      setKbMessage(`❌ Error: ${error.message}`)
    }

    setKbLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 py-12 px-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-5xl font-bold text-white mb-4 text-center">🎛️ Control Center</h1>
        <p className="text-xl text-gray-300 mb-12 text-center">Manage Knowledge Base</p>

        <div className="max-w-2xl mx-auto">
          {/* Knowledge Base Section */}
          <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all duration-300 hover:shadow-2xl">
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-white mb-2">📚 Create Knowledge Base</h2>
              <p className="text-gray-300 text-sm">Upload content from a URL to create a searchable knowledge base</p>
            </div>

            <div className="space-y-4">
              <input
                type="url"
                placeholder="Enter URL (e.g., https://example.com/docs)"
                value={kbUrl}
                onChange={(e) => setKbUrl(e.target.value)}
                className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={kbLoading}
              />

              <button
                onClick={handleCreateKB}
                className={`w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-700 hover:from-blue-600 hover:to-blue-800 text-white font-semibold rounded-lg transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 ${kbLoading ? 'animate-pulse' : ''}`}
                disabled={kbLoading}
              >
                {kbLoading ? (
                  <>
                    <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span> Processing...
                  </>
                ) : (
                  '📥 Create Knowledge Base'
                )}
              </button>

              {kbMessage && (
                <div className={`p-4 rounded-lg ${kbMessage.includes('✅') ? 'bg-green-500/20 text-green-200 border border-green-500/50' : kbMessage.includes('⏳') ? 'bg-blue-500/20 text-blue-200 border border-blue-500/50' : 'bg-red-500/20 text-red-200 border border-red-500/50'}`}>
                  {kbMessage}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
