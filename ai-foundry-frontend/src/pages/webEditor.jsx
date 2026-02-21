import React, { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'

export default function WebEditor(){
  const navigate = useNavigate()
  const loc = useLocation()
  const [html, setHtml] = useState('<!-- Paste landing page HTML here -->')

  useEffect(() => {
    if (loc && loc.state && loc.state.html) {
      setHtml(loc.state.html)
    }
  }, [loc])

  return (
    <div className="max-w-[1000px] mx-auto my-7 px-5">
      <button onClick={() => navigate(-1)} className="mb-4 px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 rounded transition-colors">← Back</button>
      <h1 className="text-3xl font-bold mb-3">Web Editor</h1>
      <p className="mb-4 text-gray-600 dark:text-gray-400">Edit the generated landing page HTML. This is a minimal editor for quick tweaks.</p>

      <textarea 
        value={html} 
        onChange={(e)=>setHtml(e.target.value)} 
        className="w-full h-[400px] font-mono text-[13px] p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 resize-y"
      />

      <div className="mt-3">
        <button 
          onClick={() => {
            const w = window.open();
            w.document.write(html);
            w.document.close();
          }}
          className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors"
        >
          Open Preview in New Window
        </button>
      </div>
    </div>
  )
}
