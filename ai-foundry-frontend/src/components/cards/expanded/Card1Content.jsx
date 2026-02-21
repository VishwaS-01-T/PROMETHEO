import { motion } from 'framer-motion'

function Card1Content({ brdUrl }) {
  const handleGenerateBRD = () => {
    if (brdUrl) {
      // If BRD URL exists from the backend, download it
      window.open(brdUrl, '_blank')
    } else {
      // Fallback to generating a sample BRD
      const brdContent = `BUSINESS REQUIREMENT DOCUMENT (BRD)\n\nProject: Website Development & Marketing Automation\nDate: ${new Date().toLocaleDateString()}\n\n1. PROJECT OVERVIEW\n   - Goal: Create a comprehensive website with automated marketing capabilities\n   - Target Audience: Small to medium businesses\n   - Timeline: 8-12 weeks\n\nEND OF DOCUMENT`.trim()

      const blob = new Blob([brdContent], { type: 'text/plain' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'BRD.txt'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      alert('BRD.txt has been downloaded successfully!')
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="space-y-6">
      <h2 className="text-4xl font-bold text-white mb-6">Get breakdown of your final goal</h2>
      
      <div className="bg-black/40 rounded-2xl p-6 border border-white/10">
        <h3 className="text-2xl font-semibold text-purple-300 mb-4">Goal Breakdown</h3>
        <div className="space-y-4 text-white/90">
          <div className="border-l-4 border-purple-500 pl-4">
            <h4 className="font-bold text-lg mb-2">1. Website Development</h4>
            <p className="text-white/70">Create a modern, responsive website using React and Tailwind CSS</p>
          </div>
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="font-bold text-lg mb-2">2. Code Generation</h4>
            <p className="text-white/70">Generate website code with a code editor interface and live preview functionality.</p>
          </div>
        </div>
      </div>

      {/* BRD Section - Integrated from Report page */}
      <div className="bg-gradient-to-br from-[#1e3a5f] to-[#2c5282] rounded-2xl p-6 border border-blue-500/30 shadow-lg">
        <h3 className="text-2xl font-bold text-white mb-4">Business Requirements Document</h3>
        {brdUrl ? (
          <div className="space-y-4">
            <div className="bg-white/5 backdrop-blur-[10px] border border-white/10 rounded-xl p-6">
              <p className="text-blue-300 text-base mb-4">✅ Your BRD has been generated successfully!</p>
              <a 
                href={brdUrl}
                download
                className="inline-block py-3 px-6 bg-gradient-to-br from-blue-500 to-blue-800 text-white no-underline rounded-lg font-semibold text-base transition-all duration-300 shadow-[0_4px_12px_rgba(59,130,246,0.3)] hover:translate-y-[-2px] hover:shadow-[0_6px_20px_rgba(59,130,246,0.4)]"
              >
                Download BRD PDF
              </a>
            </div>
          </div>
        ) : (
          <div className="text-gray-400 italic text-center py-6">
            The Business Requirements Document will appear here once the BRD Agent completes.
          </div>
        )}
      </div>

      <motion.button 
        whileHover={{ scale: 1.05 }} 
        whileTap={{ scale: 0.95 }} 
        onClick={handleGenerateBRD} 
        className="w-full py-4 px-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-xl shadow-lg"
        disabled={!brdUrl}
      >
        {brdUrl ? 'Download BRD' : 'Waiting for BRD Generation...'}
      </motion.button>
    </motion.div>
  )
}

export default Card1Content
