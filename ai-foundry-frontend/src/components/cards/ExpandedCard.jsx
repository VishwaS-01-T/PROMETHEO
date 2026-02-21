import { motion, AnimatePresence } from 'framer-motion'
import { useEffect } from 'react'
import Card1Content from './expanded/Card1Content'
import Card2Content from './expanded/Card2Content'
import Card3Content from './expanded/Card3Content'
import Card4Content from './expanded/Card4Content'

function ExpandedCard({ cardId, onClose, brdUrl, landingPageCode, contentData, generatedAssets }) {
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = 'unset' }
  }, [])

  useEffect(() => {
    const handleEscape = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [onClose])

  const renderContent = () => {
    switch (cardId) {
      case 1: return <Card1Content brdUrl={brdUrl} />
      case 2: return <Card2Content landingPageCode={landingPageCode} />
      case 3: return <Card3Content contentData={contentData} generatedAssets={generatedAssets} />
      case 4: return <Card4Content />
      default: return null
    }
  }

  const handleBackdropClick = (e) => { if (e.target === e.currentTarget) onClose() }

  return (
    <AnimatePresence>
      {cardId ? (
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }} transition={{ duration: 0.3 }} onClick={handleBackdropClick} className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md overflow-y-auto">
          <motion.button whileHover={{ scale: 1.1, rotate: 90 }} whileTap={{ scale: 0.9 }} onClick={onClose} className="absolute top-6 right-6 z-50 w-12 h-12 rounded-full bg-red-500/20 hover:bg-red-500/40 border border-red-500/50 flex items-center justify-center text-white">✕</motion.button>

          <motion.div initial={{ y: 50, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.4, delay: 0.1 }} onClick={(e) => e.stopPropagation()} className="w-full max-w-7xl my-8 bg-gradient-to-br from-gray-800 via-purple-900/30 to-blue-900/30 rounded-3xl p-8 border border-white/10 shadow-2xl">
            {renderContent()}
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

export default ExpandedCard
