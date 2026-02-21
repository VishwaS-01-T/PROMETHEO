import { motion } from 'framer-motion'
import { useState } from 'react'
import React from 'react'

const FeatureCard = ({ icon: Icon, title, description, delay = 0 }) => {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, delay }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      className="glass rounded-2xl p-6 md:p-8 hover:shadow-xl transition-all duration-300 cursor-pointer group"
    >
      <motion.div
        animate={{ 
          scale: isHovered ? 1.1 : 1,
          rotate: isHovered ? 5 : 0
        }}
        transition={{ duration: 0.3 }}
        className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mb-6 shadow-lg"
      >
        <Icon className="w-8 h-8 text-white" />
      </motion.div>
      
      <h3 className="text-xl md:text-2xl font-bold text-gray-900 dark:text-white mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200">
        {title}
      </h3>
      
      <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
        {description}
      </p>
    </motion.div>
  )
}

export default FeatureCard

