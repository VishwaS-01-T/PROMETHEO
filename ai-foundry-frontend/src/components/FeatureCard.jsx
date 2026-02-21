import React, { useState } from 'react'
import { motion } from 'framer-motion'

const FeatureCard = ({ icon: Icon, title, description, delay = 0, isExpanded, onToggle }) => {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.5, delay }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      whileHover={{ scale: 1.05, y: -8, boxShadow: '0 20px 50px rgba(87, 38, 125, 0.5)' }}
      className="glass rounded-2xl p-6 md:p-8 transition-all duration-300 group overflow-hidden"
      style={{
        aspectRatio: '1/1',
        background: '#57267D',
      }}
    >
      <div className="flex flex-col h-full relative p-4 gap-2">
        {/* Icon and Title - top section */}
        <div className="flex flex-col items-center justify-start gap-3 icon-title-container flex-shrink-0">
          <div className="relative icon-container">
            {/* Animated icon container */}
            <div className="icon-animate relative">
              <Icon className="w-14 h-14 md:w-16 md:h-16 stroke-white" strokeWidth={1.5} />
            </div>
          </div>

          <h3 className="urbanist-heading text-2xl md:text-3xl font-bold text-white card-title transition-all duration-300 text-center"
            style={{
              color: 'white',
              position: 'relative',
              zIndex: isHovered ? 20 : 10,
            }}>
            {title}
          </h3>
        </div>

        {/* Description - bottom section with more padding */}
        <div className="flex-1 flex items-center justify-center px-3 py-2 min-h-0">
          <p
            className="card-description text-white/90 text-sm md:text-base leading-relaxed opacity-0 text-center overflow-hidden"
            style={{ fontFamily: 'Roboto, sans-serif' }}
          >
            {description}
          </p>
        </div>
      </div>
    </motion.div>
  )
}

export default FeatureCard
