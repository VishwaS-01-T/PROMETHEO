import { motion } from 'framer-motion'

// small helper to convert hex -> rgba string used for shadows/gradients
function hexToRgba(hex, alpha = 1) {
  const h = hex.replace('#', '')
  const bigint = parseInt(h, 16)
  const r = (bigint >> 16) & 255
  const g = (bigint >> 8) & 255
  const b = bigint & 255
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function Card({ id, title, description, accentColor = 'accent', onClick }) {
  const colors = ['#f472b6','#a855f7', '#ec4899', '#fb7185']

  return (
    <motion.div
      whileHover={{
        scale: 1.045,
        y: -10,
        boxShadow: `0 30px 60px ${hexToRgba(colors[0], 0.12)}, 0 12px 24px rgba(2,6,23,0.5)`,
      }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      className={`
        relative cursor-pointer rounded-2xl p-8
        bg-gradient-to-br from-black/40 to-black/30
        border border-white/8
        backdrop-blur-sm
        transition-all duration-300
        min-h-[240px]
        flex flex-col justify-between
        group
        overflow-visible
      `}
    >

      <div className="absolute inset-0 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none group-hover:pointer-events-auto">
        <div className="w-full h-full p-3">
          <div
            aria-hidden
            className="w-full h-full rounded-2xl"
            style={{
              background: `radial-gradient(420px 220px at 30% 50%, ${hexToRgba(
                colors[0],
                0.28
              )}, transparent 30%), radial-gradient(420px 220px at 90% 80%, ${hexToRgba(
                colors[1],
                0.22
              )}, transparent 25%)`,
              filter: 'blur(36px)',
              transform: 'scale(1.05)'
            }}
          />
        </div>
      </div>

      <div className="relative z-10">
        <div className="text-4xl font-bold mb-2 text-white/20">
          {String(id).padStart(2, '0')}
        </div>
        <h3 className="text-xl font-bold text-white mb-3 leading-tight">
          {title}
        </h3>
        <p className="text-white/80 text-sm">
          {description}
        </p>
      </div>
    </motion.div>
  )
}

export default Card
