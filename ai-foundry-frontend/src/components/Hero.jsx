import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import Button from './Button'
import { useNavigate } from "react-router-dom";
import React from 'react'

const Hero = () => {
    const navigate = useNavigate();
  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 px-4 sm:px-6 lg:px-8 overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-400/20 rounded-full blur-3xl animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-400/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }}></div>
      </div>

      <div className="relative z-10 container mx-auto text-center max-w-4xl">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-5xl md:text-6xl lg:text-7xl font-bold text-gray-900 dark:text-white mb-6 leading-tight"
        >
          Build Powerful{' '}
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            AI Agents
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-xl md:text-2xl text-gray-600 dark:text-gray-400 mb-10 max-w-2xl mx-auto"
        >
          For yourself, your team, or your enterprise—no coding required.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Button onClick={() => navigate("/workflow")} variant="primary" size="lg">
            Get Started for Free
          </Button>
          <motion.a
            href="#features"
            className="flex items-center text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 font-semibold text-lg group transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-lg px-2"
            whileHover={{ x: 5 }}
            aria-label="Learn more about features"
          >
            Know more
            <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" aria-hidden="true" />
          </motion.a>
        </motion.div>

        {/* Floating Elements */}
        <motion.div
          animate={{ y: [0, -10, 0] }}
          transition={{ duration: 3, repeat: Infinity }}
          className="absolute top-20 left-10 hidden lg:block"
        >
          <div className="glass rounded-2xl p-4 shadow-lg">
            <div className="w-12 h-12 bg-blue-500 rounded-lg"></div>
          </div>
        </motion.div>

        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 3, repeat: Infinity, delay: 1 }}
          className="absolute bottom-20 right-10 hidden lg:block"
        >
          <div className="glass rounded-2xl p-4 shadow-lg">
            <div className="w-12 h-12 bg-purple-500 rounded-lg"></div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default Hero

