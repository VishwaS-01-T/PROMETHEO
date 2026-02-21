import { useEffect } from 'react'
import Header from '../components/Header'
import Hero from '../components/Hero'
import Features from '../components/Features'
import HowItWorks from '../components/HowItWorks'
import Footer from '../components/Footer'
import React from 'react'

function LandingPage() {

  useEffect(() => {
    // Set initial dark mode based on system preference or localStorage
    const saved = localStorage.getItem('darkMode')
    const isDark = saved !== null 
      ? saved === 'true' 
      : window.matchMedia('(prefers-color-scheme: dark)').matches
    
    if (isDark) {
      document.documentElement.classList.add('dark')
    }
  }, [])

  return (
    <div className="min-h-screen">
      {/* Skip to content link for accessibility */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-lg focus:shadow-lg"
      >
        Skip to main content
      </a>
      <Header />
      <main id="main-content">
        <Hero />
        <Features />
        <HowItWorks />
      </main>
      <Footer />
    </div>
  )
}

export default LandingPage

