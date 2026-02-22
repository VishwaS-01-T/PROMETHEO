import { motion } from 'framer-motion'
import React from 'react'

const Footer = () => {
  const currentYear = new Date().getFullYear()

  const footerLinks = {
    product: [
      { name: 'Features', href: '#features' },
      { name: 'Pricing', href: '#pricing' },
      { name: 'How It Works', href: '#learn' },
    ],
    company: [
      { name: 'About', href: '#about' },
      { name: 'Newsroom', href: '#newsroom' },
      { name: 'Careers', href: '#careers' },
    ],
    resources: [
      { name: 'Documentation', href: '#docs' },
      { name: 'Support', href: '#support' },
      { name: 'Community', href: '#community' },
    ],
  }

  return (
    <footer className="bg-gray-900 dark:bg-black text-gray-300 py-12 px-4 sm:px-6 lg:px-8">
      <div className="container mx-auto max-w-7xl">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-1">
            <motion.a
              href="#"
              className="text-2xl font-bold text-white mb-4 inline-block focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded"
              whileHover={{ scale: 1.05 }}
              aria-label="PROMETHEO"
            >
              PROMETHEO
            </motion.a>
            <p className="text-gray-400 text-sm">
              We don't only analyse your impact, We make it stronger.
            </p>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="text-white font-semibold mb-4 capitalize">
                {category}
              </h4>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-gray-400 hover:text-white transition-colors duration-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">
            © {currentYear} PROMETHEO. All rights reserved.
          </p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <a href="#privacy" className="text-gray-400 hover:text-white transition-colors duration-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded">
              Privacy
            </a>
            <a href="#terms" className="text-gray-400 hover:text-white transition-colors duration-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded">
              Terms
            </a>
            <a href="#cookies" className="text-gray-400 hover:text-white transition-colors duration-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 rounded">
              Cookies
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer

