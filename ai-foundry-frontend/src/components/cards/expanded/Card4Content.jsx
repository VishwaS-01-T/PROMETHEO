import { motion } from 'framer-motion'
import { useState } from 'react'

function Card4Content() {
  const [formData, setFormData] = useState({ phone: '', email: '' })

  const handleInputChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value })

  const handleCallAgent = () => {
    if (!formData.phone) { alert('Please enter your phone number first!'); return }
    alert(`Calling agent at ${formData.phone}... (simulated)`)
  }

  const handleSendEmail = () => {
    if (!formData.email) { alert('Please enter your email address first!'); return }
    alert(`Sending email to ${formData.email}... (simulated)`)
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="space-y-6 max-w-2xl mx-auto">
      <h2 className="text-4xl font-bold text-white mb-6 text-center">Call / Email agent for marketing</h2>

      <div className="bg-black/40 rounded-2xl p-8 border border-white/10 shadow-2xl">
        <p className="text-white/80 text-center mb-8">Fill in your contact information and our marketing agent will reach out to you!</p>

        <div className="space-y-6">
          <div>
            <label className="block text-white font-semibold mb-2">Phone Number</label>
            <input type="tel" name="phone" value={formData.phone} onChange={handleInputChange} placeholder="+1 (555) 123-4567" className="w-full px-4 py-3 bg-gray-800 border border-white/20 rounded-xl text-white" />
          </div>

          <div>
            <label className="block text-white font-semibold mb-2">Email Address</label>
            <input type="email" name="email" value={formData.email} onChange={handleInputChange} placeholder="you@example.com" className="w-full px-4 py-3 bg-gray-800 border border-white/20 rounded-xl text-white" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
            <button onClick={handleCallAgent} className="py-4 px-6 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold rounded-xl">Call Agent</button>
            <button onClick={handleSendEmail} className="py-4 px-6 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-bold rounded-xl">Send Email</button>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default Card4Content
