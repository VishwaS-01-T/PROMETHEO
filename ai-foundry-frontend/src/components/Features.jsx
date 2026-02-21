import { motion } from 'framer-motion'
import { Brain, Workflow, Globe, Share2, Zap, Target } from 'lucide-react'
import FeatureCard from './FeatureCard'
import React from 'react'

const Features = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI Breakdown',
      description: 'Intelligent analysis that transforms your big idea into structured, actionable steps',
    },
    {
      icon: Workflow,
      title: 'Smart Workflows',
      description: 'Automatically generated project roadmaps with timelines and milestones',
    },
    {
      icon: Globe,
      title: 'Website Generation',
      description: 'Professional websites created instantly, tailored to your startup\'s needs',
    },
    {
      icon: Share2,
      title: 'Social Media Automation',
      description: 'Auto-post engaging content to Instagram and Twitter with optimal timing',
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'From idea to execution in minutes, not months',
    },
    {
      icon: Target,
      title: 'Generate BRD/PRD',
      description: 'generates a BRD/PRD (bussiness requirements document/product requirements document) for your startup',
    },
  ]

  return (
    <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-gray-900/50">
      <div className="container mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Everything You Need to Launch
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            From concept to launch, our AI handles every step of your startup journey.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.title}
              icon={feature.icon}
              title={feature.title}
              description={feature.description}
              delay={index * 0.1}
            />
          ))}
        </div>
      </div>
    </section>
  )
}

export default Features

