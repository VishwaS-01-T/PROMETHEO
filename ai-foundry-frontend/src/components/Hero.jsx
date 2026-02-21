import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

function Hero() {
  const heroRef = useRef(null)
  const bgHeadingRef = useRef(null)
  const fgTextRef = useRef(null)
  const circleback = useRef(null)
  const circleleft = useRef(null)
  const circleright = useRef(null)
  const fadeRef = useRef(null)
  const scrollPromptRef = useRef(null)

  useEffect(() => {
    const heroEl = heroRef.current
    const bgEl = bgHeadingRef.current
    const fgEl = fgTextRef.current
    const cback = circleback.current
    const cleft = circleleft.current
    const cright = circleright.current
    const fade = fadeRef.current
    const scrollPrompt = scrollPromptRef.current

    if (!heroEl || !bgEl || !fgEl || !cback || !cleft || !cright || !fade || !scrollPrompt) return

    // Scroll prompt appear after 2s
    const promptTl = gsap.timeline({ delay: 2 })
    promptTl.fromTo(
      scrollPrompt,
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
    )

    // Hide scroll prompt once user scrolls past 50px
    const handleScroll = () => {
      if (window.scrollY > 50) {
        gsap.to(scrollPrompt, { opacity: 0, duration: 0.4, ease: 'power2.in' })
      } else {
        gsap.to(scrollPrompt, { opacity: 1, duration: 0.4, ease: 'power2.out' })
      }
    }
    
    window.addEventListener('scroll', handleScroll)

    // Timeline controlled by scroll
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: heroEl,
        start: 'top top',
        end: '+=900vh',
        scrub: true,
      },
    })

    // Separate ScrollTrigger only for pinning
    const pinST = ScrollTrigger.create({
      trigger: heroEl,
      start: 'top top',
      end: '+=500vh',
      pin: true,
    })

    // --- First part: pinned animations ---
    tl.fromTo(
      bgEl,
      { y: 100, opacity: 0 },
      { y: 0, opacity: 1, ease: 'none', duration: 1 },
      0
    )

    tl.fromTo(
      fgEl,
      { y: 30, opacity: 1 },
      { y: 0, opacity: 1, ease: 'power1.inOut', duration: 1 },
      0
    )

    tl.to(
      cright,
      { width: 100, height: 100, x: '-30vw', y: '10vh', duration: 1 },
      0
    )
    tl.to(
      cback,
      { width: 200, height: 200, x: '-43vw', y: '20vh', duration: 1.8 },
      0
    )
    tl.to(
      cleft,
      { width: 450, height: 450, x: '57.25vw', y: '23vh', duration: 2 },
      0
    )

    tl.to(
      cright,
      { width: 200, height: 200, x: '-68vw', y: '35vh', duration: 2 },
      1
    )

    tl.to(
      cback,
      { width: 450, height: 450, x: '-14.7vw', y: '50vh', duration: 2 },
      1.6
    )

    tl.to(
      cright,
      { width: 450, height: 450, x: '-73.7vw', y: '65vh', duration: 1 },
      3
    )

    tl.to(fgEl, { opacity: 0, duration: 1 }, 2)

    // Fade overlay
    tl.fromTo(
      fade,
      { opacity: 0 },
      { opacity: 1, duration: 2, ease: 'power2.inOut' },
      0
    )

    // --- Second part: after unpin ---
    tl.to(
      cleft,
      {
        width: 460,
        height: 460,
        x: '57vw',
        y: '54.3vh',
        borderRadius: '1rem',
        duration: 1,
      },
      '>'
    )

    tl.to(
      cback,
      {
        width: 463,
        height: 463,
        x: '-15.1vw',
        y: '81.4vh',
        borderRadius: '1rem',
        duration: 1,
      },
      '<'
    )

    tl.to(
      cright,
      {
        width: 463,
        height: 463,
        x: '-74.2vw',
        y: '98vh',
        borderRadius: '1rem',
        duration: 1,
      },
      '<'
    )

    tl.to(
      cleft,
      {
        opacity: 0,
      },
      '>'
    )

    tl.to(
      cback,
      {
        opacity: 0,
      },
      '<'
    )

    tl.to(
      cright,
      {
        opacity: 0,
      },
      '<'
    )

    return () => {
      window.removeEventListener('scroll', handleScroll)
      if (tl.scrollTrigger) tl.scrollTrigger.kill()
      tl.kill()
      pinST.kill()
      promptTl.kill()
      ScrollTrigger.clearMatchMedia?.()
    }
  }, [])

  return (
    <section
      ref={heroRef}
      className="relative min-h-screen flex items-center justify-center pt-20 px-4 sm:px-6 lg:px-8"
    >
      {/* Background GIF */}
      <div
        className="absolute inset-0 w-full h-full z-0 opacity-20"
        style={{
          backgroundImage: 'url(/giphy.gif)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />

      {/* Bottom fade overlay */}
      <div
        ref={fadeRef}
        className="absolute bottom-0 left-0 right-0 h-[40vh] z-[1] pointer-events-none"
        style={{
          background: 'linear-gradient(to top, #ffffff 0%, transparent 100%)',
          opacity: 0,
        }}
      />

      <div
        ref={circleback}
        className="absolute pointer-events-none w-[100px] h-[100px] left-[50%] top-[30%] rounded-full z-[5]"
        style={{
          background:
            'radial-gradient(circle at 30% 30%, #F3E8FF, #C084FC 35%, #9333EA 65%, #7E22CE)',
        }}
      />

      <div
        ref={circleleft}
        className="absolute pointer-events-none w-[200px] h-[200px] left-[13%] top-[55%] rounded-full z-[15]"
        style={{
          background:
            'radial-gradient(circle at 30% 30%, #F3E8FF, #C084FC 35%, #9333EA 65%, #7E22CE)',
        }}
      />

      <div
        ref={circleright}
        className="absolute pointer-events-none w-[320px] h-[320px] left-[79%] top-[14%] rounded-full z-[2]"
        style={{
          background:
            'radial-gradient(circle at 35% 30%, #FAF5FF, #D8B4FE 30%, #A855F7 60%, #7C3AED)',
        }}
      />

      <div className="relative z-10 container mx-auto text-center">
        <div className="mb-10 relative">
          {/* Background big heading */}
          <h1
            ref={bgHeadingRef}
            style={{
              fontFamily: '"Bodoni Poster Compressed", serif',
              letterSpacing: '-0.02em',
            }}
            className="text-[10rem] md:text-[10rem] lg:text-[12rem] font-bold tracking-wide absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -z-10 bg-gradient-to-b from-[#FF9999] to-[#AA6677] bg-clip-text text-transparent"
          >
            PROMETHEO
          </h1>

          {/* Foreground text */}
          <motion.p
            ref={fgTextRef}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-[8rem] md:text-[8rem] lg:text-[8rem] text-gray-700 relative z-10"
            style={{ fontFamily: '"Beau Rivage", cursive' }}
          >
            Build your startup with
          </motion.p>
        </div>
      </div>

      {/* Scroll Prompt */}
      <div
        ref={scrollPromptRef}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 opacity-0"
      >
        <p className="text-gray-600 text-sm font-medium tracking-wide">
          SCROLL TO EXPLORE
        </p>
        <svg
          className="w-6 h-6 text-gray-600 animate-bounce"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
        </svg>
      </div>
    </section>
  )
}

export default Hero