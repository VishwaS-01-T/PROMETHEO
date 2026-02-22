import { useEffect, useRef, useState } from "react";
import AnimatedBackground from "../components/AnimatedBackground";
import PromptInput from "../components/PromptInput";
import gsap from "gsap";
import React from "react";
import { Sun, Moon } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const navigate = useNavigate();

  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem("theme");
      if (saved) return saved;
    } catch {}
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    // Avoid opacity animations to prevent brief invisibility on first paint/StrictMode
    const tl = gsap.timeline({ defaults: { ease: "power3.out", overwrite: "auto" } });
    if (titleRef.current) {
      gsap.set(titleRef.current, { y: 24 });
      tl.to(titleRef.current, { y: 0, duration: 0.8 });
    }
    if (subtitleRef.current) {
      gsap.set(subtitleRef.current, { y: 16 });
      tl.to(subtitleRef.current, { y: 0, duration: 0.7 }, "-=0.4");
    }
    return () => tl.kill();
  }, []);

  useEffect(() => {
    if (theme === "dark") document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
    try { localStorage.setItem("theme", theme); } catch {}
  }, [theme]);

  const [running, setRunning] = useState(false);

  // Send prompt to backend planner endpoint and navigate to report page
  const sendPrompt = async (text) => {
    const promptText = text?.trim();
    if (!promptText) return;

    setRunning(true);

    try {
      // Run planner inference and IP geolocation in parallel
      const [planRes, geoRes] = await Promise.allSettled([
        fetch('http://localhost:8000/infer_plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ initial_prompt: promptText })
        }).then(r => r.json()),
        fetch('https://ipapi.co/json/').then(r => r.json())
      ]);

      const planData = planRes.status === 'fulfilled' && planRes.value.success ? planRes.value.plan : null;
      const geo = geoRes.status === 'fulfilled' ? geoRes.value : {};
      const userLocation = `${geo.country_name || ''}`.replace(/^,\s*|,\s*$/g, '').replace(/,\s*,/g, ', ').trim() || 'Unknown';

      if (!planData) {
        alert('Failed to infer business plan. Please try again.');
        setRunning(false);
        return;
      }

      // Navigate to report page with inferred plan (no agents started yet)
      navigate('/workflow', {
        state: {
          prompt: promptText,
          inferredPlan: { ...planData, location: userLocation },
          userLocation
        }
      });
    } catch (err) {
      console.error('sendPrompt error:', err);
      alert('Something went wrong. Please try again.');
      setRunning(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background">
      <AnimatedBackground theme={theme} />

      {/* Fixed Header */}
      <header className="fixed top-0 left-0 right-0 z-40 h-16 flex items-center justify-between px-6 backdrop-blur bg-white/60 dark:bg-black/40 border-b border-gray-200/50 dark:border-gray-800/50">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/')}
            className="font-semibold text-lg text-gray-900 dark:text-gray-100"
            title="Home"
          >
            PROMETHEO
          </button>
          <span className="hidden md:inline text-sm text-gray-600 dark:text-gray-300">— Build your startup</span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
            title="Toggle theme"
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/80 dark:bg-black/20 hover:opacity-95 transition"
          >
            {theme === "dark" ? <Sun className="w-4 h-4 text-yellow-500" /> : <Moon className="w-4 h-4 text-gray-700" />}
            <span className="hidden sm:inline text-sm text-gray-800 dark:text-gray-200">
              {theme === "dark" ? "Light" : "Dark"}
            </span>
          </button>
        </div>
      </header>

      {/* Main content with padding so fixed header/footer do not overlap */}
      <main className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 pt-24 pb-24">
        <div className="text-center mb-12 space-y-4 max-w-5xl w-full">
          <h1
            ref={titleRef}
            className={`text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight flex items-center justify-center flex-wrap ${
              theme === "light" ? "text-gray-900" : "text-foreground"
            }`}
          >
            <span>Lets Build</span>
            <span className="bg-linear-to-r from-red-100 via-accent to-pink-500 bg-clip-text text-transparent ml-3">
              Your Start-up
            </span>
          </h1>

          <p
            ref={subtitleRef}
            className={`text-lg md:text-xl max-w-2xl mx-auto ${
              theme === "light" ? "text-gray-600" : "text-muted-foreground"
            }`}
          >
            Lets initailise your startup and get ready to deploy
          </p>
        </div>

        <PromptInput onSubmit={sendPrompt} isRunning={running} />

        <div className="mt-8 text-sm text-muted-foreground/60">
          Press <kbd className="px-2 py-1 bg-secondary/50 rounded text-xs">Enter</kbd> to send or{" "}
          <kbd className="px-2 py-1 bg-secondary/50 rounded text-xs">Shift + Enter</kbd> for new line
        </div>
      </main>

      {/* Fixed Footer */}
      <footer className="fixed bottom-0 left-0 right-0 z-40 h-12 flex items-center justify-center text-sm backdrop-blur bg-white/60 dark:bg-black/40 border-t border-gray-200/50 dark:border-gray-800/50">
        <div className="text-muted-foreground/80">
          © {new Date().getFullYear()} PROMETHEO. All Rights Reserved.
        </div>
      </footer>
    </div>
  );
};

export default Index;