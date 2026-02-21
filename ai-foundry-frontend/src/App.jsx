import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/home'
import Report from './pages/report'
import Control from './pages/control'
import WebEditor from './pages/webEditor'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/report" element={<Report />} />
        <Route path="/control" element={<Control />} />
        <Route path="/web-editor" element={<WebEditor />} />
      </Routes>
    </BrowserRouter>
  )
}
