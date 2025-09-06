import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import ImageInput from './pages/ImageInput'
import Results from './pages/Results'
import About from './pages/About'
import Team from './pages/Team'
import Contact from './pages/Contact'

function App() {

  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-amber-50">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/input" element={<ImageInput />} />
            <Route path="/results" element={<Results />} />
            
            <Route path="/about" element={<About />} />
            <Route path="/team" element={<Team />} />
            <Route path="/contact" element={<Contact />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  )
}

export default App
