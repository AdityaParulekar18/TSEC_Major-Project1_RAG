import React from 'react';
import { Heart, Github, Linkedin, Mail } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-green-800 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="text-center md:text-left mb-4 md:mb-0">
            <p className="flex items-center justify-center md:justify-start">
              Developed with <Heart className="h-4 w-4 mx-1 text-red-400" /> by Aditya, Harsh, Atharva & Sairaj
            </p>
            <p className="text-green-200 text-sm mt-1">Final Year Project 2025</p>
          </div>
          
          <div className="flex space-x-6">
            <a
              href="https://github.com/AdityaParulekar18"
              target="_blank"
              rel="noopener noreferrer"
              className="text-green-200 hover:text-white transition-colors"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
            <a
              href="https://linkedin.com/in/AdityaParulekar"
              target="_blank"
              rel="noopener noreferrer"
              className="text-green-200 hover:text-white transition-colors"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </a>
            <a
              href="mailto:adit1809pro@gmail.com"
              className="text-green-200 hover:text-white transition-colors"
              aria-label="Email"
            >
              <Mail className="h-5 w-5" />
            </a>
          </div>
        </div>
        
        <div className="border-t border-green-700 mt-6 pt-6 text-center">
          <p className="text-green-200 text-sm">
            © 2025 CropAI Disease Detection. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;