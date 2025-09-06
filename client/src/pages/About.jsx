import React from 'react';
import { Brain, Database, Camera, Zap, Target } from 'lucide-react';

const About = () => {

  const technologies = [
    {
      icon: <Brain className="h-8 w-8" />,
      name: 'Convolutional Neural Networks (CNN)',
      description: 'Deep learning models trained to identify crop diseases and pests from images'
    },
    {
      icon: <Database className="h-8 w-8" />,
      name: 'Retrieval-Augmented Generation (RAG)',
      description: 'AI system that retrieves relevant information and generates contextual responses'
    },
    {
      icon: <Camera className="h-8 w-8" />,
      name: 'Image Processing',
      description: 'Advanced computer vision techniques for analyzing crop images'
    },
    {
      icon: <Zap className="h-8 w-8" />,
      name: 'Generative AI',
      description: 'Large language models providing intelligent treatment recommendations'
    }
  ];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-6">
            About the Project
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Revolutionizing agriculture through AI-powered early detection of crop diseases and pests
          </p>
        </div>

        {/* Project Goal */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-12">
          <div className="flex items-center space-x-4 mb-6">
            <div className="bg-green-100 p-3 rounded-lg">
              <Target className="h-8 w-8 text-green-600" />
            </div>
            <h2 className="text-3xl font-bold text-green-800">Project Goal</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div className="space-y-4">
              <p className="text-lg text-gray-700 leading-relaxed">
                Our project aims to develop an intelligent system that helps farmers and agronomists 
                identify crop diseases and pests at an early stage. By leveraging advanced AI technologies, 
                we provide accurate diagnosis and actionable treatment recommendations.
              </p>
              
              <p className="text-lg text-gray-700 leading-relaxed">
                The system combines computer vision for image analysis with retrieval-augmented generation 
                for providing contextual information about detected issues, treatment options, and preventive measures.
              </p>
              
              <div className="bg-amber-50 p-4 rounded-lg border-l-4 border-amber-500">
                <p className="text-amber-800 font-medium">
                  Early detection can reduce crop losses by up to 40% and improve overall yield quality.
                </p>
              </div>
            </div>
            
            <div className="bg-green-50 p-6 rounded-xl">
              <h3 className="text-xl font-semibold text-green-800 mb-4">Key Benefits</h3>
              <ul className="space-y-3">
                <li className="flex items-start space-x-3">
                  <div className="bg-green-500 rounded-full w-2 h-2 mt-2"></div>
                  <span className="text-gray-700">Faster disease identification</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="bg-green-500 rounded-full w-2 h-2 mt-2"></div>
                  <span className="text-gray-700">Reduced crop losses</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="bg-green-500 rounded-full w-2 h-2 mt-2"></div>
                  <span className="text-gray-700">Optimized treatment strategies</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="bg-green-500 rounded-full w-2 h-2 mt-2"></div>
                  <span className="text-gray-700">Improved agricultural productivity</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Technologies Used */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-green-800 text-center mb-12">
            Technologies Used
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {technologies.map((tech, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="bg-green-100 p-3 rounded-lg">
                    <div className="text-green-600">
                      {tech.icon}
                    </div>
                  </div>
                  <h3 className="text-xl font-semibold text-green-800">
                    {tech.name}
                  </h3>
                </div>
                <p className="text-gray-600 leading-relaxed">
                  {tech.description}
                </p>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default About;