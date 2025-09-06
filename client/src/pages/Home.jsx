import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Camera, Brain, Database, CheckCircle } from 'lucide-react';

const Home = () => {
  const navigate = useNavigate();

  const steps = [
    {
      icon: <Upload className="h-8 w-8" />,
      title: 'Upload or Capture',
      description: 'Upload crop image or use camera'
    },
    {
      icon: <Brain className="h-8 w-8" />,
      title: 'Model Analyzes',
      description: 'CNN model processes the image'
    },
    {
      icon: <Database className="h-8 w-8" />,
      title: 'AI Retrieves Info',
      description: 'RAG system finds relevant data'
    },
    {
      icon: <CheckCircle className="h-8 w-8" />,
      title: 'Get Results',
      description: 'Receive diagnosis & treatment'
    }
  ];

  const handleUpload = () => {
    navigate('/input?type=upload');
  };

  const handleCamera = () => {
    navigate('/input?type=camera');
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-green-800 mb-6">
            RAG-Enabled Early Disease & Pest Detection in Crops
          </h1>
          <p className="text-xl md:text-2xl text-green-600 mb-12 max-w-4xl mx-auto">
            Detect crop diseases & pests early using AI and get instant solutions
          </p>
          
          <div className="flex flex-col sm:flex-row gap-6 justify-center max-w-2xl mx-auto">
            <button
              onClick={handleUpload}
              className="flex items-center justify-center space-x-3 bg-green-600 hover:bg-green-700 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              <Upload className="h-6 w-6" />
              <span>Upload Crop Image</span>
            </button>
            
            <button
              onClick={handleCamera}
              className="flex items-center justify-center space-x-3 bg-amber-600 hover:bg-amber-700 text-white px-8 py-4 rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg"
            >
              <Camera className="h-6 w-6" />
              <span>Use Camera</span>
            </button>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-green-800 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our AI-powered system uses advanced machine learning to identify crop diseases and provide actionable insights
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="text-center group">
                <div className="bg-green-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4 group-hover:bg-green-200 transition-colors">
                  <div className="text-green-600">
                    {step.icon}
                  </div>
                </div>
                <h3 className="text-xl font-semibold text-green-800 mb-2">
                  {step.title}
                </h3>
                <p className="text-gray-600">
                  {step.description}
                </p>
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-10 left-full w-8 h-0.5 bg-green-300 transform -translate-y-1/2"></div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-green-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-green-800 mb-4">
              Why Choose CropAI?
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
              <div className="bg-green-100 rounded-lg w-12 h-12 flex items-center justify-center mb-4">
                <Brain className="h-6 w-6 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                AI-Powered Detection
              </h3>
              <p className="text-gray-600">
                Advanced CNN models trained on thousands of crop images for accurate disease identification
              </p>
            </div>
            
            <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
              <div className="bg-amber-100 rounded-lg w-12 h-12 flex items-center justify-center mb-4">
                <Database className="h-6 w-6 text-amber-600" />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                RAG Technology
              </h3>
              <p className="text-gray-600">
                Retrieval-Augmented Generation provides contextual treatment recommendations
              </p>
            </div>
            
            <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
              <div className="bg-blue-100 rounded-lg w-12 h-12 flex items-center justify-center mb-4">
                <CheckCircle className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-green-800 mb-3">
                Instant Results
              </h3>
              <p className="text-gray-600">
                Get immediate diagnosis and treatment suggestions to save your crops
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;