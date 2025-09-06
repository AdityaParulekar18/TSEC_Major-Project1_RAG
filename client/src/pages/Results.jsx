import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, AlertTriangle, Leaf, Droplets, Clock, CheckCircle } from 'lucide-react';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const image = location.state?.image;
  const result = location.state?.result;

  const { crop, disease, confidence, description, cause, treatment, prevention } = result;

  const handleTryAnother = () => {
    navigate('/');
  };

  const handleBack = () => {
    navigate(-1);
  };

  if (!image) {
    navigate('/');
    return null;
  }

  return (
    <div className="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">

        <button
          onClick={handleBack}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 mb-6 font-medium"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back</span>
        </button>

        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">

          <div className="bg-gradient-to-r from-green-600 to-green-700 px-8 py-6">
            <h1 className="text-3xl font-bold text-white text-center">
              Analysis Results
            </h1>
            <p className="text-green-100 text-center mt-2">
              AI-powered crop disease detection complete
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 p-8">
            {/* Image Section */}
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-green-800">Analyzed Image</h2>
              <img
                src={image}
                alt="Analyzed crop"
                className="w-full h-80 object-cover rounded-xl shadow-lg"
              />
            </div>

            {/* Results Section */}
            <div className="space-y-6">
              <div className="bg-green-50 p-6 rounded-xl">
                <div className="flex items-center space-x-3 mb-4">
                  <Leaf className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-semibold text-green-800">Crop Identification</h3>
                </div>
                <p className="text-2xl font-bold text-green-700"> {crop}</p>
                <p className="text-green-600 mt-1">Confidence: 100%</p>
              </div>

              <div className="bg-red-50 p-6 rounded-xl border border-red-200">
                <div className="flex items-center space-x-3 mb-4">
                  <AlertTriangle className="h-6 w-6 text-red-600" />
                  <h3 className="text-lg font-semibold text-red-800">Detected Issue</h3>
                </div>
                <p className="text-2xl font-bold text-red-700"> {disease}</p>
                <p className="text-red-600 mt-1">Confidence: {confidence*100}%</p>
              </div>

            </div>

          </div>

          {/* Description and Treatment */}

          <div className='flex flex-col justify-center items-center gap-5'>

            <div className="bg-blue-50 p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-blue-800 mb-3">Description</h3>
              <p className="text-gray-700 leading-relaxed">
                {description}
              </p>
            </div>

            <div className="bg-pink-50 p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-pink-800 mb-3">Cause</h3>
              <p className="text-gray-700 leading-relaxed">
                {cause}
              </p>
            </div>

            <div className="bg-amber-50 p-6 rounded-xl w-full">
              <h3 className="text-lg font-semibold text-amber-800 mb-3">Treatment</h3>
              <p className="text-gray-700 leading-relaxed">
                {treatment}
              </p>
            </div>

            <div className="bg-green-50 p-6 rounded-xl">
              <h3 className="text-lg font-semibold text-green-800 mb-3">Prevention</h3>
              <p className="text-gray-700 leading-relaxed">
                {prevention}
              </p>
            </div>

          </div>

          <div className="bg-gray-50 px-8 py-6 text-center">
            <button
              onClick={handleTryAnother}
              className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 shadow-lg"
            >
              Try Another Image
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Results;