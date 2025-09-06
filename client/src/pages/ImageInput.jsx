import React, { useState, useRef } from "react";
import { Camera, Upload, Loader2, AlertCircle, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

export default function ImageInput() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [cropName, setCropName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleUpload = async () => {
    if (!imageFile || !cropName) {
      setError("Please select a crop and an image.");
      return;
    }
    setError(null);
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      formData.append("crop", cropName);

      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();

      console.log(data);
      navigate("/results", { state: { image: imagePreview, result: data } });
    } catch (err) {
      setError("Failed to process image. Try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Back Button */}
        <button
          onClick={() => navigate("/")}
          className="flex items-center space-x-2 text-green-600 hover:text-green-700 mb-6 font-medium"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Home</span>
        </button>

        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Plant Disease Detection
          </h1>
          <p className="text-lg text-gray-600">
            Upload or capture a plant image and select crop type
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 space-y-6">
          {/* Crop Dropdown */}
          <select
            value={cropName}
            onChange={(e) => setCropName(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 text-gray-700"
          >
            <option value="">Select Crop</option>
            <option value="Wheat">Wheat</option>
            <option value="Rice">Rice</option>
            <option value="Maize">Maize</option>
          </select>

          {/* File Upload */}
          <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-blue-300 rounded-xl hover:border-blue-400 hover:bg-blue-50 transition-all duration-200 cursor-pointer">
            <Upload className="w-16 h-16 text-blue-500 mb-4" />
            <h3 className="text-xl font-semibold text-gray-800 mb-2">
              Upload Image
            </h3>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* Camera Option (Mobile only) */}
          {isMobile && (
            <label className="flex flex-col items-center justify-center p-8 border-2 border-dashed border-green-300 rounded-xl hover:border-green-400 hover:bg-green-50 transition-all duration-200 cursor-pointer">
              <Camera className="w-16 h-16 text-green-500 mb-4" />
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                Use Camera
              </h3>
              <input
                type="file"
                accept="image/*"
                capture="environment"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          )}

          {/* Preview */}
          {imagePreview && (
            <div className="flex justify-center">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-64 h-auto rounded-lg border border-gray-300"
              />
            </div>
          )}

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition disabled:bg-gray-400"
          >

            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin" /> Processing...
              </span>
            ) : (
              "Upload & Predict"
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-700">{error}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
