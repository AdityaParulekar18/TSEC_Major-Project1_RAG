import React, { useState } from 'react';
import { Mail, Phone, MapPin, Send, MessageSquare } from 'lucide-react';
// import { useContactForm } from '../hooks/useContactForm';

const Contact = () => {

  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });

  const [isSubmitted, setIsSubmitted] = useState(false);
  // const { submitMessage, submitting, error } = useContactForm();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    setIsSubmitted(true);
    // e.preventDefault();
    
    // const success = await submitMessage(formData);
    
    // if (success) {
    //   setIsSubmitted(true);
    //   setTimeout(() => {
    //     setIsSubmitted(false);
    //     setFormData({ name: '', email: '', message: '' });
    //   }, 3000);
    // }
  };

  const whatsappNumber = '+918369561904';
  const whatsappMessage = 'Hi! I would like to know more about the CropAI Disease Detection project.';
  const whatsappUrl = `https://wa.me/${whatsappNumber}?text=${encodeURIComponent(whatsappMessage)}`;

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-6">
            Get In Touch
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Have questions about our project? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className="bg-green-100 p-3 rounded-lg">
                <MessageSquare className="h-6 w-6 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-green-800">Send us a Message</h2>
            </div>

            {isSubmitted ? (
              <div className="text-center py-8">
                <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                  <Send className="h-8 w-8 text-green-600" />
                </div>
                <h3 className="text-xl font-semibold text-green-800 mb-2">Message Sent!</h3>
                <p className="text-gray-600">Thank you for your message. We'll get back to you soon.</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                    Full Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-colors"
                    placeholder="Enter your full name"
                  />
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                    Email Address
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-colors"
                    placeholder="Enter your email address"
                  />
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
                    Message
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    required
                    rows={5}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-colors resize-none"
                    placeholder="Tell us about your inquiry or feedback"
                  />
                </div>

                <button
                  type="submit"
                  // disabled={submitting}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition-all transform hover:scale-105 shadow-lg flex items-center justify-center space-x-2"
                >
                  <Send className="h-5 w-5" />
                  {/* <span>{submitting ? 'Sending...' : 'Send Message'}</span> */}
                </button>
                
                {/* {error && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-600 text-sm">Error: {error}</p>
                  </div>
                )} */}
              </form>
            )}
          </div>

          {/* Contact Information */}
          <div className="space-y-8">
            {/* Contact Details */}
            <div className="bg-white rounded-2xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-green-800 mb-6">Contact Information</h2>
              
              <div className="space-y-6">
                <div className="flex items-center space-x-4">
                  <div className="bg-blue-100 p-3 rounded-lg">
                    <a 
                      href="mailto:adit1809pro@gmail.com"
                      className="block hover:scale-110 transition-transform"
                      aria-label="Send email"
                    >
                      <Mail className="h-6 w-6 text-blue-600" />
                    </a>
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">Email</h3>
                    <a 
                      href="mailto:adit1809pro@gmail.com" 
                      className="text-gray-600 hover:text-blue-600 transition-colors cursor-pointer"
                    >
                      adit1809pro@gmail.com
                    </a>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="bg-green-100 p-3 rounded-lg">
                    <a 
                      href="tel:+918369561904"
                      className="block hover:scale-110 transition-transform"
                      aria-label="Make phone call"
                    >
                      <Phone className="h-6 w-6 text-green-600" />
                    </a>
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">Phone</h3>
                    <a 
                      href="tel:+918369561904" 
                      className="text-gray-600 hover:text-green-600 transition-colors cursor-pointer"
                    >
                      +91 8369561904
                    </a>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="bg-amber-100 p-3 rounded-lg">
                    <a 
                      href="https://maps.google.com/?q=THADOMAL+SHAHANI+ENGINEERING+COLLEGE+LINKING+ROAD+BANDRA+WEST" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="block hover:scale-110 transition-transform"
                      aria-label="Open location in Google Maps"
                    >
                      <MapPin className="h-6 w-6 text-amber-600" />
                    </a>
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800">Location</h3>
                    <a 
                      href="https://maps.google.com/?q=THADOMAL+SHAHANI+ENGINEERING+COLLEGE+LINKING+ROAD+BANDRA+WEST" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-amber-600 transition-colors cursor-pointer"
                    >
                      AI&DS Department<br />TSEC, Mumbai University, Bandra, Mumbai, India
                    </a>
                  </div>
                </div>
              </div>
            </div>

            {/* WhatsApp Contact */}
            <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-2xl shadow-xl p-8 text-white">
              <h2 className="text-2xl font-bold mb-4">Quick Contact</h2>
              <p className="mb-6 text-green-100">
                Get instant responses to your questions via WhatsApp
              </p>
              
              <a
                href={whatsappUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center space-x-3 bg-white text-green-600 font-semibold py-3 px-6 rounded-lg hover:bg-green-50 transition-colors transform hover:scale-105 shadow-lg"
              >
                <MessageSquare className="h-5 w-5" />
                <span>Contact via WhatsApp</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;