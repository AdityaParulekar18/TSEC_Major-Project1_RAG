import React from 'react';
import { Mail, Users } from 'lucide-react';

const teamMembers = [
  {
    name: 'Aditya Parulekar',
    role: 'AI&DS Student',
    specialization: 'TSEC, Mumbai',
    description: 'Final year AI&DS student at Thadomal Shahani Engineering College, Mumbai.',
    email: 'aditya.parulekar@student.tsec.in',
    github_url: '#',
    linkedin_url: '#',
    image_url: '', // Optionally add a custom image URL here
  },
  {
    name: 'Harsh Patel',
    role: 'AI&DS Student',
    specialization: 'TSEC, Mumbai',
    description: 'Final year AI&DS student at Thadomal Shahani Engineering College, Mumbai.',
    email: 'harsh.patel@student.tsec.in',
    github_url: '#',
    linkedin_url: '#',
    image_url: '',
  },
  {
    name: 'Atharva Vichare',
    role: 'AI&DS Student',
    specialization: 'TSEC, Mumbai',
    description: 'Final year AI&DS student at Thadomal Shahani Engineering College, Mumbai.',
    email: 'atharva.vichare@student.tsec.in',
    github_url: '#',
    linkedin_url: '#',
    image_url: '',
  },
  {
    name: 'Sairaj Vinayagamoorthy',
    role: 'AI&DS Student',
    specialization: 'TSEC, Mumbai',
    description: 'Final year AI&DS student at Thadomal Shahani Engineering College, Mumbai.',
    email: 'sairaj.vinayagamoorthy@student.tsec.in',
    github_url: '#',
    linkedin_url: '#',
    image_url: '',
  },
];

const Team = () => {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <div className="flex items-center justify-center space-x-4 mb-6">
            <div className="bg-green-100 p-3 rounded-lg">
              <Users className="h-8 w-8 text-green-600" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-green-800">
              Meet Our Team
            </h1>
          </div>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            A dedicated group of students passionate about revolutionizing agriculture through AI technology
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8 mb-16">
          {teamMembers.map((member, index) => (
            <div key={index} className="bg-white rounded-2xl shadow-xl overflow-hidden hover:shadow-2xl transition-all transform hover:-translate-y-1">
              <div className="relative">
                <img
                  src={member.image_url || 'https://static.vecteezy.com/system/resources/thumbnails/000/439/863/small/Basic_Ui__28186_29.jpg'}
                  alt={member.name}
                  className="w-full h-64 object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                <div className="absolute bottom-4 left-4 text-white">
                  <h3 className="text-2xl font-bold">{member.name}</h3>
                  <p className="text-green-200">{member.role}</p>
                </div>
              </div>
              
              <div className="p-6">
                <div className="mb-4">
                  <span className="inline-block bg-green-100 text-green-800 text-sm font-medium px-3 py-1 rounded-full">
                    {member.specialization}
                  </span>
                </div>
                
                <p className="text-gray-600 leading-relaxed mb-6">
                  {member.description}
                </p>
                
                <div className="flex space-x-4">
                  {member.github_url && member.github_url !== '#' && (
                    <a
                      href={member.github_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-400 hover:text-green-600 transition-colors"
                      aria-label={`${member.name}'s GitHub`}
                    >
                      <Github className="h-5 w-5" />
                    </a>
                  )}
                  {member.linkedin_url && member.linkedin_url !== '#' && (
                    <a
                      href={member.linkedin_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-400 hover:text-green-600 transition-colors"
                      aria-label={`${member.name}'s LinkedIn`}
                    >
                      <Linkedin className="h-5 w-5" />
                    </a>
                  )}
                  {member.email && (
                    <a
                      href={`mailto:${member.email}`}
                      className="text-gray-400 hover:text-green-600 transition-colors"
                      aria-label={`Email ${member.name}`}
                    >
                      <Mail className="h-5 w-5" />
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Team;
