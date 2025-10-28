import React from 'react';

const HomePage = ({ onGoToDashboard }) => {
  return (
    <div style={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      color: 'white',
      margin: 0,
      padding: 0
    }}>
      {/* Hero Section */}
      <div style={{ padding: '80px 20px', textAlign: 'center' }}>
        <h1 style={{ 
          fontSize: '3.5rem', 
          fontWeight: '700', 
          marginBottom: '20px',
          textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
        }}>
          Heart Disease Prediction System
        </h1>
        <p style={{ 
          fontSize: '1.3rem', 
          marginBottom: '50px',
          opacity: '0.9',
          fontWeight: '300'
        }}>
          Advanced Machine Learning Analytics for Cardiovascular Risk Assessment
        </p>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          gap: '60px',
          marginTop: '50px',
          flexWrap: 'wrap'
        }}>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '10px', color: '#ffd700' }}>270+</h3>
            <p style={{ fontSize: '1.1rem', opacity: '0.8' }}>Patient Records</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '10px', color: '#ffd700' }}>13</h3>
            <p style={{ fontSize: '1.1rem', opacity: '0.8' }}>Key Features</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <h3 style={{ fontSize: '2.5rem', fontWeight: '700', marginBottom: '10px', color: '#ffd700' }}>7</h3>
            <p style={{ fontSize: '1.1rem', opacity: '0.8' }}>ML Algorithms</p>
          </div>
        </div>
      </div>

      {/* Dataset Information */}
      <div style={{ background: 'white', color: '#333', padding: '80px 20px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          <h2 style={{ 
            textAlign: 'center', 
            fontSize: '2.5rem', 
            fontWeight: '700', 
            marginBottom: '50px',
            color: '#2c3e50'
          }}>
            About Our Heart Disease Dataset
          </h2>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
            gap: '30px',
            marginBottom: '60px'
          }}>
            <div style={{ 
              background: '#f8f9fa', 
              padding: '30px', 
              borderRadius: '15px',
              textAlign: 'center',
              boxShadow: '0 5px 15px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>🏥</div>
              <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '15px', color: '#2c3e50' }}>
                What is Heart Disease?
              </h3>
              <p style={{ lineHeight: '1.6', color: '#666' }}>
                Heart disease refers to various conditions affecting the heart's structure and function. 
                It's the leading cause of death globally, making early detection crucial for prevention 
                and treatment.
              </p>
            </div>

            <div style={{ 
              background: '#f8f9fa', 
              padding: '30px', 
              borderRadius: '15px',
              textAlign: 'center',
              boxShadow: '0 5px 15px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>📊</div>
              <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '15px', color: '#2c3e50' }}>
                Why This Dataset?
              </h3>
              <p style={{ lineHeight: '1.6', color: '#666' }}>
                Our dataset contains comprehensive patient information including age, blood pressure, 
                cholesterol levels, and other vital indicators. This rich data enables accurate 
                risk assessment and prediction modeling.
              </p>
            </div>

            <div style={{ 
              background: '#f8f9fa', 
              padding: '30px', 
              borderRadius: '15px',
              textAlign: 'center',
              boxShadow: '0 5px 15px rgba(0,0,0,0.1)'
            }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>🎯</div>
              <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '15px', color: '#2c3e50' }}>
                What It Indicates
              </h3>
              <p style={{ lineHeight: '1.6', color: '#666' }}>
                The dataset reveals patterns between lifestyle factors, medical indicators, and 
                heart disease occurrence. It helps identify high-risk patients before symptoms appear, 
                enabling preventive interventions.
              </p>
            </div>
          </div>

          {/* Benefits Section */}
          <div style={{ marginBottom: '60px' }}>
            <h3 style={{ textAlign: 'center', fontSize: '2rem', marginBottom: '40px', color: '#2c3e50' }}>
              How This Helps Healthcare
            </h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '25px'
            }}>
              <div style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: '20px',
                background: '#f0f8ff',
                padding: '25px',
                borderRadius: '12px',
                borderLeft: '5px solid #4CAF50'
              }}>
                <div style={{ fontSize: '2.5rem', minWidth: '60px' }}>🔍</div>
                <div>
                  <strong style={{ display: 'block', color: '#2e7d32', marginBottom: '8px', fontSize: '1.1rem' }}>
                    Early Detection
                  </strong>
                  <p style={{ color: '#666', margin: 0, lineHeight: '1.5' }}>
                    Identify at-risk patients before symptoms develop
                  </p>
                </div>
              </div>

              <div style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: '20px',
                background: '#f0f8ff',
                padding: '25px',
                borderRadius: '12px',
                borderLeft: '5px solid #4CAF50'
              }}>
                <div style={{ fontSize: '2.5rem', minWidth: '60px' }}>⚡</div>
                <div>
                  <strong style={{ display: 'block', color: '#2e7d32', marginBottom: '8px', fontSize: '1.1rem' }}>
                    Quick Screening
                  </strong>
                  <p style={{ color: '#666', margin: 0, lineHeight: '1.5' }}>
                    Rapid risk assessment in clinical settings
                  </p>
                </div>
              </div>

              <div style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: '20px',
                background: '#f0f8ff',
                padding: '25px',
                borderRadius: '12px',
                borderLeft: '5px solid #4CAF50'
              }}>
                <div style={{ fontSize: '2.5rem', minWidth: '60px' }}>💡</div>
                <div>
                  <strong style={{ display: 'block', color: '#2e7d32', marginBottom: '8px', fontSize: '1.1rem' }}>
                    Treatment Planning
                  </strong>
                  <p style={{ color: '#666', margin: 0, lineHeight: '1.5' }}>
                    Personalized care strategies based on risk profiles
                  </p>
                </div>
              </div>

              <div style={{ 
                display: 'flex', 
                alignItems: 'flex-start', 
                gap: '20px',
                background: '#f0f8ff',
                padding: '25px',
                borderRadius: '12px',
                borderLeft: '5px solid #4CAF50'
              }}>
                <div style={{ fontSize: '2.5rem', minWidth: '60px' }}>💰</div>
                <div>
                  <strong style={{ display: 'block', color: '#2e7d32', marginBottom: '8px', fontSize: '1.1rem' }}>
                    Cost Reduction
                  </strong>
                  <p style={{ color: '#666', margin: 0, lineHeight: '1.5' }}>
                    Prevent expensive emergency interventions
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <div style={{ 
            textAlign: 'center',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            padding: '60px',
            borderRadius: '20px',
            color: 'white',
            marginTop: '60px'
          }}>
            <h2 style={{ fontSize: '2.2rem', marginBottom: '20px', fontWeight: '600' }}>
              🚀 Explore Our ML Analytics Dashboard
            </h2>
            <p style={{ 
              fontSize: '1.1rem', 
              marginBottom: '40px',
              opacity: '0.9',
              maxWidth: '600px',
              marginLeft: 'auto',
              marginRight: 'auto'
            }}>
              Discover powerful machine learning algorithms including SVM, Random Forest, 
              Clustering, PCA, and more. Each algorithm provides unique insights into 
              heart disease prediction patterns.
            </p>
            <button 
              onClick={onGoToDashboard}
              style={{
                background: '#ffd700',
                color: '#333',
                padding: '18px 40px',
                border: 'none',
                borderRadius: '50px',
                fontSize: '1.2rem',
                fontWeight: '600',
                cursor: 'pointer',
                boxShadow: '0 5px 15px rgba(255, 215, 0, 0.3)',
                transition: 'all 0.3s ease'
              }}
            >
              Go to Dashboard →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
