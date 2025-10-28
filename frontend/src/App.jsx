import React, { useState } from 'react';
import HomePage from './HomePage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState('');

  const goToDashboard = () => {
    setCurrentPage('dashboard');
  };

  const goToHome = () => {
    setCurrentPage('home');
    setResult(null);
    setSelectedExperiment('');
  };

  const runExperiment = async () => {
    if (!selectedExperiment) {
      alert('Please select an experiment first!');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/experiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ experiment: selectedExperiment }),
      });
      
      const data = await response.json();
      console.log('API Response:', data);
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: error.message });
    }
    setLoading(false);
  };

  const experiments = [
    { id: 'data_visualization', name: '📊 Data Visualization', description: 'Explore dataset patterns and correlations' },
    { id: 'svm', name: '🎯 SVM Analysis', description: 'Support Vector Machine classification' },
    { id: 'ensemble', name: '🌳 Ensemble Learning', description: 'Random Forest vs Gradient Boosting' },
    { id: 'clustering', name: '🔍 Clustering', description: 'K-Means patient segmentation' },
    { id: 'pca', name: '📈 PCA Analysis', description: 'Principal Component Analysis' },
    { id: 'nonlinear_regression', name: '📊 Nonlinear Regression', description: 'Complex relationship modeling' },
    { id: 'polynomial_regression', name: '🔬 Polynomial Regression', description: 'Higher-order feature interactions' }
  ];

  // Show HomePage first
  if (currentPage === 'home') {
    return <HomePage onGoToDashboard={goToDashboard} />;
  }

  // Dashboard page
  return (
    <div className="App">
      <header className="dashboard-header">
        <div className="header-content">
          <button className="back-button" onClick={goToHome}>
            ← Back to Home
          </button>
          <h1>ML Analytics Dashboard</h1>
          <p>Heart Disease Prediction Experiments</p>
        </div>
      </header>
      
      <div className="dashboard-layout">
        {/* Left Sidebar */}
        <div className="sidebar">
          <h3>Select Experiment</h3>
          <div className="experiment-list">
            {experiments.map((exp) => (
              <div 
                key={exp.id}
                className={`experiment-item ${selectedExperiment === exp.id ? 'selected' : ''}`}
                onClick={() => setSelectedExperiment(exp.id)}
              >
                <div className="exp-name">{exp.name}</div>
                <div className="exp-description">{exp.description}</div>
              </div>
            ))}
          </div>
          
          <div className="run-section">
            <button 
              className="run-button" 
              onClick={runExperiment}
              disabled={!selectedExperiment || loading}
            >
              {loading ? 'Running...' : 'Run Experiment'}
            </button>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="main-content">
          {loading && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Running {experiments.find(e => e.id === selectedExperiment)?.name}...</p>
            </div>
          )}
          
          {result && !loading && (
            <div className="experiment-results">
              <h2>{result.experiment}</h2>
              
              {result.plot && (
                <div className="plot-section">
                  <img 
                    src={`data:image/png;base64,${result.plot}`} 
                    alt="Experiment Plot" 
                    style={{ maxWidth: '100%', height: 'auto' }}
                  />
                </div>
              )}
              
              <div className="metrics-section">
                <h3>📊 Metrics</h3>
                <div className="metrics-grid">
                  {Object.entries(result.metrics || {}).map(([key, value]) => (
                    <div key={key} className="metric-item">
                      <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong> {
                        typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)
                      }
                    </div>
                  ))}
                </div>
              </div>
              
              {result.analysis && (
                <div className="analysis-section">
                  <h3>🔬 Analysis & Interpretation</h3>
                  
                  <div className="graph-interpretation">
                    <h4>📈 Graph Interpretation:</h4>
                    <div className="interpretation-content">
                      {result.analysis.graph_interpretation.split('\n').map((line, index) => (
                        <p key={index}>{line}</p>
                      ))}
                    </div>
                  </div>
                  
                  <div className="key-insights">
                    <h4>🔍 Key Insights:</h4>
                    <ul>
                      {result.analysis.key_inferences?.map((insight, index) => (
                        <li key={index}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="what-graph-shows">
                    <h4>📋 What the Graph Shows:</h4>
                    <p>{result.analysis.what_graph_shows}</p>
                  </div>
                  
                  <div className="why-graph-like-this">
                    <h4>🤔 Why the Graph Looks Like This:</h4>
                    <p>{result.analysis.why_graph_like_this}</p>
                  </div>
                  
                  <div className="practical-applications">
                    <h4>🏥 Practical Applications:</h4>
                    <ul>
                      {result.analysis.practical_applications?.map((app, index) => (
                        <li key={index}>{app}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              
              {result.error && (
                <div className="error-section">
                  <h3>❌ Error</h3>
                  <p>{result.error}</p>
                </div>
              )}
            </div>
          )}

          {!result && !loading && (
            <div className="welcome-message">
              <h2>Welcome to ML Analytics Dashboard</h2>
              <p>Select an experiment from the left sidebar and click "Run Experiment" to get started.</p>
              <div className="dataset-info">
                <h3>Dataset Information</h3>
                <p>Using Heart Disease Prediction dataset with 270+ patient records and 13 key features including Age, Blood Pressure, Cholesterol, and more.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
