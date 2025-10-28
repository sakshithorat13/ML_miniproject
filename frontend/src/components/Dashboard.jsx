import React, { useState } from 'react';
import './Dashboard.css';

const Dashboard = ({ onGoHome }) => {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runExperiment = async (experimentType) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/experiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ experiment: experimentType }),
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

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div className="header-content">
          <button className="back-button" onClick={onGoHome}>
            ← Back to Home
          </button>
          <h1>ML Analytics Dashboard</h1>
          <p>Heart Disease Prediction Experiments</p>
        </div>
      </header>
      
      <div className="experiment-buttons">
        <button onClick={() => runExperiment('data_visualization')} className="exp-btn viz">
          📊 Data Visualization
        </button>
        <button onClick={() => runExperiment('svm')} className="exp-btn svm">
          🎯 SVM Analysis
        </button>
        <button onClick={() => runExperiment('ensemble')} className="exp-btn ensemble">
          🌳 Ensemble Learning
        </button>
        <button onClick={() => runExperiment('clustering')} className="exp-btn clustering">
          🔍 Clustering
        </button>
        <button onClick={() => runExperiment('pca')} className="exp-btn pca">
          📈 PCA Analysis
        </button>
        <button onClick={() => runExperiment('nonlinear_regression')} className="exp-btn nonlinear">
          📊 Nonlinear Regression
        </button>
        <button onClick={() => runExperiment('polynomial_regression')} className="exp-btn polynomial">
          🔬 Polynomial Regression
        </button>
      </div>
      
      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Running experiment...</p>
        </div>
      )}
      
      {/* Results section - same as your existing App.jsx */}
      {result && (
        <div className="experiment-results">
          {/* ... existing results display code ... */}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
