import React, { useState } from 'react';
import ExperimentResults from './components/ExperimentResults';
import './App.css';

function App() {
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
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ML Mini Project - Heart Disease Prediction</h1>
      </header>
      
      <div className="experiment-buttons" style={{ padding: '20px' }}>
        <button onClick={() => runExperiment('data_visualization')}>
          Data Visualization
        </button>
        <button onClick={() => runExperiment('svm')}>
          SVM
        </button>
        <button onClick={() => runExperiment('ensemble')}>
          Ensemble Learning
        </button>
        <button onClick={() => runExperiment('clustering')}>
          Clustering
        </button>
        <button onClick={() => runExperiment('pca')}>
          PCA Analysis
        </button>
        <button onClick={() => runExperiment('nonlinear_regression')}>
          Nonlinear Regression
        </button>
        <button onClick={() => runExperiment('polynomial_regression')}>
          Polynomial Regression
        </button>
      </div>
      
      {loading && <div>Loading...</div>}
      
      <ExperimentResults result={result} />
    </div>
  );
}

export default App;
