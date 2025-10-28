import React from 'react';

const ExperimentResults = ({ result }) => {
  if (!result) return null;

  return (
    <div className="experiment-results">
      <h2>{result.experiment}</h2>
      
      {/* Plot */}
      {result.plot && (
        <div className="plot-section">
          <img 
            src={`data:image/png;base64,${result.plot}`} 
            alt="Experiment Plot" 
            style={{ maxWidth: '100%', height: 'auto' }}
          />
        </div>
      )}
      
      {/* Metrics */}
      <div className="metrics-section">
        <h3>Metrics</h3>
        <pre>{JSON.stringify(result.metrics, null, 2)}</pre>
      </div>
      
      {/* Analysis Section - This is what's missing */}
      {result.analysis && (
        <div className="analysis-section" style={{ 
          backgroundColor: '#f5f5f5', 
          padding: '20px', 
          margin: '20px 0',
          borderRadius: '8px',
          border: '1px solid #ddd'
        }}>
          <h3 style={{ color: '#333', marginBottom: '15px' }}>
            📊 Analysis & Interpretation
          </h3>
          
          <div className="graph-interpretation" style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#555' }}>Graph Interpretation:</h4>
            <div style={{ 
              whiteSpace: 'pre-wrap', 
              backgroundColor: 'white',
              padding: '15px',
              borderRadius: '5px',
              lineHeight: '1.6'
            }}>
              {result.analysis.graph_interpretation}
            </div>
          </div>
          
          <div className="key-insights" style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#555' }}>🔍 Key Insights:</h4>
            <ul style={{ paddingLeft: '20px' }}>
              {result.analysis.key_inferences?.map((insight, index) => (
                <li key={index} style={{ marginBottom: '5px' }}>{insight}</li>
              ))}
            </ul>
          </div>
          
          <div className="practical-applications">
            <h4 style={{ color: '#555' }}>🏥 Practical Applications:</h4>
            <ul style={{ paddingLeft: '20px' }}>
              {result.analysis.practical_applications?.map((app, index) => (
                <li key={index} style={{ marginBottom: '5px' }}>{app}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      
      {/* Error Display */}
      {result.error && (
        <div className="error-section" style={{ 
          color: 'red', 
          backgroundColor: '#ffe6e6',
          padding: '10px',
          borderRadius: '5px' 
        }}>
          <h3>Error:</h3>
          <p>{result.error}</p>
        </div>
      )}
    </div>
  );
};

export default ExperimentResults;
