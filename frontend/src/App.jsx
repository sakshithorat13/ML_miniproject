






// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   const [selectedExperiment, setSelectedExperiment] = useState('');
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');

//   const experiments = [
//     { value: 'data_visualization', label: '📊 Data Visualization', description: 'Explore correlations and data distribution' },
//     { value: 'linear_regression', label: '📈 Linear Regression', description: 'Predict continuous health metrics' },
//     { value: 'classification', label: '🌳 Decision Tree Classification', description: 'Classify heart disease risk' },
//     { value: 'svm', label: '🔍 Support Vector Machine', description: 'Advanced classification analysis' },
//     { value: 'ensemble', label: '🌲 Ensemble Learning', description: 'Random Forest & Gradient Boosting' },
//     { value: 'nonlinear_regression', label: '📏 Polynomial Regression', description: 'Non-linear relationship modeling' },
//     { value: 'clustering', label: '🎯 Clustering Analysis', description: 'Discover patient groups' },
//     { value: 'pca', label: '📐 PCA Analysis', description: 'Dimensionality reduction' }
//   ];

//   const runExperiment = async () => {
//     if (!selectedExperiment) return;
    
//     setLoading(true);
//     setError('');
//     setResult(null);
    
//     try {
//       const response = await axios.post('http://localhost:5000/experiment', {
//         experiment: selectedExperiment
//       });
//       setResult(response.data);
//     } catch (err) {
//       setError(err.response?.data?.error || 'Failed to run experiment');
//     } finally {
//       setLoading(false);
//     }
//   };

//   const formatMetricValue = (value) => {
//     if (typeof value === 'number') return value.toFixed(4);
//     if (Array.isArray(value)) return JSON.stringify(value, null, 2);
//     if (typeof value === 'object') return JSON.stringify(value, null, 2);
//     return String(value);
//   };

//   return (
//     <div className="app">
//       <header className="header">
//         <div className="header-content">
//           <h1 className="title">🏥 Healthcare ML Analytics Dashboard</h1>
//           <p className="subtitle">Interactive Machine Learning Experiments on Healthcare Data</p>
//         </div>
//       </header>

//       <div className="container">
//         <aside className="sidebar">
//           <div className="sidebar-content">
//             <h2 className="sidebar-title">ML Experiments</h2>
            
//             <div className="experiment-list">
//               {experiments.map((exp) => (
//                 <div
//                   key={exp.value}
//                   className={`experiment-card ${selectedExperiment === exp.value ? 'selected' : ''}`}
//                   onClick={() => setSelectedExperiment(exp.value)}
//                 >
//                   <div className="experiment-label">{exp.label}</div>
//                   <div className="experiment-description">{exp.description}</div>
//                 </div>
//               ))}
//             </div>
            
//             <button
//               className={`run-button ${!selectedExperiment || loading ? 'disabled' : ''}`}
//               onClick={runExperiment}
//               disabled={!selectedExperiment || loading}
//             >
//               {loading ? '🔄 Running...' : '▶️ Run Experiment'}
//             </button>

//             {selectedExperiment && (
//               <div className="selected-info">
//                 <h3>Selected:</h3>
//                 <p>{experiments.find(exp => exp.value === selectedExperiment)?.label}</p>
//               </div>
//             )}
//           </div>
//         </aside>

//         <main className="main-content">
//           {loading && (
//             <div className="loading-container">
//               <div className="loading-spinner"><div className="spinner"></div></div>
//               <div className="loading-text">
//                 <h3>🧠 Running ML Experiment...</h3>
//                 <p>Processing healthcare data and generating insights</p>
//               </div>
//             </div>
//           )}

//           {error && (
//             <div className="error-container">
//               <div className="error-content">
//                 <h3>⚠️ Error</h3>
//                 <p>{error}</p>
//                 <button onClick={() => setError('')} className="error-dismiss">Dismiss</button>
//               </div>
//             </div>
//           )}

//           {result && !loading && (
//             <div className="results-container">
//               <div className="results-header">
//                 <h2 className="results-title">{result.experiment}</h2>
//                 <div className="results-timestamp">⏰ Completed: {new Date().toLocaleString()}</div>
//               </div>

//               <div className="results-content">
//                 {/* Metrics */}
//                 {result.metrics && (
//                   <div className="metrics-section">
//                     <h3 className="section-title">📊 Experiment Metrics</h3>
//                     <div className="metrics-grid">
//                       {Object.entries(result.metrics).map(([key, value]) => (
//                         <div key={key} className="metric-card">
//                           <div className="metric-label">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
//                           <div className="metric-value">{formatMetricValue(value)}</div>
//                         </div>
//                       ))}
//                     </div>
//                   </div>
//                 )}

//                 {/* Plots */}
//                 {(result.plot || result.plots) && (
//                   <div className="visualization-section">
//                     <h3 className="section-title">📈 Visualization Results</h3>

//                     {/* Single plot */}
//                     {result.plot && (
//                       <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
//                         <img
//                           src={`data:image/png;base64,${result.plot}`}
//                           alt="Experiment Plot"
//                           style={{ maxWidth: '100%', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}
//                         />
//                       </div>
//                     )}

//                     {/* Multiple plots */}
//                     {result.plots && (
//                       selectedExperiment === 'data_visualization' ? (
//                         <div className="visualization-grid">
//                           {Object.entries(result.plots).map(([key, val]) => (
//                             <div key={key} className="visualization-item">
//                               <h4>{key.replace(/_/g, ' ').toUpperCase()}</h4>
//                               <img src={`data:image/png;base64,${val}`} alt={key} />
//                             </div>
//                           ))}
//                         </div>
//                       ) : (
//                         Object.entries(result.plots).map(([key, val]) => (
//                           <div key={key} style={{ marginBottom: '3rem', textAlign: 'center' }}>
//                             <h4>{key.replace(/_/g, ' ').toUpperCase()}</h4>
//                             <img
//                               src={`data:image/png;base64,${val}`}
//                               alt={key}
//                               style={{ maxWidth: '100%', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}
//                             />
//                           </div>
//                         ))
//                       )
//                     )}
//                   </div>
//                 )}

//                 {/* Insights Table */}
//                 {result.insights_table && result.insights_table.length > 0 && (
//                   <div className="insights-section" style={{ marginTop: '2rem' }}>
//                     <h3 className="section-title">🧩 Insights and Interpretation</h3>
//                     <table className="insights-table">
//                       <thead>
//                         <tr>
//                           <th>Graph</th>
//                           <th>Insight</th>
//                         </tr>
//                       </thead>
//                      <tbody>
//   {result.insights_table.map((item, index) => (
//     <tr key={index}>
//       <td>{item.Graph}</td>
//       <td dangerouslySetInnerHTML={{ __html: item.Insight }}></td>
//     </tr>
//   ))}
// </tbody>

//                     </table>
//                   </div>
//                 )}
//               </div>
//             </div>
//           )}

//           {!result && !loading && !error && (
//             <div className="welcome-container">
//               <div className="welcome-content">
//                 <h2>🎯 Welcome to Healthcare ML Dashboard</h2>
//                 <p>Select an experiment from the sidebar to begin your machine learning analysis</p>
                
//                 <div className="features-grid">
//                   <div className="feature-card"><h3>📊 Data Analysis</h3><p>Explore correlations and statistical insights</p></div>
//                   <div className="feature-card"><h3>🤖 ML Models</h3><p>Train and evaluate various algorithms</p></div>
//                   <div className="feature-card"><h3>📈 Visualizations</h3><p>Interactive charts and plots</p></div>
//                   <div className="feature-card"><h3>🔍 Insights</h3><p>Actionable healthcare analytics</p></div>
//                 </div>
//               </div>
//             </div>
//           )}
//         </main>
//       </div>
//     </div>
//   );
// }

// export default App;

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedExperiment, setSelectedExperiment] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const experiments = [
    { value: 'data_visualization', label: '📊 Data Visualization', description: 'Explore correlations and data distribution' },
    { value: 'linear_regression', label: '📈 Linear Regression', description: 'Predict continuous health metrics' },
    { value: 'classification', label: '🌳 Decision Tree Classification', description: 'Classify heart disease risk' },
    { value: 'svm', label: '🔍 Support Vector Machine', description: 'Advanced classification analysis' },
    { value: 'ensemble', label: '🌲 Ensemble Learning', description: 'Random Forest & Gradient Boosting' },
    { value: 'nonlinear_regression', label: '📏 Polynomial Regression', description: 'Non-linear relationship modeling' },
    { value: 'clustering', label: '🎯 Clustering Analysis', description: 'Discover patient groups' },
    { value: 'pca', label: '📐 PCA Analysis', description: 'Dimensionality reduction' }
  ];

  const runExperiment = async () => {
    if (!selectedExperiment) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post('http://localhost:5000/experiment', {
        experiment: selectedExperiment
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to run experiment');
    } finally {
      setLoading(false);
    }
  };

  const formatMetricValue = (value) => {
    if (typeof value === 'number') return value.toFixed(4);
    if (Array.isArray(value)) return JSON.stringify(value, null, 2);
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="title">🏥 Healthcare ML Analytics Dashboard</h1>
          <p className="subtitle">Interactive Machine Learning Experiments on Healthcare Data</p>
        </div>
      </header>

      <div className="container">
        <aside className="sidebar">
          <div className="sidebar-content">
            <h2 className="sidebar-title">ML Experiments</h2>

            <div className="experiment-list">
              {experiments.map((exp) => (
                <div
                  key={exp.value}
                  className={`experiment-card ${selectedExperiment === exp.value ? 'selected' : ''}`}
                  onClick={() => setSelectedExperiment(exp.value)}
                >
                  <div className="experiment-label">{exp.label}</div>
                  <div className="experiment-description">{exp.description}</div>
                </div>
              ))}
            </div>

            <button
              className={`run-button ${!selectedExperiment || loading ? 'disabled' : ''}`}
              onClick={runExperiment}
              disabled={!selectedExperiment || loading}
            >
              {loading ? '🔄 Running...' : '▶️ Run Experiment'}
            </button>

            {selectedExperiment && (
              <div className="selected-info">
                <h3>Selected:</h3>
                <p>{experiments.find(exp => exp.value === selectedExperiment)?.label}</p>
              </div>
            )}
          </div>
        </aside>

        <main className="main-content">
          {loading && (
            <div className="loading-container">
              <div className="loading-spinner"><div className="spinner"></div></div>
              <div className="loading-text">
                <h3>🧠 Running ML Experiment...</h3>
                <p>Processing healthcare data and generating insights</p>
              </div>
            </div>
          )}

          {error && (
            <div className="error-container">
              <div className="error-content">
                <h3>⚠️ Error</h3>
                <p>{error}</p>
                <button onClick={() => setError('')} className="error-dismiss">Dismiss</button>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="results-container">
              <div className="results-header">
                <h2 className="results-title">{result.experiment}</h2>
                <div className="results-timestamp">⏰ Completed: {new Date().toLocaleString()}</div>
              </div>

              <div className="results-content">
                {/* Metrics Section */}
                {result.metrics && (
                  <div className="metrics-section">
                    <h3 className="section-title">📊 Experiment Metrics</h3>
                    <div className="metrics-grid">
                      {Object.entries(result.metrics).map(([key, value]) => (
                        <div key={key} className="metric-card">
                          <div className="metric-label">
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </div>
                          <div className="metric-value">{formatMetricValue(value)}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Visualization Section */}
                {(result.plot || result.plots) && (
                  <div className="visualization-section">
                    <h3 className="section-title">📈 Visualization Results</h3>

                    {/* For multiple plots */}
                    {result.plots &&
                      Object.entries(result.plots).map(([key, val]) => (
                        <div key={key} style={{ marginBottom: '2rem', textAlign: 'center' }}>
                          <h4>{key.replace(/_/g, ' ').toUpperCase()}</h4>
                          <img
                            src={`data:image/png;base64,${val}`}
                            alt={key}
                            style={{
                              maxWidth: key === 'confusion_matrix' ? '55%' : '90%',
                              borderRadius: '10px',
                              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
                              marginTop: '1rem'
                            }}
                          />
                        </div>
                      ))}
                  </div>
                )}

                {/* Insight Section */}
                {result.insight && (
                  <div
                    className="insight-section"
                    style={{
                      background: '#f0f8ff',
                      padding: '1.5rem',
                      borderRadius: '12px',
                      marginTop: '2rem',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}
                  >
                    <h3 style={{ color: '#2563eb', marginBottom: '1rem' }}>
                      {result.insight.Graph || '💡 Insights and Interpretation'}
                    </h3>
                    <div
                      dangerouslySetInnerHTML={{ __html: result.insight.Insight }}
                      style={{
                        color: '#333',
                        lineHeight: '1.7',
                        fontSize: '1rem'
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="welcome-container">
              <div className="welcome-content">
                <h2>🎯 Welcome to Healthcare ML Dashboard</h2>
                <p>Select an experiment from the sidebar to begin your machine learning analysis</p>

                <div className="features-grid">
                  <div className="feature-card"><h3>📊 Data Analysis</h3><p>Explore correlations and statistical insights</p></div>
                  <div className="feature-card"><h3>🤖 ML Models</h3><p>Train and evaluate various algorithms</p></div>
                  <div className="feature-card"><h3>📈 Visualizations</h3><p>Interactive charts and plots</p></div>
                  <div className="feature-card"><h3>🔍 Insights</h3><p>Actionable healthcare analytics</p></div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
