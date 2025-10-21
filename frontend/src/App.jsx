// frontend/src/App.jsx
import React, { useState } from 'react';
import './App.css'; // Import the CSS file for styling

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [nasMethod, setNasMethod] = useState('random'); // Default to random NAS

  // Handle file selection by the user
  const handleFileChange = (event) => {
    const file = event.target.files[0]; // Get the first selected file
    setSelectedFile(file);
    if (file) {
      setPreviewImage(URL.createObjectURL(file)); // Create a URL for image preview
      setPredictionResult(null); // Clear previous results when a new file is selected
      setError(null); // Clear any previous errors
    } else {
      setPreviewImage(null); // Clear preview if no file is selected
    }
  };

  // Handle image upload and prediction request
  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an MRI image to upload.");
      return;
    }

    setLoading(true); // Set loading state to true
    setError(null);    // Clear any previous errors
    setPredictionResult(null); // Clear previous results

    const formData = new FormData(); // Create a FormData object to send the file
    formData.append('file', selectedFile); // Append the selected file under the key 'file'

    try {
      // Make a POST request to your FastAPI backend's /predict/ endpoint with NAS method
      // **IMPORTANT**: Ensure this URL matches where your FastAPI backend is running.
      const response = await fetch(`http://localhost:8001/predict/?nas_method=${nasMethod}`, { 
        method: 'POST',
        body: formData, // Send the FormData object
      });

      // Check if the response was successful (HTTP status code 2xx)
      if (!response.ok) {
        // Try to parse error details from the response
        try {
          const errorData = await response.json();
          
          // Check if this is a "not implemented" error for gradient/RL models
          if (response.status === 501 && errorData.error) {
            setError(
              `⚠️ ${errorData.message}\n\n` +
              `💡 ${errorData.suggestion || 'Try selecting Random Search NAS instead.'}\n\n` +
              `To enable ${errorData.nas_method} NAS:\n` +
              (errorData.nas_method === 'gradient' 
                ? '• Run: python train_mobilenetv3.py\n• Wait for training to complete\n• Restart the backend'
                : '• This feature needs to be implemented')
            );
            return;
          }
          
          throw new Error(errorData.detail || errorData.message || `HTTP error! Status: ${response.status}`);
        } catch (parseError) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
      }

      const data = await response.json(); // Parse the JSON response from the backend
      
      // Check if the model is not available
      if (data.error) {
        setError(data.message + "\n\nSuggestion: " + (data.suggestion || "Try another NAS method."));
        return;
      }
      
      setPredictionResult(data); // Store the prediction results
    } catch (e) {
      console.error("Error during prediction:", e);
      setError(`Failed to get prediction: ${e.message}`); // Display error message to the user
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 Brain Tumor Classifier</h1>
        <p>Upload an MRI scan to get an AI-powered preliminary analysis and diagnostic-style report.</p>
      </header>

      <div className="nas-method-selector card">
        <h2>🔍 Select NAS Search Method:</h2>
        <div className="radio-group">
          <label className="radio-label">
            <input
              type="radio"
              value="random"
              checked={nasMethod === 'random'}
              onChange={(e) => setNasMethod(e.target.value)}
            />
            <span>Random Search NAS</span>
            <small className="method-description">Explores architectures randomly (60% accuracy)</small>
          </label>
          
          <label className="radio-label">
            <input
              type="radio"
              value="gradient"
              checked={nasMethod === 'gradient'}
              onChange={(e) => setNasMethod(e.target.value)}
            />
            <span>Gradient-Based NAS</span>
            <small className="method-description">Uses gradient descent optimization (98% accuracy)</small>
          </label>
          
          <label className="radio-label">
            <input
              type="radio"
              value="reinforcement"
              checked={nasMethod === 'reinforcement'}
              onChange={(e) => setNasMethod(e.target.value)}
            />
            <span>Reinforcement Learning NAS</span>
            <small className="method-description">Agent-based architecture search (Coming soon)</small>
          </label>
        </div>
      </div>

      <div className="upload-section card">
        <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />
        <button onClick={handleUpload} disabled={!selectedFile || loading} className="upload-button">
          {loading ? 'Analyzing...' : 'Analyze MRI Image'}
        </button>
        {loading && <div className="spinner"></div>} 
      </div>

      {error && <p className="message error-message">Error: {error}</p>}

      {previewImage && (
        <div className="image-preview card">
          <h2>Uploaded MRI Image:</h2>
          <img src={previewImage} alt="MRI Preview" className="uploaded-image" />
        </div>
      )}

      {predictionResult && (
        <div className="results-section card">
          <div className="model-info-banner">
            <h3>🤖 Model Used: {predictionResult.model_info?.name || nasMethod.toUpperCase()}</h3>
            <p>{predictionResult.model_info?.description}</p>
            <p><strong>Architecture:</strong> {predictionResult.model_info?.architecture}</p>
            <p><strong>Reported Accuracy:</strong> {predictionResult.model_info?.reported_accuracy}</p>
          </div>
          
          <h2>📊 Prediction Results:</h2>
          <p><strong>Predicted Tumor Type:</strong> <span className="highlight-text">{predictionResult.predicted_class.replace('_', ' ').toUpperCase()}</span></p>
          <p><strong>Confidence:</strong> <span className="highlight-text">{predictionResult.confidence}</span></p>

          <h3>All Class Probabilities:</h3>
          <ul className="probabilities-list">
            {Object.entries(predictionResult.all_probabilities).map(([className, probability]) => (
              <li key={className}>
                {className.replace('_', ' ').toUpperCase()}: {probability.toFixed(2)}%
              </li>
            ))}
          </ul>

          <h2>📄 AI-Generated Diagnostic Report:</h2>
          <div className="diagnostic-report">
            <p>{predictionResult.diagnostic_report}</p>
          </div>
        </div>
      )}

      <footer className="App-footer">
        <p>Powered by Deep Learning (NAS), FastAPI, React, and Google Gemini AI</p>
        <p>&copy; 2025 Brain Tumor Detector</p>
      </footer>
    </div>
  );
}

export default App;