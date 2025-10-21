# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Import the new model manager
from .model_manager import model_manager
from .gemini_report import generate_diagnostic_report

# Initialize FastAPI app
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for brain tumor classification from MRI images using a NAS-trained model and Gemini AI for diagnostic reports.",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This is crucial to allow your React frontend (running on a different port/origin)
# to make requests to your FastAPI backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your React app's URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    Pre-load the default Random Search NAS model.
    """
    try:
        # Pre-load the random search model (most commonly used)
        model_manager.load_model("random")
        print("FastAPI application started. Random Search NAS model is ready.")
        print("Other models will be loaded on-demand when selected.")
    except Exception as e:
        print(f"Note: Could not pre-load model: {e}")
        print("Models will be loaded on-demand.")

@app.get("/")
async def read_root():
    """Root endpoint for basic API health check."""
    return {"message": "Welcome to the Brain Tumor Detection API! Visit /docs for API documentation."}

@app.post("/predict/", summary="Upload MRI image for tumor classification with selected NAS method")
async def predict(
    file: UploadFile = File(..., description="Upload an MRI image in PNG format."),
    nas_method: str = Query("random", description="NAS search method: 'random', 'gradient', or 'reinforcement'")
):
    """
    Receives an uploaded MRI image and NAS method selection,
    processes it with the corresponding deep learning model,
    and generates a diagnostic-style report using Gemini AI.
    """
    # Validate NAS method
    valid_methods = ["random", "gradient", "reinforcement"]
    if nas_method not in valid_methods:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid NAS method. Must be one of: {', '.join(valid_methods)}"
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Uploaded file is not a valid image. Please upload an image file."
        )
    
    # Read image bytes from the uploaded file
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    try:
        # Perform prediction using the selected NAS method's model
        result = await model_manager.predict(image_bytes, nas_method)
        
        # Check if there was an error loading the model
        if result.get("error"):
            return JSONResponse(
                status_code=501,  # Not Implemented
                content={
                    "error": True,
                    "message": result["message"],
                    "nas_method": nas_method,
                    "suggestion": "This NAS method's model is not yet available. Please select another method or train this model first."
                }
            )
        
        # Extract prediction results
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        all_probabilities = result["all_probabilities"]

        # Generate the diagnostic report using Gemini AI
        gemini_report_text = await generate_diagnostic_report(
            predicted_class, confidence, all_probabilities
        )

        # Return the results as a JSON response
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "all_probabilities": all_probabilities,
            "diagnostic_report": gemini_report_text,
            "nas_method": nas_method,
            "model_info": get_model_info(nas_method)
        })
    except Exception as e:
        # Catch any exceptions during prediction or report generation
        print(f"An error occurred during prediction or report generation: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

def get_model_info(nas_method: str) -> dict:
    """Return information about the NAS method and its model"""
    info = {
        "random": {
            "name": "Random Search NAS",
            "description": "Explores architectures through random sampling",
            "reported_accuracy": "60.15%",
            "architecture": "Custom 6-layer CNN with adaptive pooling"
        },
        "gradient": {
            "name": "Gradient-Based NAS",
            "description": "Uses gradient descent for architecture optimization",
            "reported_accuracy": "98.43%",
            "architecture": "MobileNetV3-based with squeeze-excitation blocks"
        },
        "reinforcement": {
            "name": "Reinforcement Learning NAS",
            "description": "Agent-based architecture search using RL",
            "reported_accuracy": "TBD",
            "architecture": "To be determined"
        }
    }
    return info.get(nas_method, {})