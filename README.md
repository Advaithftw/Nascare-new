# Brain Tumor Classification using Neural Architecture Search (NAS)

## Project Overview
This project implements a **multi-model brain tumor classification system** using different Neural Architecture Search (NAS) approaches. It features a web interface for uploading MRI images and comparing predictions across different NAS methods.

## 🎯 Key Features
- **Three NAS Training Methods**:
  - Random Search NAS (`newtraining.py`)
  - Gradient-Based NAS (`oldcode.py`)
  - Reinforcement Learning NAS (`rltraining.py`)
- **Web Interface**: React frontend with FastAPI backend
- **Real-time Predictions**: Upload MRI images and get instant classifications
- **Model Comparison**: Switch between different NAS methods to compare results
- **Four Tumor Classes**: Glioma, Meningioma, Pituitary, No Tumor

## 📁 Project Structure
```
mednas/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── model.py            # Random Search NAS model architecture
│   ├── github_model.py     # Gradient-Based NAS (EfficientNet-B4)
│   ├── model_manager.py    # Multi-model handler
│   ├── mat_dataset.py      # Dataset utilities
│   └── models/             # Saved model checkpoints
│       ├── the_nas_model.pth      # Random Search model
│       └── best_final_model.pt    # Pre-trained model
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.jsx        # Main application
│   │   └── ...
│   └── package.json
├── tools/                  # Utility scripts
│   ├── convert_mat_to_images.py
│   └── test_mat_dataset.py
├── newtraining.py         # 🔴 Random Search NAS training
├── oldcode.py             # 🔴 Gradient-Based NAS training
├── rltraining.py          # 🔴 Reinforcement Learning NAS training
└── training.py            # Legacy training script
```

## 🚀 Setup Instructions

### Backend Setup
```bash
# Navigate to project root
cd mednas

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the backend server
python -m uvicorn backend.main:app --port 8001 --reload
```

### Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8001

## 📊 NAS Training Methods

### 1. Random Search NAS (`newtraining.py`)
Explores the architecture search space randomly, evaluating different combinations of:
- **Conv layers**: 128, 32, 32, 16, 32, 128 filters
- **Kernel sizes**: 2×2 and 5×5
- **FC layers**: 128, 128 units
- **Activations**: ReLU, LeakyReLU
- **Dropout**: 0.3
- **Accuracy**: 60.15%

### 2. Gradient-Based NAS (`oldcode.py`)
Uses EfficientNet-B4 architecture with custom classifier:
- **Base**: EfficientNet-B4 (1792 features)
- **Classifier**: Linear(1792→256) → ReLU → Dropout → Linear(256→4)
- **Image size**: 380×380
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Accuracy**: 98.43%

### 3. Reinforcement Learning NAS (`rltraining.py`)
Uses RL agent to learn optimal architectures. Currently using the same EfficientNet-B4 model:
- **Architecture**: Same as Gradient-Based NAS (EfficientNet-B4)
- **Image size**: 380×380
- **Normalization**: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Accuracy**: 98.43% (training)

## 🎨 Frontend Features

### NAS Method Selection
Users can select from three NAS methods:
- **Random Search NAS**: Fast, simpler architecture
- **Gradient-Based NAS**: High accuracy, EfficientNet-B4
- **Reinforcement Learning NAS**: Adaptive architecture learning

### Image Upload & Prediction
1. Upload brain MRI image
2. Select NAS method
3. View prediction results with confidence scores
4. See probability distribution across all classes

## 🔧 API Endpoints

### `POST /predict/`
**Parameters:**
- `file`: Image file (multipart/form-data)
- `nas_method`: "random" | "gradient" | "reinforcement"

**Response:**
```json
{
  "error": false,
  "predicted_class": "glioma_tumor",
  "confidence": 86.23,
  "all_probabilities": {
    "glioma_tumor": 86.23,
    "meningioma_tumor": 10.45,
    "no_tumor": 2.15,
    "pituitary_tumor": 1.17
  },
  "nas_method": "gradient"
}
```

## 📝 Dataset

**Source**: [Brain Tumor Classification (MRI) - Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

**Classes:**
- Glioma Tumor
- Meningioma Tumor
- No Tumor
- Pituitary Tumor

**Format**: `.mat` files (MATLAB format) containing MRI images

## 🧠 Model Architecture Details

### Random Search NAS Model
```
Input (224×224×3)
├── Conv2D(128, 2×2) → BatchNorm → ReLU → MaxPool
├── Conv2D(32, 2×2) → BatchNorm → ReLU → MaxPool
├── Conv2D(32, 2×2) → BatchNorm → ReLU → MaxPool
├── Conv2D(16, 2×2) → BatchNorm → LeakyReLU → MaxPool
├── Conv2D(32, 2×2) → BatchNorm → LeakyReLU → MaxPool
├── Conv2D(128, 5×5) → BatchNorm → LeakyReLU → MaxPool
├── GlobalAvgPool2D
├── FC(?, 128) → LeakyReLU → Dropout(0.3)
├── FC(128, 128) → ReLU → Dropout(0.3)
└── FC(128, 4)
```

### Gradient-Based NAS Model (EfficientNet-B4)
```
Input (380×380×3) → Normalize
├── EfficientNet-B4 Base (1792 features)
├── AdaptiveAvgPool2D(1×1)
├── Flatten
├── Linear(1792 → 256)
├── ReLU
├── Dropout(0.5)
└── Linear(256 → 4)
```

## 🐛 Known Issues & Solutions

### BatchNorm Issue with Gradient Model
The EfficientNet-B4 checkpoint has corrupted BatchNorm running statistics. 

**Solution**: Model uses `.train()` mode for BatchNorm (computes batch statistics) while keeping Dropout in `.eval()` mode.

## 📚 Key Learning Resources

### Neural Architecture Search
- [NAS Survey Paper](https://arxiv.org/abs/1808.05377)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

### Implementation Details
- Random Search: Explores predefined architecture space
- Gradient-Based: Uses pre-trained EfficientNet with transfer learning
- RL-Based: Agent learns to select optimal layer configurations

## 🎓 Assignment Submission Notes

**Important Training Files:**
1. **`newtraining.py`**: Random Search NAS implementation
2. **`oldcode.py`**: Gradient-Based NAS (EfficientNet fine-tuning)
3. **`rltraining.py`**: Reinforcement Learning NAS approach

**Additional Components:**
- Full-stack web application (React + FastAPI)
- Multi-model comparison system
- Real-time inference API
- Dataset preprocessing utilities

## 🤝 Contributing
This is an academic project. For improvements or suggestions, please follow standard git workflow.

## 📄 License
Academic Project - All Rights Reserved

## 👨‍💻 Author
Sakthivel - Neural Architecture Search for Brain Tumor Classification

---

**Note**: Model checkpoint files (`.pth`, `.pt`) are not included in the repository due to size constraints. Train the models using the provided training scripts or contact the author for pre-trained weights.
