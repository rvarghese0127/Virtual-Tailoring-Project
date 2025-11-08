# Virtual Tailoring Measurement System
## AI-Powered Body Measurement CV Model for T-Pose Analysis

### Overview
A sophisticated Computer Vision system designed for virtual tailoring that captures body measurements from T-Pose images. The system uses advanced pose detection to identify skeletal joints and calculate three key measurements: Wingspan, Inseam/Leg Length, and Stomach Width.

### Key Features
- **Automatic T-Pose Validation**: Ensures proper pose quality before measurements
- **Scale Factor Calculation**: Uses known user height to establish accurate measurements
- **Multi-Unit Support**: Works with cm, inches, feet, and meters
- **Confidence Scoring**: Provides confidence levels for each measurement
- **Error Handling**: Clear error messages for pose issues or visibility problems
- **Web API Interface**: REST API with interactive web UI for easy integration

### 🔧 System Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Flask (for web API)


### Output Format

The system provides measurements in a clear, concise table format:

```
======================================================================
VIRTUAL TAILORING MEASUREMENTS
======================================================================
Measurement          Value           Confidence      Quality        
----------------------------------------------------------------------
Wingspan             178.5 cm         93.0%         Excellent      
Inseam/Leg Length     81.2 cm         89.0%         Good           
Stomach Width         36.8 cm         85.0%         Good           
======================================================================
Overall Pose Quality: Excellent
======================================================================
```

### T-Pose Requirements

For optimal results, ensure:
1. **Arms Extended**: Straight out horizontally (T-pose)
2. **Full Body Visible**: From head to feet in frame
3. **Clear Background**: Plain background preferred
4. **Proper Lighting**: Even lighting without shadows
5. **Fitted Clothing**: For better body contour detection

### Measurements Explained

#### 1. **Wingspan** (Fingertip to Fingertip)
- Measures the distance from left fingertip to right fingertip
- Uses wrist detection with hand length approximation
- Critical for shirt/jacket sizing

#### 2. **Inseam/Leg Length** (Crotch to Floor)
- Measures from crotch point to floor
- Essential for pants/trouser fitting
- Calculated using hip and ankle positions

#### 3. **Stomach Width** (Side-to-Side Diameter)
- Measures waist width at stomach level
- Uses segmentation when available for accuracy
- Important for waist sizing

### Pose Quality Levels

- **EXCELLENT**: Perfect T-pose, all joints clearly visible (90-100% score)
- **GOOD**: Minor issues, measurements reliable (75-89% score)
- **ACCEPTABLE**: Some problems, measurements usable (60-74% score)
- **POOR**: Significant issues, measurements uncertain (40-59% score)
- **INVALID**: Cannot process, major pose problems (<40% score)

### 🚀 Deployment on Vercel

This project is configured for deployment on Vercel. Follow these steps:

1. **Push to GitHub**: Make sure your code is pushed to a GitHub repository
2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the Python configuration
3. **Deploy**: Click "Deploy" and wait for the build to complete

**Project Structure for Vercel:**
```
├── api/
│   └── measure.py          # API endpoint for processing images
├── index.html           # Frontend web interface
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel configuration
└── virtual_tailor.py    # Original desktop application
```

**Important Notes:**
- The YOLO model (`yolov8n-pose.pt`) will be automatically downloaded on first cold start
- First request may take longer due to model download (~30-60 seconds)
- Subsequent requests will be faster
- The API endpoint is available at `/api/measure`

**API Usage:**
```javascript
POST /api/measure
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",  // Base64 encoded image
  "height": 170,                          // User height
  "unit": "cm"                            // Unit: "cm", "inches", or "m"
}
```

### Support

For issues or questions:
1. Check the error messages for specific guidance
2. Ensure T-pose requirements are met
3. Verify image quality and lighting
4. Check Vercel deployment logs if deployment fails
