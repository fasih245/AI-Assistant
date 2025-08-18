# 🤖 Enhanced Smart AI Assistant with MediaPipe Integration

> A comprehensive AI-powered assistant featuring advanced computer vision, natural language processing, and real-time analysis capabilities.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/enhanced-ai-assistant/graphs/commit-activity)

## 🌟 Overview

The Enhanced Smart AI Assistant is a cutting-edge application that combines artificial intelligence with advanced computer vision capabilities. Built with Streamlit and powered by multiple AI models, it offers a comprehensive suite of tools for object detection, pose analysis, emotion recognition, and intelligent conversation.

## ✨ Key Features

### 🧠 AI Conversation Engine
- **Multi-Model Support**: Integration with Google Gemini and Groq APIs
- **13 Conversation Moods**: From professional to casual, sarcastic to inspirational
- **RAG (Retrieval-Augmented Generation)**: Upload documents for context-aware responses
- **Temperature Control**: Adjust creativity and randomness of AI responses
- **Smart Document Processing**: Support for PDF and TXT files up to 100MB

### 👁️ Computer Vision Suite
- **YOLO Object Detection**: Real-time object segmentation with three output modes
  - Object Segmented (SG): Masks overlaid on original image
  - Object Detected (BB): Bounding boxes with labels
  - Pure Segmentation Masks: Clean mask overlays
- **MediaPipe Integration**: Advanced face, hand, and pose landmark detection
- **Emotion Recognition**: Real-time facial emotion analysis with confidence scoring
- **Pose Estimation**: Human pose detection with 17 keypoint analysis

### 📹 Real-Time Analysis
- **Live Camera Feed**: Real-time video processing with auto-capture
- **Multi-Modal Analysis**: Simultaneous object, pose, and emotion detection
- **Auto-Capture System**: Configurable intervals for automatic frame analysis
- **Performance Monitoring**: Memory usage tracking and cache management

### 📊 Analytics Dashboard
- **Object Detection Dashboard**: Track detection history and confidence metrics
- **Sentiment Analysis**: Real-time text sentiment with VADER and TextBlob
- **Detection Analytics**: Object frequency, confidence distribution, and trends
- **Export Functionality**: Download analysis results and application settings

### 🎨 User Experience
- **Dark/Light Themes**: Customizable UI with professional styling
- **Responsive Design**: Optimized for desktop and mobile viewing
- **Interactive Controls**: Real-time sliders and settings panels
- **Progress Indicators**: Visual feedback for long-running operations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Webcam (for real-time analysis features)
- API Keys (optional, for enhanced AI features):
  - Google Gemini API key
  - Groq API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/enhanced-ai-assistant.git
   cd enhanced-ai-assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   # Create .env file
   touch .env  # On Windows: type nul > .env
   ```
   
   Add your API keys to `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   GMAIL_EMAIL=your_email@gmail.com
   GMAIL_APP_PASSWORD=your_app_password_here
   ```

5. **Initialize emotion detection model:**
   ```bash
   python emotion_setup.py
   ```

6. **Run the application:**
   ```bash
   streamlit run ultra_simple_bot.py
   ```

7. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Getting Started
1. **Configure Settings**: Use the sidebar to adjust AI temperature, mood, and confidence thresholds
2. **Choose Your Page**: Navigate between different features using the page selector
3. **Upload Content**: Add documents for RAG or images for computer vision analysis
4. **Start Analyzing**: Use real-time camera or upload images for instant analysis

### Feature Walkthrough

#### 💬 Chat Assistant
- Upload PDF/TXT documents for context-aware conversations
- Adjust temperature (0.0-2.0) for response creativity
- Choose from 13 different conversation moods
- Send responses via email integration

#### 🎭 Object Segmentation
- Upload images for YOLO-based object detection
- View three different output types
- Access detailed detection dashboard with analytics
- Export segmentation results

#### 🤸 Pose Detection
- Choose between YOLO and MediaPipe pose detection
- Analyze human poses with 17 keypoint detection
- View confidence scores and keypoint details
- Track pose detection history

#### 😊 Emotion Analysis
- Real-time facial emotion recognition
- Support for 7 emotion categories
- Confidence scoring and probability breakdown
- Detailed emotion insights and analytics

#### 📹 Real-Time Camera
- Live video feed with multiple analysis modes
- Auto-capture with configurable intervals
- Simultaneous multi-modal analysis
- Real-time results dashboard

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key for AI responses | Optional |
| `GROQ_API_KEY` | Groq API key for AI responses | Optional |
| `GMAIL_EMAIL` | Gmail address for email features | Optional |
| `GMAIL_APP_PASSWORD` | Gmail app password for email features | Optional |

### Model Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_SEGMENTATION_MODEL` | `yolov8m-seg.pt` | YOLO model for object segmentation |
| `YOLO_POSE_MODEL` | `yolov8m-pose.pt` | YOLO model for pose detection |
| `EMOTION_MODEL_PATH` | `./models/emotion_model.h5` | Path to emotion detection model |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload file size |

## 📁 Project Structure

```
enhanced-ai-assistant/
├── ultra_simple_bot.py          # Main application file
├── emotion_setup.py              # Emotion model setup script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
├── .env                          # Environment variables (create this)
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── models/                       # AI model storage (auto-created)
├── documents/                    # Document upload storage
├── temp/                         # Temporary files
└── logs/                         # Application logs
```

## 🔧 API Keys Setup

### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to your `.env` file as `GOOGLE_API_KEY`

### Groq API
1. Visit [Groq Console](https://console.groq.com/keys)
2. Create a new API key
3. Add to your `.env` file as `GROQ_API_KEY`

### Gmail Integration (Optional)
1. Enable 2-factor authentication on your Gmail account
2. Generate an app password
3. Add your email and app password to `.env`

## 🎯 Performance Optimization

### System Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB+ RAM, 4+ core CPU
- **GPU**: Optional, CUDA-compatible GPU for faster processing

### Optimization Tips
- Use CPU-only PyTorch for better compatibility: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Reduce auto-capture frequency for lower-end systems
- Close unnecessary applications when using real-time features
- Clear cache regularly using the performance monitor

## 🐛 Troubleshooting

### Common Issues

**PyTorch/YOLO Import Errors:**
```bash
pip uninstall torch torchvision ultralytics -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

**Camera Access Issues:**
- Check camera permissions in system settings
- Ensure no other applications are using the camera
- Try different camera indices (0, 1, 2) in settings

**Model Loading Errors:**
```bash
python emotion_setup.py
```

**Memory Issues:**
- Use the "Clear Cache" button in Performance Monitor
- Reduce image upload sizes
- Close other memory-intensive applications

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Google MediaPipe**: For advanced computer vision capabilities
- **Ultralytics YOLO**: For state-of-the-art object detection
- **OpenCV**: For computer vision fundamentals
- **TensorFlow**: For deep learning capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/enhanced-ai-assistant/issues) page
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Roadmap

- [ ] Voice interaction capabilities
- [ ] Custom model training interface
- [ ] Advanced analytics export (PDF reports)
- [ ] Multi-language support
- [ ] Real-time collaborative features
- [ ] Mobile app development
- [ ] Cloud deployment templates

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Fasih Ul Haq](https://github.com/yourusername)

[🐛 Report Bug](https://github.com/yourusername/enhanced-ai-assistant/issues) • [✨ Request Feature](https://github.com/yourusername/enhanced-ai-assistant/issues) • [📖 Documentation](https://github.com/yourusername/enhanced-ai-assistant/wiki)

</div>