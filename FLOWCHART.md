<!-- Mermaid Based Flowchart -->

flowchart TD
    A[🚀 Da_Bot Startup] --> B{🔧 System Initialization}
    B --> C[📦 Load Dependencies]
    B --> D[🎛️ Initialize Session State]
    B --> E[🎨 Apply Theme Configuration]
    
    C --> C1[🤖 AI Models<br/>Gemini Pro & Groq LLaMA]
    C --> C2[👁️ Computer Vision<br/>YOLO v8 & MediaPipe]
    C --> C3[🎤 Audio Systems<br/>TTS & Speech Recognition]
    C --> C4[😊 Emotion Detection<br/>TensorFlow Model]
    C --> C5[💭 Sentiment Analysis<br/>VADER & TextBlob]
    
    D --> D1[💬 Chat History: 20 msgs]
    D --> D2[📄 Documents: 200MB limit]
    D --> D3[🎭 Detection History: 50 items]
    D --> D4[⚙️ User Preferences]
    
    E --> E1[🌙 Dark Theme<br/>Professional Gradients]
    E --> E2[☀️ Light Theme<br/>Clean Interface]
    
    C1 --> F[📱 Main Navigation]
    C2 --> F
    C3 --> F
    C4 --> F
    C5 --> F
    
    F --> G1[💬 Enhanced Chat Assistant]
    F --> G2[🎭 Object Segmentation]
    F --> G3[🤖 MediaPipe Analysis]
    F --> G4[🤸 Pose Detection]
    F --> G5[😊 Emotion Analysis]
    F --> G6[📹 Real-time Camera]
    F --> G7[💭 Sentiment Dashboard]
    F --> G8[📊 Analytics & History]
    
    %% Enhanced Chat Assistant Flow
    G1 --> H1[📝 Text Input]
    G1 --> H2[🎤 Voice Input]
    G1 --> H3[📁 Document Upload]
    
    H1 --> I1{🧠 AI Processing}
    H2 --> I2[🔄 Speech-to-Text] --> I1
    H3 --> I3[📖 RAG Processing] --> I1
    
    I1 --> J1[🎭 Mood Selection<br/>13 Conversation Styles]
    I1 --> J2[🌡️ Temperature Control<br/>0.0-2.0 Creativity]
    I1 --> J3[📏 Token Limit<br/>100-1000 words]
    
    J1 --> K[🤖 AI Response Generation]
    J2 --> K
    J3 --> K
    
    K --> L1[💬 Text Response]
    K --> L2[🔊 Audio Response]
    K --> L3[📧 Email Integration]
    
    %% Computer Vision Flow
    G2 --> M1[📷 Image Upload]
    G3 --> M1
    G4 --> M1
    G5 --> M1
    G6 --> M2[📹 Live Camera Feed]
    
    M1 --> N{🔍 Analysis Selection}
    M2 --> N
    
    N --> N1[🎭 YOLO Segmentation<br/>3 Output Modes]
    N --> N2[👤 MediaPipe Face<br/>468 Landmarks]
    N --> N3[✋ MediaPipe Hands<br/>21 Points Each]
    N --> N4[🤸 MediaPipe Pose<br/>33 Body Points]
    N --> N5[😊 Emotion Recognition<br/>7 Categories]
    
    N1 --> O1[🖼️ SG: Segmented Image]
    N1 --> O2[📦 BB: Bounding Boxes]
    N1 --> O3[🎨 Pure Masks]
    
    N2 --> P1[👤 Face Bounding Boxes]
    N3 --> P2[✋ Hand Skeleton]
    N4 --> P3[🤸 Pose Skeleton]
    N5 --> P4[😊 Emotion Labels]
    
    %% Real-time Processing
    G6 --> Q1[🔄 Auto-Capture System<br/>1-10s Intervals]
    Q1 --> Q2{⏱️ Timeout Protection}
    Q2 --> Q3[🖥️ Windows Threading]
    Q2 --> Q4[🐧 Unix Signal Handling]
    
    Q3 --> R[📊 Live Results Dashboard]
    Q4 --> R
    
    %% Analytics and Dashboard
    G7 --> S1[📈 Real-time Sentiment]
    G8 --> S2[📊 Detection Analytics]
    
    S1 --> T1[📝 Text Analysis<br/>VADER + TextBlob]
    S1 --> T2[📊 Batch Processing]
    S1 --> T3[📈 Trend Visualization]
    
    S2 --> U1[🎭 Object Frequency]
    S2 --> U2[🎯 Confidence Distribution]
    S2 --> U3[📅 Historical Trends]
    
    %% Data Flow and Storage
    R --> V[💾 Data Storage]
    T1 --> V
    T2 --> V
    U1 --> V
    
    V --> V1[📄 Documents Cache<br/>200MB Support]
    V --> V2[🎭 Detection History<br/>50 Items Max]
    V --> V3[💭 Sentiment History<br/>50 Analyses]
    V --> V4[📊 Performance Metrics]
    
    %% Export and Integration
    V --> W[📤 Export Options]
    W --> W1[📧 Email Reports<br/>Unicode Safe]
    W --> W2[📊 CSV Analytics]
    W --> W3[📋 JSON Settings]
    W --> W4[📄 PDF Reports]
    
    %% Error Handling and Recovery
    K --> X{❌ Error Detection}
    R --> X
    V --> X
    
    X --> X1[🔄 Automatic Retry<br/>3 Attempts]
    X --> X2[⬇️ Graceful Fallback<br/>Alternative Methods]
    X --> X3[📝 Error Logging<br/>User Friendly Messages]
    
    X1 --> Y[🔧 Recovery Actions]
    X2 --> Y
    X3 --> Y
    
    Y --> Y1[🧹 Cache Cleanup]
    Y --> Y2[🔄 Model Reload]
    Y --> Y3[📱 UI Reset]
    
    %% Performance Optimization
    Y --> Z[⚡ Performance Monitor]
    Z --> Z1[💾 Memory Usage<br/>Real-time Tracking]
    Z --> Z2[🖥️ CPU Utilization]
    Z --> Z3[📊 Session Metrics]
    Z --> Z4[🧹 Auto Cache Clear]
    
    %% Security and Configuration
    Z --> AA[🔐 Security Layer]
    AA --> AA1[🔑 API Key Management<br/>.env Protection]
    AA --> AA2[📧 Email Encryption<br/>SMTP TLS]
    AA --> AA3[📄 File Validation<br/>Size & Type Limits]
    AA --> AA4[🛡️ Input Sanitization]
    
    %% Final Output and User Experience
    AA --> BB[🎯 User Experience]
    BB --> BB1[🎨 Professional UI<br/>Dark/Light Themes]
    BB --> BB2[📱 Responsive Design<br/>Mobile Optimized]
    BB --> BB3[⚡ Real-time Feedback<br/>Progress Indicators]
    BB --> BB4[🔊 Accessibility<br/>Screen Reader Support]
    
    BB --> CC[✅ Mission Complete<br/>Enterprise-Ready AI Assistant]
    
    %% Styling
    classDef aiModel fill:#ff6b6b,stroke:#333,stroke-width:2px,color:#fff
    classDef vision fill:#4ecdc4,stroke:#333,stroke-width:2px,color:#fff
    classDef audio fill:#45b7d1,stroke:#333,stroke-width:2px,color:#fff
    classDef analytics fill:#f9ca24,stroke:#333,stroke-width:2px,color:#333
    classDef storage fill:#6c5ce7,stroke:#333,stroke-width:2px,color:#fff
    classDef security fill:#fd79a8,stroke:#333,stroke-width:2px,color:#fff
    classDef success fill:#00b894,stroke:#333,stroke-width:2px,color:#fff
    
    class C1,I1,K aiModel
    class C2,N1,N2,N3,N4,N5 vision
    class C3,H2,I2,L2 audio
    class S1,S2,T1,T2,U1,U2 analytics
    class V,V1,V2,V3,V4 storage
    class AA,AA1,AA2,AA3,AA4 security
    class CC success