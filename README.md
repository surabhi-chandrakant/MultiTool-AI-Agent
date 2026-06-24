
# 🤖 MultiTool-AI-Agent

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/surabhi-chandrakant/MultiTool-AI-Agent?style=social)](https://github.com/surabhi-chandrakant/MultiTool-AI-Agent)

**An intelligent, all-in-one AI agent that combines web search, mathematical calculations, unit conversions, and AI-powered document querying into a beautiful web interface.**

[Features](#-features) • [Demo](#-live-demo) • [Installation](#-installation) • [Usage](#-usage) • [Commands](#-commands) • [Deployment](#-deployment) • [Contributing](#-contributing)

</div>

---


</div>

---

## ✨ Features

### 🔍 **Web Search**
- Search the entire web using DuckDuckGo API
- Get instant results with titles, links, and snippets
- **No API key required** - 100% free
- Smart query routing - just type naturally

### 📰 **News Search**
- Get latest news on any topic
- Shows publication date and source
- Curated results from trusted news outlets
- Real-time news aggregation

### 🧮 **Smart Calculator**
- Complex mathematical expressions
- Support for 20+ mathematical functions:
  - Trigonometry: `sin()`, `cos()`, `tan()`
  - Logarithms: `log()`, `log10()`, `log2()`
  - Constants: `pi`, `e`
  - Advanced: `sqrt()`, `factorial()`, `gcd()`
- Safe evaluation - no code injection
- Natural language math detection

### 📏 **Unit Converter**
- **Length**: mm, cm, m, km, inches, feet, yards, miles
- **Weight**: mg, g, kg, oz, pounds
- **Temperature**: Celsius, Fahrenheit, Kelvin
- Natural language conversion: *"convert 100 km to miles"*
- Instant, accurate results

### 📚 **Document Query System**
- Upload documents via drag & drop or file picker
- AI-powered semantic search using Sentence Transformers
- Supported formats:
  - 📄 Text files (.txt)
  - 📕 PDF documents (.pdf)
  - 📘 Word documents (.docx)
  - 📊 CSV files (.csv)
  - 💻 Code files (.py, .js, .html, .css, .json, .xml)
  - 📝 Markdown (.md)
- Automatic text chunking for better retrieval
- Persistent storage - documents survive restarts
- View document statistics and manage files

### 🎨 **Professional UI/UX**
- Modern, responsive design
- Dark sidebar for document management
- Drag & drop file upload
- Toast notifications
- Quick action buttons
- Mobile-friendly interface
- Loading animations
- Clear chat history option

---

## 🚀 Live Demo

<div align="center">

### [🔗 Try it Live!](https://huggingface.co/spaces/surabhic/multitool-ai-agent)

*Deployed on HF Spaces - Always available, no installation needed!*

</div>

---

## 📋 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 2GB RAM (for the AI model)
- Internet connection (for web search)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/surabhi-chandrakant/MultiTool-AI-Agent.git
cd MultiTool-AI-Agent

# 2. Create virtual environment (optional but recommended)
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python web_app.py
```

### Access the App
Open your browser and navigate to:
```
http://localhost:5000
```

> **Note:** First run will download the sentence transformer model (~90MB). This happens only once.

---

## 💻 Usage

### Basic Commands

Simply type your command in the chat input and press Enter or click Send. The AI agent automatically understands your intent!

#### **Web Search**
```
search latest AI breakthroughs 2024
search Python programming tutorials
search best restaurants in New York
```
*Just type "search" followed by your query*

#### **News Search**
```
news technology
news climate change
news space exploration
```
*Get the latest news on any topic*

#### **Calculator**
```
calc 15 * 3 + 27
calc sqrt(256)
calc sin(45) + cos(45)
calc factorial(10)
```
*Supports complex mathematical expressions*

#### **Unit Conversion**
```
convert 100 km to miles
convert 25 celsius to fahrenheit
convert 10 kg to pounds
convert 500 ml to oz
```
*Convert between various units naturally*

#### **Document Management**
```
add_doc path/to/your/file.pdf
query_doc machine learning concepts
doc_stats
list_docs
remove_doc filename.pdf
clear_all
```
*Upload, search, and manage your documents*

### Smart Detection

The agent automatically detects your intent:
- Just type `2 + 2` → It calculates
- Type `100 km to miles` → It converts
- Type anything else → It searches the web

---

## 📚 Complete Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `search <query>` | Search the web | `search latest AI news` |
| `news <topic>` | Get latest news | `news technology` |
| `calc <expression>` | Calculate math | `calc 15 * 3 + 27` |
| `convert <value> <from> to <to>` | Convert units | `convert 100 km to miles` |
| `add_doc <filepath>` | Add document | `add_doc report.pdf` |
| `query_doc <text>` | Search documents | `query_doc AI trends` |
| `doc_stats` | Document statistics | `doc_stats` |
| `list_docs` | List all documents | `list_docs` |
| `remove_doc <filename>` | Remove document | `remove_doc old.pdf` |
| `clear_all` | Clear all documents | `clear_all` |
| `help` | Show all commands | `help` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│                   (HTML/CSS/JavaScript)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Flask Web Server                      │
│                  (REST API Endpoints)                    │
└──────────┬──────────┬──────────┬────────────────────────┘
           │          │          │
           ▼          ▼          ▼
    ┌──────────┐ ┌────────┐ ┌──────────────┐
    │   Web    │ │  Calc  │ │   Document   │
    │ Search   │ │ Engine │ │    Store     │
    │(DuckDuckGo)│ │(Math)  │ │(Embeddings) │
    └──────────┘ └────────┘ └──────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.10 | Core application logic |
| **Web Framework** | Flask 3.0 | REST API and server |
| **AI Model** | Sentence Transformers (all-MiniLM-L6-v2) | Document embeddings |
| **Web Search** | DuckDuckGo Search API | Free web search |
| **Document Processing** | PyPDF2, python-docx | Multi-format support |
| **Frontend** | HTML5, CSS3, JavaScript | Responsive UI |
| **Styling** | Custom CSS with CSS Grid/Flexbox | Modern design |

---

## 🗂️ Project Structure

```
MultiTool-AI-Agent/
│
├── web_app.py                 # Main Flask application
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
├── Procfile                  # Deployment configuration
├── Dockerfile                # Docker container configuration
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── README.md                # Project documentation
│
├── screenshots/             # Application screenshots
│   ├── main-interface.png
│   ├── search-results.png
│   ├── document-upload.png
│   └── calculator.png
│
├── uploads/                 # Uploaded documents directory
│   └── (uploaded files)
│
└── document_store/          # Document embeddings storage
    └── store.pkl           # Serialized document data
```

---

## 🚀 Deployment

### Option 1: Render (Recommended - Free Tier)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

1. Fork this repository
2. Go to [render.com](https://render.com)
3. Click "New Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
6. Click "Create Web Service"

### Option 2: Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Fork this repository
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python and deploys

### Option 3: Hugging Face Spaces

[![Deploy on HF Spaces](https://img.shields.io/badge/Deploy-HuggingFace-FF9D00)](https://huggingface.co/spaces)

1. Create a new Space at huggingface.co/spaces
2. Choose "Docker" SDK
3. Upload files or connect GitHub
4. App runs automatically

### Option 4: Docker Deployment

```bash
# Build Docker image
docker build -t multitool-ai-agent .

# Run container
docker run -d -p 5000:5000 --name ai-agent \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/document_store:/app/document_store \
  multitool-ai-agent

# Access at http://localhost:5000
```

### Option 5: PythonAnywhere (Free)

```bash
# 1. Sign up at pythonanywhere.com
# 2. Upload files via Web tab
# 3. Create new Web App (Flask)
# 4. Set WSGI configuration to point to web_app.py
```

---

## ⚙️ Configuration

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Application port |
| `FLASK_ENV` | `production` | Flask environment |
| `MAX_CONTENT_LENGTH` | `16777216` | Max file upload size (16MB) |
| `SECRET_KEY` | Auto-generated | Flask secret key |

### Supported File Types

You can modify allowed file types in `web_app.py`:
```python
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'md', 'py', 'js', 'html', 'css', 'json', 'xml'}
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Module not found** | Run `pip install -r requirements.txt` |
| **Model download fails** | Check internet connection, retry |
| **Port already in use** | Change port: `app.run(port=8080)` |
| **Search not working** | Check internet connection |
| **Document upload fails** | Check file format and size (<16MB) |
| **Slow first load** | Model downloads ~90MB on first run |

### Getting Help

- 📖 Check [documentation](docs/)
- 🐛 Report bugs in [Issues](https://github.com/surabhi-chandrakant/MultiTool-AI-Agent/issues)
- 💬 Ask questions in [Discussions](https://github.com/surabhi-chandrakant/MultiTool-AI-Agent/discussions)
- ✉️ Email: [your-email@example.com]

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/MultiTool-AI-Agent.git
cd MultiTool-AI-Agent

# Create branch
git checkout -b feature/amazing-feature

# Make changes and commit
git add .
git commit -m "Add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

### Code Style
- Follow PEP 8 guidelines
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation if needed

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Search Response Time** | < 2 seconds |
| **Document Processing** | ~1 sec per page |
| **Model Size** | ~90MB |
| **Memory Usage** | ~500MB RAM |
| **Concurrent Users** | 10-20 (single instance) |

---

## 🛣️ Roadmap

### Upcoming Features
- [ ] User authentication system
- [ ] Chat history persistence
- [ ] Export search results
- [ ] Dark mode toggle
- [ ] Voice input support
- [ ] API key management
- [ ] Multi-language support
- [ ] Advanced document analytics
- [ ] Custom embedding models
- [ ] Real-time collaboration

### Completed
- [x] Web search integration
- [x] Calculator & converter
- [x] Document upload & query
- [x] Drag & drop interface
- [x] Document management panel
- [x] Mobile responsive design
- [x] Docker support

---

## ⚠️ Limitations

- **DuckDuckGo Rate Limits**: Heavy search usage may be rate-limited
- **Document Size**: Large documents (>50MB) may be slow to process
- **Memory Usage**: AI model requires ~500MB RAM
- **Single User**: Designed for single-user or small team use
- **No Authentication**: Currently no user login system
- **Embedding Quality**: Uses lightweight model for speed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Surabhi Chandrakant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Libraries & APIs
- [Sentence Transformers](https://www.sbert.net/) - Document embeddings
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) - Free web search
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyPDF2](https://pypi.org/project/PyPDF2/) - PDF processing
- [python-docx](https://python-docx.readthedocs.io/) - Word document processing

### Inspiration
- OpenAI's ChatGPT for multi-tool agent concept
- Perplexity AI for search + AI integration
- Various open-source AI projects

---

## 📞 Contact

<div align="center">

**Surabhi Chandrakant**

[![GitHub](https://img.shields.io/badge/GitHub-surabhi--chandrakant-181717?logo=github)](https://github.com/surabhi-chandrakant)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/your-profile)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:your-email@example.com)

**Project Link:** [https://github.com/surabhi-chandrakant/MultiTool-AI-Agent](https://github.com/surabhi-chandrakant/MultiTool-AI-Agent)

</div>

---

## ⭐ Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=surabhi-chandrakant/MultiTool-AI-Agent&type=Date)](https://star-history.com/#surabhi-chandrakant/MultiTool-AI-Agent&Date)

</div>

---

<div align="center">

### 💡 **If this project helped you, please give it a ⭐!**

**[⬆ Back to Top](#-multitool-ai-agent)**

</div>
```

---

## **📝 Additional Files Needed**

### **LICENSE file:**
```bash
# Create LICENSE file with MIT license text
```

### **CONTRIBUTING.md:**
```markdown
# Contributing to MultiTool-AI-Agent

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Code Style
- Follow PEP 8
- Add comments for complex logic
- Update README if needed

## Report Bugs
Open an issue with:
- Bug description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
```

---

## **⚡ Quick Setup Commands:**

```bash
# Navigate to your repo
cd MultiTool-AI-Agent

# Create necessary files
touch LICENSE
touch CONTRIBUTING.md
mkdir screenshots

# Add README
# (Copy the README content above)
nano README.md

# Stage and commit
git add README.md LICENSE CONTRIBUTING.md
git commit -m "📝 Add comprehensive README with full documentation"

# Push to GitHub
git push origin main
```

---


- ✅ **Deployment-Ready** - Multiple deployment options explained

The README will make your repository stand out and attract more users and contributors! 🌟
