# NoteWise Quiz 📚

An intelligent quiz generation platform that transforms your study materials into interactive quizzes using AI. Upload your class notes in PDF, PPTX, or DOCX format and generate customized quizzes instantly.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **📄 Multi-Format Support**: Upload PDF, PPTX, and DOCX files
- **🤖 AI-Powered Generation**: Utilizes Google Gemini 2.0 for intelligent quiz creation
- **⚡ Customizable Quizzes**: Set the number of questions and time limits
- **🏆 Live Quiz Experience**: Real-time quiz taking with instant feedback
- **📊 Performance Tracking**: View your quiz history and scores
- **🔐 User Authentication**: Secure login and registration system
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Technology Stack

- **Backend**: Flask 3.1+ (Python web framework)
- **AI Engine**: Google Gemini 2.0 API
- **Authentication**: Flask-Login for user management
- **Database**: SQLite (configurable for production databases)
- **Frontend**: HTML5, CSS3, JavaScript
- **File Processing**: PDF, PPTX, DOCX parsing capabilities

## 📁 Project Structure

```
quiz-app/
├── app.py              # Main Flask application entry point
├── config.py           # Configuration settings (development/production)
├── models.py           # Database models (User, Quiz, etc.)
├── auth.py             # Authentication routes and logic
├── requirements.txt    # Python dependencies
├── Procfile           # Deployment configuration for Heroku
├── .env               # Environment variables (API keys, secrets)
├── .gitignore         # Git ignore rules
├── static/            # Static assets
│   ├── css/          # Stylesheets
│   └── js/           # JavaScript files
└── templates/         # Jinja2 HTML templates
    ├── base.html     # Base template with common layout
    ├── index.html    # Home page
    ├── login.html    # User login form
    └── quiz.html     # Quiz interface
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SifatAhmed28/AI-Quiz-Generator.git
   cd notewise-quiz
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your configurations
   GEMINI_API_KEY=your_gemini_api_key_here
   SECRET_KEY=your_secret_key_here
   FLASK_ENV=development
   ```

5. **Initialize the database**
   ```bash
   python -c "from app import app, db; app.app_context().push(); db.create_all()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:5000`

## 📝 Environment Variables

Create a `.env` file in the root directory:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_flask_secret_key

# Optional
FLASK_ENV=development
DATABASE_URL=sqlite:///quiz_app.db
MAX_CONTENT_LENGTH=16777216  # 16MB file upload limit
```

## 🎯 Usage

### 1. **Register/Login**
   - Create a new account or log in with existing credentials
   - Secure authentication powered by Flask-Login

### 2. **Upload Study Materials**
   - Navigate to the upload section
   - Select your PDF, PPTX, or DOCX files
   - Supported file formats:
     - 📄 PDF documents
     - 📊 PowerPoint presentations (.pptx)
     - 📃 Word documents (.docx)

### 3. **Generate Quiz**
   - Choose the number of questions (1-50)
   - Set time limit for the quiz
   - Click "Generate Quiz" to create your personalized quiz

### 4. **Take Quiz**
   - Answer questions in real-time
   - Track your progress with the timer
   - Submit for instant results

### 5. **View History**
   - Access your quiz history from the dashboard
   - Review previous scores and performance
   - Track your learning progress over time

## 🔧 Configuration

### Flask Configuration (`config.py`)

```python
import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///quiz_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'
```

### Gemini 2.0 Integration

The application leverages Google's Gemini 2.0 API for:
- **Document Processing**: Extract content from uploaded files
- **Quiz Generation**: Create relevant questions based on content
- **Content Analysis**: Understand document context and structure

## 🧪 Testing

```bash
# Install testing dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 🚀 Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

4. **Set environment variables**
   ```bash
   heroku config:set GEMINI_API_KEY=your_api_key
   heroku config:set SECRET_KEY=your_secret_key
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| POST | `/login` | User authentication |
| POST | `/register` | User registration |
| POST | `/upload` | File upload for quiz generation |
| GET | `/quiz/<id>` | Take specific quiz |
| GET | `/history` | View quiz history |
| POST | `/logout` | User logout |

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Write comprehensive tests for new features
- Update documentation for API changes
- Use meaningful commit messages

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages include:

```txt
Flask>=3.0.0
flask-login>=0.6.0
flask-sqlalchemy>=3.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
werkzeug>=3.0.0
PyPDF2>=3.0.0
python-pptx>=0.6.0
python-docx>=0.8.0
```

## 🐛 Troubleshooting

### Common Issues

**1. Gemini API Key Error**
```
Error: Invalid API key
Solution: Verify your GEMINI_API_KEY in .env file
```

**2. File Upload Issues**
```
Error: File too large
Solution: Check MAX_CONTENT_LENGTH in config.py
```

**3. Database Connection Error**
```
Error: Database not found
Solution: Run database initialization command
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sifat Ahmed** - *Initial work* - [YourGitHub](https://github.com/SifatAhmed28)

## 🙏 Acknowledgments

- Google Gemini 2.0 API for AI capabilities
- Flask community for the excellent framework
- Contributors and testers

## 📞 Support

- **Documentation**: [Link to docs]
- **Issues**: [GitHub Issues](https://github.com/SifatAhmed28/AI-Quiz-Generator/issues)
- **Email**: sifata353@gmail.com

## 🔄 Changelog

### Version 1.0.0 (Current)
- Initial release
- Basic quiz generation functionality
- User authentication system
- File upload support for PDF, PPTX, DOCX
- Quiz history tracking

### Upcoming Features
- [ ] Multiple choice question types
- [ ] Collaborative quizzes
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Integration with LMS platforms

---

**Made with ❤️ by Sifat Ahmed & Tanvir Rahat**
