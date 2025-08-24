"""
Database initialization script for NoteWise Quiz
Creates all necessary tables and seeds default data
"""

from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

# ─────────────── FLASK APP SETUP ───────────────
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///notewise_quiz.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'temporary-secret-key'

db = SQLAlchemy(app)

# ─────────────── MODELS ───────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20), nullable=False)  # student or teacher
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active_user = db.Column(db.Boolean, default=True)

    quiz_attempts = db.relationship('QuizAttempt', backref='user', lazy=True)
    quiz_sessions = db.relationship('QuizSession', backref='user', lazy=True)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, plaintext):
        self.password_hash = generate_password_hash(plaintext)

class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # pdf, image, docx, pptx
    quiz_raw = db.Column(db.Text)
    answers = db.Column(db.Text)
    score = db.Column(db.Integer)
    total = db.Column(db.Integer)
    quiz_timer = db.Column(db.Integer, default=0)
    time_taken = db.Column(db.Integer)
    completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def is_expired(self):
        return datetime.utcnow() > self.created_at + timedelta(days=3)

class QuizSession(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quiz_attempt_id = db.Column(db.Integer, db.ForeignKey('quiz_attempt.id'), nullable=False)
    current_question = db.Column(db.Integer, default=0)
    answers_data = db.Column(db.Text)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    time_remaining = db.Column(db.Integer)
    is_active = db.Column(db.Boolean, default=True)

# ─────────────── INITIALIZATION ───────────────
if __name__ == "__main__":
    with app.app_context():
        # Drop & recreate all tables
        db.drop_all()
        db.create_all()

        # Default admin user
        admin = User.query.filter_by(email='admin@notewise.com').first()
        if not admin:
            admin = User(
                email='admin@notewise.com',
                name='Admin User',
                role='teacher'
            )
            admin.password = 'admin123'
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created: admin@notewise.com / admin123")

        print("✅ Database initialized successfully: notewise_quiz.db")
        print("🚀 Run the app with: python app.py")
