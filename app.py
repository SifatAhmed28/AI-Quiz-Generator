"""AI-Quiz Generator – Flask 3 + Gemini 2.0"""
import os, re, tempfile, uuid, textwrap, logging, smtplib, json
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from flask import (Flask, request, send_file, session,
                   render_template, redirect, url_for, flash, jsonify)
import google.generativeai as genai
import PyPDF2
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, validators
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# ─────────────── env & logging ───────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s"
)
log = logging.getLogger("quiz-api")

app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database configuration
database_uri = os.getenv('DATABASE_URL', 'sqlite:///quiz.db')
if database_uri.startswith("postgres://"):
    database_uri = database_uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = True
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)

db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")
genai.configure(api_key=GEMINI_API_KEY)

# ─────────────── Models ───────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20), nullable=False)  # student or teacher
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    quiz_attempts = db.relationship('QuizAttempt', backref='user', lazy=True)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pdf_filename = db.Column(db.String(200), nullable=False)
    pdf_data = db.Column(db.LargeBinary)
    quiz_raw = db.Column(db.Text)
    answers = db.Column(db.Text)
    score = db.Column(db.Integer)
    total = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def is_expired(self):
        return datetime.now(timezone.utc) > self.created_at.replace(tzinfo=timezone.utc) + timedelta(days=3)

# ─────────────── Forms ───────────────
class ProfileForm(FlaskForm):
    name = StringField('Full Name', validators=[
        validators.DataRequired(),
        validators.Length(min=2, max=100)
    ])
    address = StringField('Address', validators=[
        validators.Length(max=200)
    ])
    phone = StringField('Phone Number', validators=[
        validators.Length(max=20)
    ])
    role = SelectField('Role', choices=[
        ('student', 'Student'), 
        ('teacher', 'Teacher')
    ])

class PasswordChangeForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[
        validators.DataRequired(),
        validators.Length(min=6)
    ])
    new_password = PasswordField('New Password', validators=[
        validators.DataRequired(),
        validators.Length(min=6),
        validators.EqualTo('confirm_password', message='Passwords must match')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        validators.DataRequired()
    ])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─────────────── CLEANUP FUNCTION & SCHEDULER ───────────────
def cleanup_old_data():
    with app.app_context():
        three_days_ago = datetime.utcnow() - timedelta(days=3)
        old_attempts = QuizAttempt.query.filter(QuizAttempt.created_at < three_days_ago).all()
        for attempt in old_attempts:
            db.session.delete(attempt)
        db.session.commit()
        log.info(f"Cleaned up {len(old_attempts)} old quiz attempts")

scheduler = BackgroundScheduler()
scheduler.add_job(func=cleanup_old_data, trigger='interval', days=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

# ─────────────── helpers ───────────────
def extract_text(pdf_data: bytes) -> str:
    out = []
    file_stream = BytesIO(pdf_data)
    rdr = PyPDF2.PdfReader(file_stream)
    for p in rdr.pages:
        txt = p.extract_text() or ""
        out.append(txt)
    return "\n".join(out)

def safe_generate(prompt: str, *, max_tokens: int = 2048) -> str | None:
    model = genai.GenerativeModel("gemini-2.0-flash")
    cfg = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=max_tokens,
        temperature=0.7
    )
    try:
        r = model.generate_content(prompt, generation_config=cfg)
    except Exception as e:
        log.error("Gemini error %s", e)
        return None
    if not r.candidates:
        return None
    parts = r.candidates[0].content.parts
    return "".join(p.text or "" for p in parts).strip() or None

def make_prompt(content: str, n: int, qtype: str) -> str:
    kind = "multiple-choice" if qtype == "multiple-choice" else "true/false"
    return textwrap.dedent(f"""
        Generate a {kind} quiz with exactly {n} questions.
        1. Start each question with "Q1.", "Q2.", …
        2. Multiple-choice: give 4 options A)–D)
        3. True/False: end with "(True/False)"
        4. After each question add "Answer:" and the correct answer.
        5. No explanations.

        Source text:

        {content[:15000]}
    """)

def parse_quiz(raw: str):
    items = []
    cur = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^Q\d+\.", s):
            if cur:
                items.append(cur)
            cur = {"question": s.split(".", 1)[1].strip(),
                   "options": [], "answer": ""}
        elif re.match(r"^[A-D]\)", s):
            cur["options"].append(s)
        elif s.startswith("Answer:"):
            cur["answer"] = s.split(":", 1)[1].strip()
    if cur:
        items.append(cur)
    return items

def pdf_from_text(text: str, title: str) -> str:
    path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 72
    c.setFont("Helvetica", 12)
    c.drawString(72, y, title)
    y -= 24
    for line in text.splitlines():
        for seg in textwrap.wrap(line, 95):
            if y < 72:
                c.showPage()
                y = height - 72
            c.drawString(72, y, seg)
            y -= 14
    c.save()
    return path

def email_quiz(to: str, files: list[str]) -> bool:
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to
    msg["Subject"] = "Your generated quiz"
    msg.attach(MIMEText("Quiz + answers attached.", "plain"))
    for fp in files:
        with open(fp, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(fp))
            part["Content-Disposition"] = f'attachment; filename="{os.path.basename(fp)}"'
            msg.attach(part)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(EMAIL_USER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_USER, to, msg.as_string())
        return True
    except Exception as e:
        log.error("SMTP error %s", e)
        return False

# Custom template filter for filename extraction
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path) if path else ""

# ─────────────── routes ───────────────
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('spa_root'))
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        address = request.form.get('address')
        phone = request.form.get('phone')
        role = request.form.get('role')
        password = request.form.get('password')
        if not all([email, name, role, password]):
            flash('Please fill in all required fields', 'danger')
            return redirect(url_for('signup'))
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists', 'danger')
            return redirect(url_for('signup'))
        # Create new user
        new_user = User(
            email=email,
            name=name,
            address=address,
            phone=phone,
            role=role
        )
        new_user.password = password  # uses the setter
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('spa_root'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = bool(request.form.get('remember'))
        user = User.query.filter_by(email=email).first()
        if user and user.verify_password(password):
            login_user(user, remember=remember)
            return redirect(url_for('spa_root'))
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Main pages
@app.get("/")
@login_required
def spa_root():
    return render_template("index.html")

@app.get('/quiz')
@login_required
def quiz_page():
    return render_template('quiz.html')

@app.get('/results')
@login_required
def results_page():
    return render_template('results.html')

# Profile page
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_page():
    profile_form = ProfileForm()
    password_form = PasswordChangeForm()
    
    # Get user's quiz history (last 10 attempts)
    quiz_history = QuizAttempt.query.filter_by(user_id=current_user.id)\
        .order_by(QuizAttempt.created_at.desc()).limit(10).all()
    
    # Handle profile update
    if profile_form.validate_on_submit():
        current_user.name = profile_form.name.data
        current_user.address = profile_form.address.data
        current_user.phone = profile_form.phone.data
        current_user.role = profile_form.role.data
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile_page'))
    
    # Handle password change
    if password_form.validate_on_submit():
        if current_user.verify_password(password_form.current_password.data):
            current_user.password = password_form.new_password.data
            db.session.commit()
            flash('Password changed successfully!', 'success')
        else:
            flash('Current password is incorrect', 'danger')
        return redirect(url_for('profile_page'))
    
    # Populate form with current data
    if not profile_form.is_submitted():
        profile_form.name.data = current_user.name
        profile_form.address.data = current_user.address
        profile_form.phone.data = current_user.phone
        profile_form.role.data = current_user.role
    
    return render_template('profile.html', 
                           profile_form=profile_form,
                           password_form=password_form,
                           quiz_history=quiz_history)

# API routes
@app.post("/api/upload")
@login_required
def api_upload():
    # Check file size (25MB max)
    if request.content_length > 25 * 1024 * 1024:
        return {"error": "File size exceeds 25MB limit"}, 400
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return {"error": "pdf only"}, 400
    
    # Read PDF content
    pdf_data = f.read()
    
    # Create quiz attempt record
    attempt = QuizAttempt(
        user_id=current_user.id,
        pdf_filename=f.filename,
        pdf_data=pdf_data
    )
    db.session.add(attempt)
    db.session.commit()
    session["quiz_attempt_id"] = attempt.id
    return {"ok": True, "filename": f.filename}

@app.post("/api/generate")
@login_required
def api_generate():
    data = request.json or {}
    attempt_id = session.get("quiz_attempt_id")
    
    if not attempt_id:
        return {"error": "no file"}, 400
        
    attempt = QuizAttempt.query.get(attempt_id)
    if not attempt:
        return {"error": "invalid attempt"}, 400
        
    txt = extract_text(attempt.pdf_data)
    if len(txt) < 50:
        return {"error": "empty text"}, 400
        
    prompt = make_prompt(txt, int(data.get("num_questions", 5)),
                        data.get("question_type", "multiple-choice"))
    raw = safe_generate(prompt)
    if not raw:
        return {"error": "model failed"}, 500
        
    quiz = parse_quiz(raw)
    answers = "\n".join(f"Q{i+1}. {q['answer']}" for i, q in enumerate(quiz))
    
    # Update quiz attempt with generated content
    attempt.quiz_raw = raw
    attempt.answers = answers
    db.session.commit()
    
    session.update({"quiz_raw": raw, "quiz": quiz, "answers": answers})
    return {"quiz": quiz}

@app.post("/api/submit")
@login_required
def api_submit():
    if "quiz" not in session:
        return {"error": "no quiz"}, 400
    user = request.json or {}
    score = 0
    res = []
    for i, q in enumerate(session["quiz"]):
        ans = (user.get(str(i), "") or "")[:1].upper()
        ok = ans == q["answer"][:1].upper()
        if ok:
            score += 1
        res.append({"q": q["question"], "user": ans,
                    "correct": q["answer"], "ok": ok})
    # Save results to quiz attempt
    attempt_id = session.get("quiz_attempt_id")
    if attempt_id:
        attempt = QuizAttempt.query.get(attempt_id)
        if attempt:
            attempt.score = score
            attempt.total = len(res)
            db.session.commit()
    session["results"] = res
    session["score"] = score
    return {"score": score, "total": len(res), "results": res}

@app.get("/api/download/<kind>.pdf")
@login_required
def api_download(kind: str):
    if kind == "quiz":
        text, title = session.get("quiz_raw"), "Quiz"
    elif kind == "answers":
        text, title = session.get("answers"), "Answer Key"
    else:
        return {"error": "bad kind"}, 400
    if not text:
        return {"error": "nothing to download"}, 400
    path = pdf_from_text(text, title)
    return send_file(path, as_attachment=True,
                     download_name=f"{kind}.pdf",
                     mimetype="application/pdf")

@app.post("/api/share")
@login_required
def api_share():
    data = request.json or {}
    email = data.get("email", "")
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return {"error": "bad email"}, 400
    if "quiz_raw" not in session:
        return {"error": "no quiz"}, 400
    qpdf = pdf_from_text(session["quiz_raw"], "Quiz")
    apdf = pdf_from_text(session["answers"], "Answer Key")
    ok = email_quiz(email, [qpdf, apdf])
    return {"sent": ok}

# ─────────────── main ───────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    log.info("➡ http://127.0.0.1:5001")
    app.run(debug=True, port=5001, use_reloader=False)