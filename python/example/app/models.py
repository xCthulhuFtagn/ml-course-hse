from app import db

class User(db.Models):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(60), unique=True)
    password_hash = db.Column(db.String(128), nullable=False)
    users_jobs = db.relationship('UsersToJobs', backref='userstojobs', lazy=True)
    def __repr__(self):
        return f'User {self.id} {self.email}'
    
class UserToJobs(db.Models):
    id = db.Column(db.Integer, primary_key=True)
    cv = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('User.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('Job.id'), nullable=False)
    
class Jobs(db.Models):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(128), nullable=False)
    date_created = db.Column(db.DateTime)